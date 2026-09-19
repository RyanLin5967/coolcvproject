"""Frozen coverage-aware stable query assignment pilot and independent GPU scoring."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import digest, file_digest, read_json, safe_child

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/continuous/stable_assignment"
app = modal.App("coveragecv-stable-assignment")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-stable-assignment")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(OUT / "requirements.txt"))
            .pip_install("pytest==9.1.1")
            .add_local_dir(str(OUT / "source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_file(str(OUT / "test_stable_assignment.py"), "/tests/test_stable_assignment.py", copy=True)
            .add_local_file(str(OUT / "inputs.zip"), "/inputs.zip", copy=True)
            .add_local_file(str(ROOT / "artifacts/research_v2/references/pawns.zip"), "/reference.zip", copy=True)
            .add_local_file(str(OUT / "protocol.json"), "/protocol.json", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


def verify_protocol(sha):
    if file_digest(Path("/protocol.json")) != sha:
        raise ValueError("Deployed protocol differs from request")
    protocol = read_json(Path("/protocol.json"))
    for name, expected in protocol["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Source snapshot differs from protocol")
    return protocol


def unpack(source, destination, expected):
    if file_digest(source) != expected:
        raise ValueError("Input archive differs from protocol")
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            target = safe_child(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def run_case(root, protocol, case, *, smoke=False):
    from coveragecv.training.research_v2 import train
    spec = dict(protocol["cases"][case])
    sampling = spec.pop("sampling")
    plan = protocol["exposure_plans"][sampling]
    if smoke:
        spec["steps"] = 2
        plan = {k: v for k, v in plan.items() if k != "digest"}
        plan["sample_ids"] = plan["sample_ids"][:8]
        plan = {**plan, "digest": digest(plan)}
    arm = spec["arm"]
    return train(root / arm / "view", root / arm / "parent", root / case,
                 **spec, exposure_plan=plan, device="cuda", max_seconds=100 if smoke else 1100)


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=1300,
              max_containers=6, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def train_case(case: str, protocol_sha256: str):
    protocol = verify_protocol(protocol_sha256)
    if case not in protocol["cases"]:
        raise ValueError("Case is outside frozen matrix")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        unpack(Path("/inputs.zip"), root, protocol["inputs_sha256"])
        run_case(root, protocol, case)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "detector.pt", "progress.json"):
                archive.write(root / case / name, name)
        return output.getvalue()


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=240,
              max_containers=1, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def smoke(protocol_sha256: str):
    protocol = verify_protocol(protocol_sha256)
    import subprocess
    import sys
    if file_digest(Path("/tests/test_stable_assignment.py")) != protocol["tests_sha256"]:
        raise ValueError("Cloud test source mismatch")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "/tests/test_stable_assignment.py"],
                           capture_output=True, text=True, timeout=90, check=False)
    if tests.returncode:
        return {"status": "failed", "protocol_sha256": protocol_sha256,
                "test_stdout": tests.stdout, "test_stderr": tests.stderr}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        unpack(Path("/inputs.zip"), root, protocol["inputs_sha256"])
        outcomes = {case: run_case(root, protocol, case, smoke=True)
                    for case in ("aware-both", "complete-both")}
        if outcomes["aware-both"]["exposure_plan_digest"] != outcomes["complete-both"]["exposure_plan_digest"]:
            raise ValueError("Partial/full image exposures differ")
        return {"status": "passed", "protocol_sha256": protocol_sha256, "runs": outcomes, "test_stdout": tests.stdout, "test_stderr": tests.stderr}


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=160,
              max_containers=2, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def evaluate(checkpoint: bytes, checkpoint_sha256: str, protocol_sha256: str):
    from coveragecv.training.evaluate import evaluate_checkpoint
    protocol = verify_protocol(protocol_sha256)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        model = root / "detector.pt"
        model.write_bytes(checkpoint)
        if file_digest(model) != checkpoint_sha256:
            raise ValueError("Checkpoint hash mismatch")
        unpack(Path("/reference.zip"), root / "reference", protocol["reference_payload_sha256"])
        result = evaluate_checkpoint(model, root / "reference", root / "evaluation.json", device="cuda")
        if result["bundle_digest"] != protocol["reference_bundle_digest"]:
            raise ValueError("Reference bundle mismatch")
        return {**result, "experiment_protocol_sha256": protocol_sha256}
