"""Inference-only runtime-resolution experiment on unchanged saved checkpoints."""
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/continuous/cross_resolution"
app = modal.App("coveragecv-cross-resolution")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-cross-resolution")
    built = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("libgl1", "libglib2.0-0")
             .pip_install_from_requirements(str(OUT / "requirements.txt"))
             .add_local_dir(str(OUT / "source/coveragecv"), "/opt/coveragecv", copy=True)
             .add_local_file(str(OUT / "payload.zip"), "/payload.zip", copy=True)
             .add_local_file(str(OUT / "protocol.json"), "/protocol.json", copy=True))
    for task in ("construction", "pawns"):
        built = built.add_local_file(str(ROOT / "artifacts/research_v2/references" / f"{task}.zip"),
                                     f"/references/{task}.zip", copy=True)
    return built.env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "2"})


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=160,
              max_containers=2, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def evaluate(case: str, protocol_sha256: str):
    from coveragecv.training.evaluate import evaluate_checkpoint
    if file_digest(Path("/protocol.json")) != protocol_sha256:
        raise ValueError("Protocol mismatch")
    protocol = read_json(Path("/protocol.json"))
    for name, expected in protocol["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Source snapshot mismatch")
    if case not in protocol["cases"]:
        raise ValueError("Case outside frozen matrix")
    spec = protocol["cases"][case]
    if file_digest(Path("/payload.zip")) != protocol["payload_sha256"]:
        raise ValueError("Checkpoint payload mismatch")
    reference = Path("/references") / f"{spec['task']}.zip"
    if file_digest(reference) != protocol["references"][spec["task"]]["payload_sha256"]:
        raise ValueError("Reference payload mismatch")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        checkpoint = root / "detector.pt"
        with zipfile.ZipFile("/payload.zip") as archive:
            checkpoint.write_bytes(archive.read(f"checkpoints/{spec['checkpoint']}.pt"))
        binding = protocol["checkpoints"][spec["checkpoint"]]
        if file_digest(checkpoint) != binding["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from original")
        bundle = root / "reference"
        with zipfile.ZipFile(reference) as archive:
            for name in archive.namelist():
                target = safe_child(bundle, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        result = evaluate_checkpoint(checkpoint, bundle, root / "evaluation.json", device="cuda",
                                     resolution_override=spec["resolution"])
        if result["checkpoint_resolution"] != binding["checkpoint_resolution"]:
            raise ValueError("Recorded training resolution differs from checkpoint")
        return {**result, "experiment_protocol_sha256": protocol_sha256, "case": case}
