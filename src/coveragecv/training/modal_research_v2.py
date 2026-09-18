"""Isolated, credit-bounded ontology and localization training; scoring stays local."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child

ROOT = Path(__file__).resolve().parents[3]
app = modal.App("coveragecv-research-v2")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-research-v2")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(ROOT / "cloud-requirements.txt"))
            .add_local_dir(str(ROOT / "artifacts/research_v2/source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_dir(str(ROOT / "artifacts/research_v2/inputs"), "/input", copy=True)
            .add_local_file(str(ROOT / "artifacts/research_v2/protocol.json"), "/protocol.json", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(4, 4), memory=(8192, 24576), timeout=1600,
              max_containers=8, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(case: str, protocol_sha256: str):
    from coveragecv.training.research_v2 import train
    if file_digest(Path("/protocol.json")) != protocol_sha256:
        raise ValueError("Deployed protocol differs from caller")
    protocol = read_json(Path("/protocol.json"))
    if case not in protocol["cases"]:
        raise ValueError("Case is outside the fixed experiment matrix")
    spec = dict(protocol["cases"][case])
    task = spec.pop("task")
    payload = Path("/input") / f"{task}.zip"
    if file_digest(payload) != protocol["datasets"][task]["payload_sha256"]:
        raise ValueError("Frozen learner/parent archive was changed")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with zipfile.ZipFile(payload) as archive:
            for name in archive.namelist():
                target = safe_child(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        arm = spec["arm"]
        output = root / "output"
        train(root / arm / "view", root / arm / "parent", output, **spec, device="cuda", max_seconds=1300)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("run.json", "detector.pt", "progress.json"):
                archive.write(output / name, name)
        return packed.getvalue()
