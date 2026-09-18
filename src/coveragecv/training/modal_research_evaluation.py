"""GPU evaluation isolated from training, with hash-bound unchanged references."""
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child

ROOT = Path(__file__).resolve().parents[3]
app = modal.App("coveragecv-research-evaluation")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-research-evaluation")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(ROOT / "cloud-requirements.txt"))
            .add_local_dir(str(ROOT / "artifacts/research_v2/evaluation-source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_dir(str(ROOT / "artifacts/research_v2/references"), "/references", copy=True)
            .add_local_file(str(ROOT / "artifacts/research_v2/evaluation_protocol.json"), "/evaluation_protocol.json",
                            copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "2"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=240,
              max_containers=2, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def evaluate(task: str, checkpoint: bytes, checkpoint_sha256: str, protocol_sha256: str):
    from coveragecv.training.evaluate import evaluate_checkpoint
    if file_digest(Path("/evaluation_protocol.json")) != protocol_sha256:
        raise ValueError("Evaluation protocol differs from caller")
    protocol = read_json(Path("/evaluation_protocol.json"))
    if task not in protocol["references"]:
        raise ValueError("Unknown evaluation dataset")
    for name, expected in protocol["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Scoring source differs from the frozen snapshot")
    payload = Path("/references") / f"{task}.zip"
    if file_digest(payload) != protocol["references"][task]["payload_sha256"]:
        raise ValueError("Reference payload was changed")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "detector.pt"
        model.write_bytes(checkpoint)
        if file_digest(model) != checkpoint_sha256:
            raise ValueError("Checkpoint bytes differ from the saved training receipt")
        bundle = root / "reference"
        with zipfile.ZipFile(payload) as archive:
            for name in archive.namelist():
                target = safe_child(bundle, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        result = evaluate_checkpoint(model, bundle, root / "evaluation.json", device="cuda")
        if result["bundle_digest"] != protocol["references"][task]["bundle_digest"]:
            raise ValueError("Scorer used a different reference bundle")
        result["evaluation_protocol_sha256"] = protocol_sha256
        return result
