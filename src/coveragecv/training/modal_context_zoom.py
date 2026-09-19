"""Frozen wide-context zoom pilot, with a fresh GPU training-only acceptance gate."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/continuous/context_zoom"
app = modal.App("coveragecv-context-zoom")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-context-zoom")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(OUT / "requirements.txt"))
            .add_local_dir(str(OUT / "source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_file(str(ROOT / "artifacts/continuous/geometry/payload.zip"), "/payload.zip", copy=True)
            .add_local_file(str(OUT / "protocol.json"), "/protocol.json", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "2"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=900,
              max_containers=1, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(protocol_sha256: str):
    from coveragecv.training.context_zoom import run
    if file_digest(Path("/protocol.json")) != protocol_sha256:
        raise ValueError("Protocol mismatch")
    protocol = read_json(Path("/protocol.json"))
    for name, expected in protocol["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Source snapshot mismatch")
    if file_digest(Path("/payload.zip")) != protocol["payload_sha256"]:
        raise ValueError("Payload mismatch")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with zipfile.ZipFile("/payload.zip") as archive:
            for name in archive.namelist():
                target = safe_child(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        for case, metadata in protocol["checkpoints"].items():
            if file_digest(root / "checkpoints" / f"{case}.pt") != metadata["checkpoint_sha256"]:
                raise ValueError("Checkpoint mismatch")
        output = root / "output"
        result = run(root, protocol, output)
        result["protocol_sha256"] = protocol_sha256
        write_json(output / "result.json", result)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*.json")):
                archive.write(path, path.relative_to(output).as_posix())
        return packed.getvalue()
