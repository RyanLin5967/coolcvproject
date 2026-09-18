"""One credit-bounded foundation-model localization pilot, entirely on a cloud GPU."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[3]
MODEL = "facebook/sam2.1-hiera-large"
REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
app = modal.App("coveragecv-segment-refinement")


def cache_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION, local_dir="/sam2", allow_patterns=["*.json", "*.safetensors"])


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    from coveragecv.training.modal_research_evaluation import build_image as evaluation_image
    require_cloud_execution("coveragecv-segment-refinement")
    return (evaluation_image()
            .add_local_file(str(Path(__file__)), "/opt/coveragecv/training/modal_segment_refinement.py", copy=True)
            .add_local_file(str(ROOT / "src/coveragecv/training/segment_refinement.py"),
                            "/opt/coveragecv/training/segment_refinement.py", copy=True)
            .add_local_file(str(ROOT / "artifacts/segment_refinement/payload.zip"), "/pilot.zip", copy=True)
            .add_local_file(str(ROOT / "artifacts/segment_refinement/protocol.json"), "/pilot_protocol.json", copy=True)
            .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
            .run_function(cache_model, cpu=2, memory=4096, timeout=600))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=1100,
              max_containers=1, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(protocol_sha256: str):
    from coveragecv.training.segment_refinement import run
    if file_digest(Path("/pilot_protocol.json")) != protocol_sha256:
        raise ValueError("Pilot protocol mismatch")
    protocol = read_json(Path("/pilot_protocol.json"))
    for name, expected in protocol["pilot_source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv/training"), name)) != expected:
            raise ValueError("Pilot source mismatch")
    if file_digest(Path("/pilot.zip")) != protocol["payload_sha256"]:
        raise ValueError("Pilot payload mismatch")
    reference = read_json(Path("/evaluation_protocol.json"))
    if (file_digest(Path("/evaluation_protocol.json")) != protocol["evaluation_protocol_sha256"]
            or file_digest(Path("/references/all-pieces.zip")) != reference["references"]["all-pieces"]["payload_sha256"]):
        raise ValueError("Reference protocol or archive mismatch")
    for name, expected in reference["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Base source mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for payload, destination in ((Path("/pilot.zip"), root), (Path("/references/all-pieces.zip"), Path("/reference"))):
            with zipfile.ZipFile(payload) as archive:
                for name in archive.namelist():
                    target = safe_child(destination, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
        output = root / "output"
        result = run(root, protocol, output)
        result.update(protocol_sha256=protocol_sha256, foundation_model=MODEL, revision=REVISION,
                      weight_files={p.name: file_digest(p) for p in Path("/sam2").glob("*.safetensors")})
        write_json(output / "result.json", result)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.glob("*.json")):
                archive.write(path, path.name)
        return packed.getvalue()
