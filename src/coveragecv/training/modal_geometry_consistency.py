"""One frozen cloud-only geometry pilot with a TRAIN gate before validation."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/continuous/geometry"
app = modal.App("coveragecv-geometry-consistency")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-geometry-consistency")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(OUT / "requirements.txt"))
            .add_local_dir(str(OUT / "source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_file(str(OUT / "payload.zip"), "/payload.zip", copy=True)
            .add_local_file(str(OUT / "protocol.json"), "/protocol.json", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "2"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=600,
              max_containers=1, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def experiment(protocol_sha256: str):
    from coveragecv.training.geometry_consistency import evaluate_pair, train_gate
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
        proposals = read_json(root / "proposals.json")
        if (file_digest(root / "proposals.json") != protocol["proposals_sha256"]
                or proposals["binding"]["checkpoint_sha256"] != protocol["proposal_checkpoint_sha256"]):
            raise ValueError("Training proposals do not match the declared parent")
        gate = train_gate(root / "view", proposals, protocol["policy"])
        output = root / "output"
        write_json(output / "train_selection.json", gate)
        result = {"protocol_sha256": protocol_sha256, "status": "rejected_before_validation",
                  "gate": {k: v for k, v in gate.items() if k != "rows"}, "evaluations": {}, "failures": {}}
        if gate["accepted"]:
            from coveragecv.artifacts import verify
            if verify(root / "reference")["digest"] != protocol["reference_bundle_digest"]:
                raise ValueError("Evaluation reference mismatch")
            for case, expected in protocol["checkpoints"].items():
                checkpoint = root / "checkpoints" / f"{case}.pt"
                if file_digest(checkpoint) != expected["checkpoint_sha256"]:
                    raise ValueError("Checkpoint mismatch")
                try:
                    result["evaluations"][case] = evaluate_pair(checkpoint, root / "reference", output / case,
                                                               policy=protocol["policy"])
                except Exception as error:  # noqa: BLE001 -- preserve the paid gate and other comparisons
                    result["failures"][case] = {"error_type": type(error).__name__, "error": str(error)}
            result["status"] = "evaluated" if not result["failures"] else "evaluation_failed"
        write_json(output / "result.json", result)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*.json")):
                archive.write(path, path.relative_to(output).as_posix())
        return packed.getvalue()
