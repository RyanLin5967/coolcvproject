"""Frozen construction test confirmation and unchanged validation tile interaction."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/continuous/confirmation"
app = modal.App("coveragecv-scale-confirmation")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    require_cloud_execution("coveragecv-scale-confirmation")
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install_from_requirements(str(OUT / "requirements.txt"))
            .add_local_dir(str(OUT / "source/coveragecv"), "/opt/coveragecv", copy=True)
            .add_local_dir(str(OUT / "baselines"), "/baselines", copy=True)
            .add_local_file(str(ROOT / "artifacts/continuous/cross_resolution/payload.zip"), "/payload.zip", copy=True)
            .add_local_file(str(ROOT / "artifacts/research_v2/references/construction.zip"), "/reference.zip", copy=True)
            .add_local_file(str(OUT / "protocol.json"), "/protocol.json", copy=True)
            .env({"PYTHONPATH": "/opt", "OMP_NUM_THREADS": "2"}))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=180,
              max_containers=2, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def evaluate(case: str, protocol_sha256: str):
    from coveragecv.training.evaluate import evaluate_checkpoint, score_predictions
    from coveragecv.training.tiled import PROTOCOL, evaluate_tiled
    if file_digest(Path("/protocol.json")) != protocol_sha256:
        raise ValueError("Protocol mismatch")
    protocol = read_json(Path("/protocol.json"))
    if case not in protocol["cases"] or PROTOCOL != protocol["tile_policy"]:
        raise ValueError("Unknown case or modified tile policy")
    for name, expected in protocol["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Source snapshot mismatch")
    if (file_digest(Path("/payload.zip")) != protocol["payload_sha256"]
            or file_digest(Path("/reference.zip")) != protocol["reference"]["payload_sha256"]):
        raise ValueError("Model/reference archive mismatch")
    baseline = Path("/baselines") / f"{case}.json"
    if file_digest(baseline) != protocol["cases"][case]["baseline_sha256"]:
        raise ValueError("Validation baseline mismatch")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        checkpoint = root / "detector.pt"
        with zipfile.ZipFile("/payload.zip") as archive:
            checkpoint.write_bytes(archive.read(f"checkpoints/{case}.pt"))
        if file_digest(checkpoint) != protocol["cases"][case]["checkpoint_sha256"]:
            raise ValueError("Checkpoint mismatch")
        bundle = root / "reference"
        with zipfile.ZipFile("/reference.zip") as archive:
            for name in archive.namelist():
                target = safe_child(bundle, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        output = root / "output"
        test = evaluate_checkpoint(checkpoint, bundle, output / "test.json", split="test", device="cuda")
        receipt = {"case": case, "protocol_sha256": protocol_sha256,
                   "checkpoint_sha256": file_digest(checkpoint), "test_metrics": test["metrics"],
                   "bundle_digest": test["bundle_digest"], "status": "test_scored"}
        try:
            evaluate_tiled(checkpoint, bundle, output / "validation", baseline_evaluation=baseline, device="cuda")
            full = read_json(output / "validation/full_frame.json")
            tiled = read_json(output / "validation/tiled.json")
            ids = protocol["class_gate"]["tiled_category_ids"]
            predictions = ([row for row in full["predictions"] if row["category_id"] not in ids]
                           + [row for row in tiled["predictions"] if row["category_id"] in ids])
            reference = read_json(bundle / "splits/valid.coco.json")
            metrics = score_predictions(predictions, reference)
            expected = sum((tiled if cat["id"] in ids else full)["metrics"]["per_class_AP"][cat["name"]]
                           for cat in reference["categories"]) / len(reference["categories"])
            if abs(metrics["AP"]-expected) > 1e-10:
                raise ValueError("Class routing audit failed")
            result = {**{k: v for k, v in full.items() if k not in ("predictions", "metrics", "timing")},
                      "predictions": predictions, "metrics": metrics, "class_gate": protocol["class_gate"],
                      "timing": tiled["timing"], "case": case, "confirmation_protocol_sha256": protocol_sha256,
                      "per_class_routing_audit": True}
            write_json(output / "validation/size_gated.json", result)
            receipt.update(status="completed", validation_tiled_metrics=metrics)
        except Exception as error:  # noqa: BLE001 -- retain primary test evidence if secondary scoring fails
            receipt.update(status="secondary_failed", failure={"type": type(error).__name__, "error": str(error)})
        write_json(output / "receipt.json", receipt)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*.json")):
                archive.write(path, path.relative_to(output).as_posix())
        return packed.getvalue()
