"""Apply the existing frozen construction tile policy to all six new/parent cases."""
import io
import tempfile
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[3]
app = modal.App("coveragecv-research-tiling")


def build_image():
    from coveragecv.training.cloud_budget import require_cloud_execution
    from coveragecv.training.modal_research_evaluation import build_image as evaluation_image
    require_cloud_execution("coveragecv-research-tiling")
    return (evaluation_image()
            .add_local_file(str(Path(__file__)), "/opt/coveragecv/training/modal_research_tiling.py", copy=True)
            .add_local_file(str(ROOT / "artifacts/research_v2/tiling_protocol.json"), "/tiling_protocol.json", copy=True))


image = build_image() if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu="L40S", cpu=(2, 2), memory=(8192, 16384), timeout=240,
              max_containers=2, min_containers=0, scaledown_window=2, retries=0, include_source=False)
def evaluate(case: str, checkpoint: bytes, baseline: dict, protocol_sha256: str):
    from coveragecv.training.evaluate import score_predictions
    from coveragecv.training.tiled import PROTOCOL, evaluate_tiled
    if file_digest(Path("/tiling_protocol.json")) != protocol_sha256:
        raise ValueError("Tiling protocol differs from caller")
    protocol = read_json(Path("/tiling_protocol.json"))
    if case not in protocol["cases"] or PROTOCOL != protocol["tile_policy"]:
        raise ValueError("Case or tile policy differs from the frozen declaration")
    if file_digest(Path(__file__)) != protocol["wrapper_sha256"]:
        raise ValueError("Tiling wrapper source changed")
    evaluation = read_json(Path("/evaluation_protocol.json"))
    if file_digest(Path("/evaluation_protocol.json")) != protocol["evaluation_protocol_sha256"]:
        raise ValueError("Base evaluator protocol changed")
    for name, expected in evaluation["source_files"].items():
        if file_digest(safe_child(Path("/opt/coveragecv"), name)) != expected:
            raise ValueError("Frozen source changed")
    payload = Path("/references/construction.zip")
    if file_digest(payload) != evaluation["references"]["construction"]["payload_sha256"]:
        raise ValueError("Reference archive changed")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "detector.pt"
        model.write_bytes(checkpoint)
        if file_digest(model) != protocol["cases"][case]["checkpoint_sha256"]:
            raise ValueError("Unexpected checkpoint")
        if (baseline["checkpoint_sha256"] != file_digest(model) or baseline["device"] != "cuda"
                or baseline["evaluation_protocol_sha256"] != protocol["evaluation_protocol_sha256"]):
            raise ValueError("Single-pass baseline differs from the declared cloud evaluation")
        write_json(root / "baseline.json", baseline)
        bundle = root / "reference"
        with zipfile.ZipFile(payload) as archive:
            for name in archive.namelist():
                target = safe_child(bundle, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        output = root / "output"
        evaluate_tiled(model, bundle, output, baseline_evaluation=root / "baseline.json", device="cuda")
        full, tiled = read_json(output / "full_frame.json"), read_json(output / "tiled.json")
        selected = protocol["class_gate"]["tiled_category_ids"]
        predictions = ([row for row in full["predictions"] if row["category_id"] not in selected]
                       +[row for row in tiled["predictions"] if row["category_id"] in selected])
        reference = read_json(bundle / "splits/valid.coco.json")
        metrics = score_predictions(predictions, reference)
        expected = sum((tiled if cat["id"] in selected else full)["metrics"]["per_class_AP"][cat["name"]]
                       for cat in reference["categories"])/len(reference["categories"])
        if abs(expected-metrics["AP"]) > 1e-10:
            raise ValueError("Per-class routing audit failed")
        result = {**{k: v for k, v in full.items() if k not in ("predictions", "metrics", "timing")},
                  "metrics": metrics, "predictions": predictions, "timing": tiled["timing"],
                  "case": case, "class_gate": protocol["class_gate"], "model_variant": baseline["model_variant"],
                  "research_tiling_protocol_sha256": protocol_sha256, "per_class_routing_audit": True}
        write_json(output / "size_gated.json", result)
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.glob("*.json")):
                archive.write(path, path.name)
        return packed.getvalue()
