"""Fit on observed train proposals, freeze the model, then score unchanged validation."""
import time
from pathlib import Path

import numpy as np
import torch

from coveragecv.artifacts import file_digest, read_json, verify, write_json
from coveragecv.training.evaluate import score_predictions
from coveragecv.training.localization_cascade import (
    PROTOCOL,
    FeatureExtractor,
    apply_corrections,
    matched_proposals,
    predict_ridge,
    train_cascade,
    training_holdout,
)
from coveragecv.training.refiner import _save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/localization_v2/all-pieces/aware"
SPEC = read_json(ROOT / "artifacts/full_chess/experiment.json")
VIEW = Path(SPEC["partial_view"])
ENCODER = ROOT / "artifacts/refinement/all-pieces/20260917/partial-human/refiner.pt"


def main():
    torch.set_num_threads(4)
    rows, images, manifest, classes = matched_proposals(VIEW, read_json(OUT / "proposals.json"))
    protocol = {**PROTOCOL, "view_digest": manifest["digest"], "encoder_sha256": file_digest(ENCODER),
                "proposals_sha256": file_digest(OUT / "proposals.json"),
                "source_sha256": file_digest(ROOT / "src/coveragecv/training/localization_cascade.py"),
                "detector_train_images_previously_seen": True,
                "holdout_limitation": "Held out from cascade fit, but previously seen by detector/encoder training",
                "matched_proposals": len(rows), "classes": classes, "device": "mps"}
    if (OUT / "protocol.json").exists() and read_json(OUT / "protocol.json") != protocol:
        raise ValueError("Frozen experiment changed; use a new experiment directory")
    write_json(OUT / "protocol.json", protocol)
    extractor = FeatureExtractor(ENCODER, classes)
    if extractor.metadata["training_view_digest"] != manifest["digest"]:
        raise ValueError("Encoder learned from a different annotation budget")
    cache = OUT / "features.pt"
    if cache.exists():
        cached = torch.load(cache, weights_only=True)
        if cached["protocol"] != protocol:
            raise ValueError("Stale feature cache")
        features = cached["features"].numpy()
    else:
        features = extractor.extract([{k: v for k, v in row.items() if k not in ("target", "annotation_id")}
                                      for row in rows], images, VIEW / "train")
        _save_checkpoint(cache, {"protocol": protocol, "features": torch.from_numpy(features)})
    model, selection = train_cascade(features, rows, training_holdout(rows, images, manifest))
    selection["status"] = "accepted_on_train_holdout" if model["strength"] else "rejected_on_train_holdout"
    write_json(OUT / "train_selection.json", selection)
    _save_checkpoint(OUT / "cascade.pt", {"protocol": protocol,
        "model": {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in model.items()}})
    print(selection, flush=True)
    if not model["strength"]:
        return
    bundle = Path(SPEC["complete_bundle"])
    reference_manifest = verify(bundle)
    reference = read_json(bundle / "splits/valid.coco.json")
    eval_images = {im["id"]: {k: im[k] for k in ("id", "file_name", "width", "height")}
                   for im in reference["images"]}
    train_hashes = {manifest["files"]["train/"+im["file_name"]] for im in images.values()}
    eval_hashes = {reference_manifest["files"][im["file_name"]] for im in eval_images.values()}
    if train_hashes & eval_hashes:
        raise ValueError("Train/evaluation exact image overlap")
    results = {}
    for method in ("aware_large_704_ema", "naive_large_704_ema", "complete_reference_large_704_ema"):
        folder = ROOT / "artifacts/capacity/all-pieces/20260917" / method
        baseline = read_json(folder / "evaluation.json")
        if (baseline["checkpoint_sha256"] != file_digest(folder / "detector.pt")
                or baseline["bundle_digest"] != reference_manifest["digest"] or baseline["split"] != "valid"):
            raise ValueError("Baseline checkpoint/reference mismatch")
        predictions = baseline["predictions"]
        indices = [i for i, p in enumerate(predictions) if p["score"] >= PROTOCOL["minimum_confidence"]
                   and p["bbox"][2] > 0 and p["bbox"][3] > 0]
        eligible = [predictions[i] for i in indices]
        started = time.monotonic()
        f = extractor.extract(eligible, eval_images, bundle)
        corrected = apply_corrections(eligible, predict_ridge(model, f), model["strength"])
        updated = [dict(row) for row in predictions]
        for i, row in zip(indices, corrected, strict=True):
            updated[i] = row
        result = {"kind": "observed_proposal_localization_cascade", "protocol": protocol,
                  "cascade_sha256": file_digest(OUT / "cascade.pt"),
                  "checkpoint_sha256": baseline["checkpoint_sha256"],
                  "baseline_sha256": file_digest(folder / "evaluation.json"),
                  "bundle_digest": reference_manifest["digest"], "split": "valid",
                  "baseline_metrics": score_predictions(predictions, reference),
                  "metrics": score_predictions(updated, reference), "predictions": updated,
                  "refinement_seconds": time.monotonic()-started, "refined_boxes": len(indices),
                  "interpretation": "Same partial-trained cascade applied to each detector; one seed. "
                    "Cascade trained on aware proposals; not an independently trained full-label cascade control."}
        write_json(OUT / f"{method}.json", result)
        results[method] = {k: v for k, v in result.items() if k not in ("predictions", "protocol")}
        print({"method": method, "before": result["baseline_metrics"]["AP"],
               "after": result["metrics"]["AP"]}, flush=True)
    write_json(OUT / "comparison.json", {"status": "completed", "results": results})


if __name__ == "__main__":
    main()
