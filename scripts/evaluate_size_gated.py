"""One exploratory train-size gate over cached full-frame/tiled predictions.

This follow-up is validation-guided: earlier tiled results motivated the mechanism.
The class rule is computed only from observed human TRAIN annotations and frozen
before combined scoring; test annotations remain unused. No parameter search.
"""
import argparse
import statistics
from datetime import UTC, datetime
from pathlib import Path

from coveragecv.artifacts import digest, file_digest, read_json, verify, write_json
from coveragecv.training.evaluate import score_predictions

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "original": ("artifacts/tiled/construction/20260917", ["naive", "aware", "complete_reference"]),
    "crops": ("artifacts/crop-tiled/construction/20260917", ["aware_object_crops", "aware_standard",
                                                          "naive_object_crops", "complete_reference_object_crops"]),
}


def training_rule(view):
    manifest = verify(view)
    if manifest["kind"] != "training_view":
        raise ValueError("size gate requires an immutable learner view")
    training = read_json(view / "train/_annotations.coco.json")
    images = {image["id"]: image for image in training["images"]}
    longest = {category["id"]: [] for category in training["categories"]}
    for annotation in training["annotations"]:
        if not isinstance(annotation.get("is_pseudo", False), bool) or annotation.get("is_pseudo", False):
            raise ValueError("human-only gate forbids pseudo labels")
        image = images[annotation["image_id"]]
        _, _, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            raise ValueError("positive human box dimensions required")
        longest[annotation["category_id"]].append(max(width*512/image["width"], height*512/image["height"]))
    rows = [{"category_id": category["id"], "name": category["name"],
             "human_boxes": len(longest[category["id"]]),
             "median_longest_side_at512": statistics.median(longest[category["id"]])
             if longest[category["id"]] else None} for category in training["categories"]]
    small = [row["category_id"] for row in rows
             if row["median_longest_side_at512"] is not None and row["median_longest_side_at512"] <= 96]
    return {"partial_view_digest": manifest["digest"], "class_statistics": rows, "tiled_category_ids": small,
            "rule": "median human-observed TRAIN longest box side at512 <=96px uses tiled; others use rawfull",
            "threshold_origin": "same96px object-size target as crop-training plan; one fixed threshold",
            "empty_class_policy": "no observed box means use raw full-frame; do not infer size",
            "useful": bool(small) and len(small) < len(rows)}


def combine_predictions(full, tiled, small_categories):
    # Classes are disjoint between branches; preserve original boxes and scores verbatim.
    selected = set(small_categories)
    return ([dict(row) for row in full if row["category_id"] not in selected]
            + [dict(row) for row in tiled if row["category_id"] in selected])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--cohort", choices=COHORTS, default="original")
    args = parser.parse_args()
    experiment = read_json(ROOT / "artifacts/construction/experiment.json")
    rule = training_rule(Path(experiment["partial_view"]))
    declaration = {"rule": rule, "source_sha256": file_digest(Path(__file__)), "cohorts": COHORTS,
                   "validation_guided_exploration": True, "threshold_search": False,
                   "primary_crop_comparison_unchanged": True, "test_labels_used": False,
                   "scope": "all120validationimages/all5classes; identical rule for all3original and all4crop arms",
                   "design_context": "earlier aware tiled output improved rare-small recall but hurt large-person AP",
                   "latency": "same full+fourtiles computation; cached combination adds no model passes"}
    # Canonical round trip normalizes tuple-valued cohort constants before comparison.
    import json
    declaration = json.loads(json.dumps(declaration))
    frozen_path = ROOT / "artifacts/size-gated/declaration.json"
    if frozen_path.exists():
        if read_json(frozen_path)["declaration"] != declaration:
            raise ValueError("frozen size rule or implementation changed")
    else:
        if not args.declare_only:
            raise ValueError("freeze training-derived size rule before scoring")
        write_json(frozen_path, {"frozen_at": datetime.now(UTC).isoformat(), "declaration": declaration,
                                "declaration_sha256": digest(declaration)})
    if args.declare_only or not rule["useful"]:
        print({"declaration": str(frozen_path.relative_to(ROOT)), **rule}, flush=True)
        return
    source, cases = COHORTS[args.cohort]
    source = ROOT / source
    if not (source / "results.json").exists() or len(read_json(source / "results.json")["runs"]) != len(cases):
        raise ValueError("complete every sliced control in the cohort first")
    bundle = Path(experiment["complete_bundle"])
    bundle_digest = verify(bundle)["digest"]
    reference = read_json(bundle / "splits/valid.coco.json")
    rows = []
    for case in cases:
        folder = source / case
        full, tiled = read_json(folder / "full_frame.json"), read_json(folder / "tiled.json")
        if (full["checkpoint_sha256"] != tiled["checkpoint_sha256"] or full["bundle_digest"] != bundle_digest
                or tiled["bundle_digest"] != bundle_digest or full["split"] != "valid" or tiled["split"] != "valid"
                or full["device"] != "cpu" or tiled["device"] != "cpu"):
            raise ValueError("cached prediction sources must bind the same model, reference, split, and device")
        predictions = combine_predictions(full["predictions"], tiled["predictions"], rule["tiled_category_ids"])
        metrics = score_predictions(predictions, reference)
        result = {"case": case, "checkpoint_sha256": full["checkpoint_sha256"], "bundle_digest": bundle_digest,
                  "declaration_sha256": file_digest(frozen_path), "split": "valid", "device": "cpu",
                  "full_evaluation_sha256": file_digest(folder / "full_frame.json"),
                  "tiled_evaluation_sha256": file_digest(folder / "tiled.json"), "metrics": metrics,
                  "full_metrics": full["metrics"], "tiled_metrics": tiled["metrics"],
                  "AP_delta_points": 100*(metrics["AP"]-full["metrics"]["AP"]),
                  "tiled_category_ids": rule["tiled_category_ids"], "timing": tiled["timing"],
                  "validation_guided_exploration": True}
        target = ROOT / "artifacts/size-gated" / args.cohort / case / "evaluation.json"
        write_json(target, {**result, "predictions": predictions})
        rows.append(result)
        print({"cohort": args.cohort, "case": case, "AP": metrics["AP"]*100,
               "delta_points": result["AP_delta_points"]}, flush=True)
    write_json(ROOT / "artifacts/size-gated" / args.cohort / "results.json", {"runs": rows})


if __name__ == "__main__":
    main()
