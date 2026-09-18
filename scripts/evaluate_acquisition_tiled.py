"""One frozen acquisition/sliced-inference interaction; no new training or settings."""
import argparse
from datetime import UTC, datetime
from pathlib import Path

from evaluate_size_gated import combine_predictions, training_rule

from coveragecv.artifacts import digest, file_digest, read_json, verify, write_json
from coveragecv.training.evaluate import score_predictions
from coveragecv.training.tiled import PROTOCOL, evaluate_tiled

ROOT = Path(__file__).resolve().parents[1]
PARENTS = ROOT / "artifacts/acquisition/construction/20260917"
OUTPUT = ROOT / "artifacts/acquisition-tiled/construction/20260917"
CASES = ["guided", "random", "complete_standard"]
SEMANTIC_FIELDS = ("image_id", "category_id", "bbox", "score")


def semantic_predictions(rows):
    return [{key: row[key] for key in SEMANTIC_FIELDS} for row in rows]


def freeze_inputs():
    experiment = read_json(ROOT / "artifacts/construction/experiment.json")
    gate = training_rule(Path(experiment["partial_view"]))
    original_gate = read_json(ROOT / "artifacts/size-gated/declaration.json")["declaration"]["rule"]
    original_tiling = read_json(ROOT / "artifacts/tiled/construction/20260917/protocol.json")
    if gate != original_gate or gate["tiled_category_ids"] != [1, 2]:
        raise ValueError("must preserve ORIGINAL partial-TRAIN class gate, not acquired-view medians")
    if PROTOCOL != original_tiling["binding"]["protocol"]:
        raise ValueError("tiled protocol differs from original frozen settings")
    primary_path = PARENTS / "comparison.json"
    primary = read_json(primary_path)
    if primary["status"] != "completed" or not set(CASES).issubset(primary["results"]):
        raise ValueError("all three primary evaluations must finish before interaction")
    parents = {}
    for case in CASES:
        folder = PARENTS / case
        receipt, evaluation = read_json(folder / "receipt.json"), read_json(folder / "evaluation.json")
        checkpoint_sha = file_digest(folder / "detector.pt")
        if (receipt["status"] != "completed" or receipt["evaluation_status"] != "completed"
                or receipt["checkpoint_sha256"] != checkpoint_sha
                or evaluation["checkpoint_sha256"] != checkpoint_sha
                or primary["results"][case]["checkpoint_sha256"] != checkpoint_sha
                or evaluation["split"] != "valid" or evaluation["device"] != "cpu"):
            raise ValueError("checkpoint receipt and primary evidence must agree")
        parents[case] = {"checkpoint_sha256": checkpoint_sha,
                         "evaluation_sha256": file_digest(folder / "evaluation.json")}
    zero_path = ROOT / "artifacts/size-gated/crops/aware_standard/evaluation.json"
    zero = read_json(zero_path)
    sources = [Path(__file__), ROOT / "scripts/evaluate_size_gated.py",
               ROOT / "src/coveragecv/training/tiled.py", ROOT / "src/coveragecv/training/evaluate.py"]
    declaration = {
        "kind": "annotation_acquisition_frozen_inference_interaction", "cases": CASES,
        "device": "cpu", "tiled_protocol": PROTOCOL, "original_partial_train_gate": gate,
        "parents": parents, "primary_comparison_sha256": file_digest(primary_path),
        "training_protocol_sha256": file_digest(ROOT / "artifacts/acquisition/protocol.json"),
        "source_files": {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sources},
        "zero_review_control": {"checkpoint_sha256": zero["checkpoint_sha256"],
                                "gated_evaluation_sha256": file_digest(zero_path)},
        "scope": "all120validationimages/all717boxes/all5classes; test unevaluated",
        "selection": "all three cases, one existing setting, no parameter search or class removal",
        "primary_comparison_unchanged": "guided versus random under original single-pass CPU evaluator",
        "interpretation": "separate exploratory interaction; acquisition primary scores already observed",
        "compute": "CPU only; five image passes versus one, no new cloud training or inference",
    }
    return declaration, experiment, primary, zero


def score_gate(case, declaration, reference):
    folder = OUTPUT / case
    full, tiled = read_json(folder / "full_frame.json"), read_json(folder / "tiled.json")
    expected_sha = declaration["parents"][case]["checkpoint_sha256"]
    if (full["checkpoint_sha256"] != expected_sha or tiled["checkpoint_sha256"] != expected_sha
            or full["bundle_digest"] != tiled["bundle_digest"]
            or full["split"] != "valid" or tiled["split"] != "valid"
            or full["device"] != "cpu" or tiled["device"] != "cpu"):
        raise ValueError("full-frame and tile sources must bind the declared model/reference/device")
    selected = declaration["original_partial_train_gate"]["tiled_category_ids"]
    predictions = combine_predictions(full["predictions"], tiled["predictions"], selected)
    expected_semantics = semantic_predictions(predictions)
    metrics = score_predictions(predictions, reference)
    if semantic_predictions(predictions) != expected_semantics:
        raise ValueError("scoring changed prediction image/class/box/score values")
    classes = sorted(reference["categories"], key=lambda category: category["id"])
    expected_ap = sum((tiled if category["id"] in selected else full)["metrics"]["per_class_AP"]
                      [category["name"]] for category in classes)/len(classes)
    if (abs(expected_ap-metrics["AP"]) > 1e-12 or metrics["images"] != 120 or metrics["boxes"] != 717
            or len(metrics["per_class_AP"]) != 5):
        raise ValueError("gate scope or independent per-class AP arithmetic failed")
    result = {"case": case, "checkpoint_sha256": expected_sha, "bundle_digest": full["bundle_digest"],
              "split": "valid", "device": "cpu", "metrics": metrics, "tiled_category_ids": selected,
              "original_partial_view_digest": declaration["original_partial_train_gate"]["partial_view_digest"],
              "full_evaluation_sha256": file_digest(folder / "full_frame.json"),
              "tiled_evaluation_sha256": file_digest(folder / "tiled.json"),
              "timing": tiled["timing"], "predictions": predictions}
    write_json(folder / "size_gated.json", result)
    audit = {"case": case, "passed": True, "exact_semantic_source_selection": True,
             "all120images_all717boxes_all5classes": True,
             "independent_selected_per_class_AP_error": abs(expected_ap-metrics["AP"]),
             "gated_evaluation_sha256": file_digest(folder / "size_gated.json"),
             "COCO_metadata_note": "scorer assigns sequential prediction IDs; semantic fields unchanged"}
    write_json(folder / "integrity.json", audit)
    return result, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    declaration, experiment, primary, zero = freeze_inputs()
    declaration_path = ROOT / "artifacts/acquisition-tiled/declaration.json"
    if declaration_path.exists():
        if read_json(declaration_path)["declaration"] != declaration:
            raise ValueError("frozen sources or inputs changed")
    else:
        if not args.declare_only:
            raise ValueError("freeze cohort before first new tiled score")
        if any((OUTPUT / case / "tiled.json").exists() for case in CASES):
            raise ValueError("cannot declare before scores after a tiled score already exists")
        write_json(declaration_path, {"frozen_at": datetime.now(UTC).isoformat(), "declaration": declaration,
                                     "declaration_sha256": digest(declaration)})
    if args.declare_only:
        print({"declared": str(declaration_path.relative_to(ROOT)), "cases": CASES,
               "gate_categories": declaration["original_partial_train_gate"]["tiled_category_ids"]}, flush=True)
        return
    bundle = Path(experiment["complete_bundle"])
    verify(bundle)
    reference = read_json(bundle / "splits/valid.coco.json")
    rows, audits = [], []
    for case in CASES:
        folder = OUTPUT / case
        if (folder / "comparison.json").exists():
            comparison = read_json(folder / "comparison.json")
        else:
            print({"starting": case}, flush=True)
            comparison = evaluate_tiled(PARENTS / case / "detector.pt", bundle, folder,
                                        baseline_evaluation=PARENTS / case / "evaluation.json", device="cpu")
        gated, audit = score_gate(case, declaration, reference)
        audits.append(audit)
        acquisition = primary["results"][case].get("acquisition")
        timing = comparison["timing"]
        full_seconds = timing["full"]["elapsed_seconds"]
        combined_seconds = full_seconds+timing["tiles"]["elapsed_seconds"]+timing["merge_seconds"]
        row = {"case": case, "checkpoint_sha256": comparison["checkpoint_sha256"],
               "primary_metrics": primary["results"][case]["metrics"],
               "full_metrics": comparison["full_frame_metrics"], "nms_metrics": comparison["full_frame_nms_metrics"],
               "tiled_metrics": comparison["tiled_metrics"], "gated_metrics": gated["metrics"],
               "gated_minus_fresh_full_AP_points": 100*(gated["metrics"]["AP"]-comparison["full_frame_metrics"]["AP"]),
               "gated_minus_zero_review_AP_points": 100*(gated["metrics"]["AP"]-zero["metrics"]["AP"]),
               "review_units": acquisition["review_units"] if acquisition else None,
               "added_boxes": acquisition["added_boxes"] if acquisition else None,
               "full_image_passes": timing["full"]["image_passes"],
               "combined_image_passes": timing["full"]["image_passes"]+timing["tiles"]["image_passes"],
               "full_seconds": full_seconds, "combined_seconds": combined_seconds,
               "observed_time_ratio": combined_seconds/full_seconds, "integrity": audit}
        rows.append(row)
        write_json(OUTPUT / "results.json", {
            "status": "completed" if len(rows) == len(CASES) else "running", "runs": rows,
            "declaration_sha256": file_digest(declaration_path), "metric": "COCO AP50:95, fractions",
            "zero_review_control": {"full_metrics": zero["full_metrics"], "gated_metrics": zero["metrics"],
                                    **declaration["zero_review_control"]},
            "scope": declaration["scope"], "interpretation": declaration["interpretation"],
            "cost": "five image passes per640px image; CPU only, zero new cloud compute"})
        write_json(OUTPUT / "integrity.json", {"passed": True, "completed_cases": len(audits), "audits": audits})
        print({"case": case, "full_AP": 100*row["full_metrics"]["AP"],
               "tiled_AP": 100*row["tiled_metrics"]["AP"], "gated_AP": 100*row["gated_metrics"]["AP"],
               "gated_vs_zero_review": row["gated_minus_zero_review_AP_points"]}, flush=True)


if __name__ == "__main__":
    main()
