"""Recompute the fixed acquisition comparison from saved validation predictions."""
from pathlib import Path

from coveragecv.artifacts import digest, file_digest, read_json, verify, write_json
from coveragecv.training.evaluate import score_predictions

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = ROOT / "artifacts/acquisition"
    protocol = read_json(study / "protocol.json")
    output = study / "construction" / str(protocol["seed"])
    comparison = read_json(output / "comparison.json")
    if comparison["protocol_sha256"] != file_digest(study / "protocol.json"):
        raise ValueError("comparison protocol identity differs")
    bundle = Path(read_json(ROOT / "artifacts/construction/experiment.json")["complete_bundle"])
    if verify(bundle)["digest"] != protocol["reference_bundle_digest"]:
        raise ValueError("reference bundle identity differs")
    reference = read_json(bundle / "splits/valid.coco.json")
    rows = {}
    for case, expected in comparison["results"].items():
        folder = ROOT / protocol["zero_review_control"]["folder"] if case == "zero_review" else output / case
        result = read_json(folder / "evaluation.json")
        if (result["checkpoint_sha256"] != expected["checkpoint_sha256"]
                or result["checkpoint_sha256"] != file_digest(folder / "detector.pt")
                or file_digest(folder / "evaluation.json") != expected["evaluation_sha256"]
                or result["bundle_digest"] != protocol["reference_bundle_digest"] or result["split"] != "valid"
                or result["device"] != "cpu" or result["resolution"] != 512 or result["score_threshold"] != .25
                or result["postprocess"] != "stock RF-DETR; reserved output omitted from semantic COCO mapping"):
            raise ValueError(f"{case} evaluation provenance differs")
        # All predictions and all five classes; no coverage masking or reference changes.
        metrics = score_predictions(result["predictions"], reference, threshold=.25)
        if metrics != result["metrics"] or metrics != expected["metrics"]:
            raise ValueError(f"{case} independently rescored metrics differ")
        rows[case] = {"checkpoint_sha256": result["checkpoint_sha256"],
                      "evaluation_sha256": expected["evaluation_sha256"],
                      "predictions": len(result["predictions"]), "metrics_exact": True,
                      "AP": metrics["AP"], "AP50": metrics["AP50"], "per_class_AP": metrics["per_class_AP"]}
    recomputed = 100 * (rows["guided"]["AP"] - rows["random"]["AP"])
    if abs(recomputed - comparison["primary_AP_delta_points"]) > 1e-12:
        raise ValueError("primary comparison arithmetic differs")
    audit = {"status": "passed", "comparison_sha256": file_digest(output / "comparison.json"),
        "reference_validation_sha256": file_digest(bundle / "splits/valid.coco.json"),
        "reference_images": len(reference["images"]), "reference_boxes": len(reference["annotations"]),
        "reference_categories": reference["categories"], "results": rows,
        "primary_AP_delta_points": recomputed, "test_labels_evaluated": False,
        "scope": "Rescored saved predictions through COCO and operating-point scorer; not an independent inference rerun."}
    write_json(study / "rescoring_audit.json", {**audit, "digest": digest(audit)})
    print({"status": "passed", "cases": len(rows), "primary_AP_delta_points": recomputed}, flush=True)


if __name__ == "__main__":
    main()
