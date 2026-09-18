"""Label-blind fusion of independent detectors, with explicit inference cost/provenance."""
from collections import defaultdict
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, verify, write_json


def fuse_predictions(members, images, *, iou_threshold=.55, skip_score=.001):
    from ensemble_boxes import weighted_boxes_fusion

    if len(members) < 2 or not 0 < iou_threshold <= 1 or not 0 <= skip_score < 1:
        raise ValueError("fusion requires multiple members and valid thresholds")
    by_id = {im["id"]: im for im in images}
    grouped = []
    for predictions in members:
        rows = defaultdict(list)
        for p in predictions:
            if p["image_id"] not in by_id:
                raise ValueError("prediction contains an unknown image")
            rows[p["image_id"]].append(p)
        grouped.append(rows)
    fused = []
    for image_id, image in by_id.items():
        width, height = image["width"], image["height"]
        all_boxes, all_scores, all_labels = [], [], []
        for member in grouped:
            boxes, scores, labels = [], [], []
            for p in member[image_id]:
                x, y, w, h = p["bbox"]
                coords = [max(0., min(1., x/width)), max(0., min(1., y/height)),
                          max(0., min(1., (x+w)/width)), max(0., min(1., (y+h)/height))]
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    continue
                boxes.append(coords)
                scores.append(p["score"])
                labels.append(p["category_id"])
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        boxes, scores, labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels,
            weights=[1.]*len(members), iou_thr=iou_threshold, skip_box_thr=skip_score, conf_type="avg")
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            fused.append({"image_id": image_id, "category_id": int(label), "score": float(score),
                          "bbox": [float(x1*width), float(y1*height),
                                   float((x2-x1)*width), float((y2-y1)*height)]})
    return fused


def evaluate_ensemble(folders: list[Path], bundle: Path, output: Path, *, split="valid"):
    from coveragecv.training.evaluate import score_predictions

    if split not in ("valid", "test"):
        raise ValueError("ensemble scoring requires validation or test")
    manifest = verify(bundle)
    evaluations, checkpoints, contracts = [], [], set()
    for folder in folders:
        evaluation = read_json(folder / ("evaluation.json" if split == "valid" else "test_evaluation.json"))
        run = read_json(folder / "run.json")
        sha = file_digest(folder / "detector.pt")
        if (evaluation["bundle_digest"] != manifest["digest"] or evaluation["split"] != split
                or evaluation["checkpoint_sha256"] != sha or run["detector_sha256"] != sha
                or run["status"] != "completed"):
            raise ValueError("ensemble member binding mismatch")
        contracts.add((run["arm"], run.get("recipe"), run["steps"], run["model_config"]["resolution"]))
        evaluations.append(evaluation)
        checkpoints.append({"checkpoint_sha256": sha, "seed": run["seed"],
                            "single_model_AP": evaluation["metrics"]["AP"]})
    if len(contracts) != 1 or len({c["seed"] for c in checkpoints}) != len(checkpoints):
        raise ValueError("ensemble must use distinct seeds of one matched training recipe")
    reference = read_json(bundle / f"splits/{split}.coco.json")
    images = [{k: im[k] for k in ("id", "width", "height")} for im in reference["images"]]
    predictions = fuse_predictions([e["predictions"] for e in evaluations], images)
    result = {"bundle_digest": manifest["digest"], "split": split, "members": checkpoints,
              "model_passes_per_image": len(checkpoints), "algorithm": "ensemble-boxes 1.0.9 WBF",
              "iou_threshold": .55, "skip_score": .001, "score_threshold": .25,
              "metrics": score_predictions(predictions, reference), "predictions": predictions,
              "interpretation": "One ensemble, not a seed mean; incurs one inference pass per member."}
    write_json(output, result)
    return result
