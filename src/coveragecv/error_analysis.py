"""Post-training diagnostics. These labels never feed teacher mining or training."""
import contextlib
import io
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coveragecv.artifacts import read_json, write_json
from coveragecv.training.evaluate import iou


def diagnostic_metrics(evaluation, reference, *, threshold=.25):
    predictions = evaluation["predictions"]
    if not predictions:
        return {"threshold": threshold, "error_counts": {"missed": len(reference["annotations"])},
                "AP_by_IoU": {}, "confusions": []}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {**reference, "info": {}}
        gt.createIndex()
        dt = gt.loadRes(predictions)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    curve = {}
    for index, cutoff in enumerate(evaluator.params.iouThrs):
        cells = precision[index]
        curve[f"{cutoff:.2f}"] = float(cells[cells >= 0].mean()) if np.any(cells >= 0) else None
    errors, confusions, match_ious = Counter(), Counter(), []
    annotations, per_image = {}, {}
    for ann in reference["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)
    for prediction in predictions:
        if prediction["score"] >= threshold:
            per_image.setdefault(prediction["image_id"], []).append(prediction)
    for image in reference["images"]:
        ground = annotations.get(image["id"], [])
        matched = set()
        for pred in sorted(per_image.get(image["id"], []), key=lambda p: -p["score"]):
            overlaps = [(iou(pred["bbox"], ann["bbox"]), index, ann) for index, ann in enumerate(ground)]
            same_class = [(score, index) for score, index, ann in overlaps if ann["category_id"] == pred["category_id"]]
            best, index = max(((score, index) for score, index in same_class if index not in matched), default=(0, -1))
            if best >= .5:
                errors["true_positive"] += 1
                match_ious.append(best)
                matched.add(index)
                continue
            if any(score >= .5 and index in matched for score, index in same_class):
                errors["duplicate"] += 1
            elif overlaps and (best_other := max(overlaps, key=lambda row: row[0]))[0] >= .5:
                errors["class_confusion"] += 1
                confusions[(best_other[2]["category_id"], pred["category_id"])] += 1
            elif best >= .1:
                errors["localization"] += 1
            else:
                errors["background_or_bad_localization"] += 1
        errors["missed"] += len(ground)-len(matched)
    return {"threshold": threshold, "AP_by_IoU": curve, "error_counts": dict(errors),
            "matched_box_iou_mean": float(np.mean(match_ious)) if match_ious else None,
            "matched_boxes_above_90_iou": sum(x >= .9 for x in match_ious),
            "confusions": [{"ground_category": a, "predicted_category": b, "count": n}
                           for (a, b), n in confusions.most_common()],
            "interpretation": "Greedy diagnostic categories at a fixed score threshold, not TIDE causal AP attribution."}


def analyze_completed(workspace: Path):
    records = []
    for task, experiment in (("pawns", "chess_experiment.json"), ("all-pieces", "full_chess/experiment.json"),
                             ("construction", "construction/experiment.json")):
        if not (workspace / "artifacts" / experiment).exists():
            continue
        ex = read_json(workspace / "artifacts" / experiment)
        reference = read_json(Path(ex["complete_bundle"]) / "splits/valid.coco.json")
        for stage in ("gpu", "improved", "capacity"):
            for path in sorted((workspace / "artifacts" / stage / task).glob("*/*/evaluation.json")):
                if not (path.parent / "receipt.json").exists():
                    continue
                target = path.parent / "error_analysis.json"
                if target.exists():
                    analysis = read_json(target)
                else:
                    evaluation = read_json(path)
                    analysis = {"checkpoint_sha256": evaluation["checkpoint_sha256"],
                                "bundle_digest": evaluation["bundle_digest"],
                                **diagnostic_metrics(evaluation, reference)}
                    write_json(target, analysis)
                records.append({"task": task, "seed": int(path.parent.parent.name),
                                "method": path.parent.name, **analysis})
    output = {"runs": records, "purpose": "Post-training error diagnosis; never input to the training pipeline."}
    write_json(workspace / "artifacts/error_analysis.json", output)
    return output
