"""Common complete-reference evaluation; coverage never hides false positives."""
import contextlib
import io
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr.training import RFDETRModelModule
from torchvision.transforms import functional as TF

from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json
from coveragecv.training.runner import checkpoint_config, configs


def evaluate_checkpoint(checkpoint: Path, bundle: Path, output: Path, *, split="valid", threshold=0.25,
                        device="cpu", resolution_override=None):
    if split not in ("valid", "test"):
        raise ValueError("evaluation requires an explicit validation or test split")
    if not 0 <= threshold <= 1:
        raise ValueError("confidence threshold must be between zero and one")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    manifest = verify(bundle)
    classes = read_json(bundle / "ontology.json")["classes"]
    from coveragecv.compiler import _jsonl
    if any(state not in ("exhaustive", "verified_absent")
           for row in _jsonl(bundle / "coverage.jsonl") if row["split"] == split for state in row["states"]):
        raise ValueError("evaluation requires complete reference coverage for every class")
    reference = read_json(bundle / f"splits/{split}.coco.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("class_names") != classes:
        raise ValueError("checkpoint and evaluation ontology differ")
    mc, tc = configs(bundle, output.parent)
    if "model_config" in state:
        mc = checkpoint_config(state, device=device)
    checkpoint_resolution = mc.resolution
    if resolution_override is not None:
        if (isinstance(resolution_override, bool) or not isinstance(resolution_override, int)
                or resolution_override < 64 or resolution_override > 1536 or resolution_override % 64):
            raise ValueError("Inference resolution must be an explicit multiple of64 between64 and1536")
        mc.resolution = resolution_override
    module = RFDETRModelModule(mc, tc).to(device).eval()
    module.model.load_state_dict(state["model"], strict=True)
    predictions = []
    reserved = 0
    for offset in range(0, len(reference["images"]), 4):
        chunk = reference["images"][offset:offset+4]
        tensors, sizes = [], []
        for item in chunk:
            with Image.open(safe_child(bundle, item["file_name"])) as image:
                image = image.convert("RGB").resize((mc.resolution, mc.resolution))
                tensor = TF.to_tensor(image)
                tensors.append(TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225]))
                sizes.append([item["height"], item["width"]])
        with torch.no_grad():
            raw = module.model(torch.stack(tensors).to(device))
            results = module.postprocess(raw, torch.tensor(sizes, device=device))
        for item, result in zip(chunk, results):
            for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
                if label == len(classes):
                    reserved += 1
                    continue
                if not 0 <= label < len(classes):
                    raise ValueError("postprocessor returned an unmapped semantic class")
                x1, y1, x2, y2 = box
                predictions.append({"image_id": item["id"], "category_id": label+1,
                                    "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
    metrics = score_predictions(predictions, reference, threshold=threshold)
    result = {"checkpoint_sha256": file_digest(checkpoint), "bundle_digest": manifest["digest"], "split": split,
              "resolution": mc.resolution, "checkpoint_resolution": checkpoint_resolution,
              "resolution_override": resolution_override, "device": device,
              "model_variant": state.get("model_variant", "nano"),
              "score_threshold": threshold, "threshold_policy": "fixed before viewing pilot results",
              "postprocess": "stock RF-DETR; reserved output omitted from semantic COCO mapping",
              "reserved_slot_selections": reserved, "metrics": metrics, "predictions": predictions}
    write_json(output, result)
    return result


def score_predictions(predictions, reference, *, threshold=.25):
    """One COCO/operating-point scorer shared by single models and ensembles."""
    classes = [c["name"] for c in sorted(reference["categories"], key=lambda c: c["id"])]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {**reference, "info": {}}
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            metrics = {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
                       "AP75": float(evaluator.stats[2]), "AR100": float(evaluator.stats[8])}
            precision = evaluator.eval["precision"]
            metrics["per_class_AP"] = {}
            for k, name in enumerate(classes):
                values = precision[:, :, k, 0, -1]
                valid = values[values >= 0]
                metrics["per_class_AP"][name] = float(valid.mean()) if valid.size else None
        else:
            metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0,
                       "per_class_AP": dict.fromkeys(classes, 0.0)}
    observed_ids = {a["image_id"] for a in reference["annotations"]}
    negatives = {im["id"] for im in reference["images"]} - observed_ids
    active = [p for p in predictions if p["score"] >= threshold]
    true_positive = false_positive = 0
    for im in reference["images"]:
        ground = [a for a in reference["annotations"] if a["image_id"] == im["id"]]
        matched = set()
        for pred in sorted([p for p in active if p["image_id"] == im["id"]], key=lambda p: -p["score"]):
            candidates = [(iou(pred["bbox"], ann["bbox"]), j) for j, ann in enumerate(ground)
                          if j not in matched and pred["category_id"] == ann["category_id"]]
            score, j = max(candidates, default=(0, -1))
            if score >= .5:
                true_positive += 1
                matched.add(j)
            else:
                false_positive += 1
    metrics.update(precision_at_threshold=true_positive/max(1, true_positive+false_positive),
                   recall_at_threshold=true_positive/max(1, len(reference["annotations"])),
                   false_positives_per_image=false_positive/max(1, len(reference["images"])),
                   negative_images=len(negatives),
                   negative_image_false_positives_per_image=(sum(p["image_id"] in negatives for p in active)/
                                                            max(1, len(negatives))),
                   images=len(reference["images"]), boxes=len(reference["annotations"]),
                   class_support=dict(Counter(str(a["category_id"]) for a in reference["annotations"])))
    return metrics


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
    return intersection/max(aw*ah+bw*bh-intersection, 1e-12)
