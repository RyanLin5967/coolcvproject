"""Pair transformed views to adjust geometry without changing detection decisions."""
import hashlib
import math
from collections import defaultdict
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json

POLICY = {
    "minimum_score": .05, "minimum_pair_iou": .70, "flip_fraction": .5,
    "maximum_corner_fraction": .15, "holdout_fraction": .20, "seed": 20260919,
    "minimum_holdout_iou_gain": .003, "strict_iou": .90,
    "association": "descending IoU greedy one-to-one within image and category; deterministic index ties",
    "preserve": "original cardinality, class, confidence, ordering, and every unmatched box",
    "gate": "Observed TRAIN matches only; require mean IoU gain and no loss of strict-IoU matches",
}


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0., min(ax+aw, bx+bw)-max(ax, bx)) * max(0., min(ay+ah, by+bh)-max(ay, by))
    return area / max(aw*ah+bw*bh-area, 1e-12)


def paired_geometry(original, flipped, policy=POLICY):
    """Flipped boxes must already be mapped back into original image coordinates."""
    groups = defaultdict(lambda: [[], []])
    for side, rows in enumerate((original, flipped)):
        for index, row in enumerate(rows):
            box, score = row["bbox"], row["score"]
            if len(box) != 4 or not all(math.isfinite(x) for x in (*box, score)):
                raise ValueError("Nonfinite box or score")
            if min(box[2:]) < 0 or not 0 <= score <= 1:
                raise ValueError("Invalid box geometry or confidence")
            # Stock top-k includes occasional zero-area, low-score boxes. Retain
            # their original rows for exact baseline parity, but never pair them.
            if min(box[2:]) > 0 and score >= policy["minimum_score"]:
                groups[(row["image_id"], row["category_id"])][side].append(index)
    result = [{**row, "bbox": list(row["bbox"])} for row in original]
    matched = 0
    for left, right in groups.values():
        candidates = [(overlap(original[i]["bbox"], flipped[j]["bbox"]), i, j) for i in left for j in right]
        used_left, used_right = set(), set()
        for iou, i, j in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
            if iou < policy["minimum_pair_iou"]:
                break
            if i in used_left or j in used_right:
                continue
            used_left.add(i)
            used_right.add(j)
            a, b = original[i]["bbox"], flipped[j]["bbox"]
            old = [a[0], a[1], a[0]+a[2], a[1]+a[3]]
            new = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
            limits = [a[2], a[3], a[2], a[3]]
            corners = [x + policy["flip_fraction"] * max(-size*policy["maximum_corner_fraction"],
                       min(size*policy["maximum_corner_fraction"], y-x))
                       for x, y, size in zip(old, new, limits, strict=True)]
            result[i]["bbox"] = [corners[0], corners[1], corners[2]-corners[0], corners[3]-corners[1]]
            matched += 1
    return result, matched


def train_gate(view, proposals, policy=POLICY):
    manifest = verify(view)
    data = read_json(view / "train/_annotations.coco.json")
    classes = read_json(view / "ontology.json")["classes"]
    images = {im["id"]: im for im in data["images"]}
    if (manifest["kind"] != "training_view" or proposals["status"] != "completed"
            or proposals["binding"]["view_digest"] != manifest["digest"]
            or proposals["binding"]["classes"] != classes
            or set(proposals["completed_images"]) != set(images)):
        raise ValueError("Proposal collection is not bound to this complete TRAIN view")
    predictions = proposals["predictions"]
    if any(p["view"] not in ("original", "flip") or p["image_id"] not in images
           or not 1 <= p["category_id"] <= len(classes) for p in predictions):
        raise ValueError("Unknown proposal image, class or transform")
    original = [p for p in predictions if p["view"] == "original"]
    flipped = [p for p in predictions if p["view"] == "flip"]
    corrected, paired = paired_geometry(original, flipped, policy)
    by_class = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("is_pseudo") or ann.get("iscrowd") or ann.get("ignore"):
            raise ValueError("Gate requires ordinary human-observed training boxes")
        by_class[(ann["image_id"], ann["category_id"])].append(ann)
    held = {}
    for iid, im in images.items():
        image_sha = manifest["files"]["train/" + im["file_name"]]
        value = hashlib.sha256((str(policy["seed"])+image_sha).encode()).digest()
        held[iid] = int.from_bytes(value[:4], "big") / 2**32 < policy["holdout_fraction"]
    rows, used = [], set()
    for index in sorted(range(len(original)), key=lambda i: (-original[i]["score"], i)):
        pred = original[index]
        choices = [(overlap(pred["bbox"], a["bbox"]), a) for a in by_class[(pred["image_id"], pred["category_id"])]
                   if a["id"] not in used]
        before, target = max(choices, key=lambda x: x[0], default=(0., None))
        if before < .5:
            continue
        used.add(target["id"])
        rows.append({"image_id": pred["image_id"], "annotation_id": target["id"],
                     "before_iou": before, "after_iou": overlap(corrected[index]["bbox"], target["bbox"]),
                     "heldout": held[pred["image_id"]]})
    selected = [r for r in rows if r["heldout"]]
    if len(selected) < 100:
        raise ValueError("Insufficient image-grouped training holdout matches")
    before = sum(r["before_iou"] for r in selected) / len(selected)
    after = sum(r["after_iou"] for r in selected) / len(selected)
    before_strict = sum(r["before_iou"] >= policy["strict_iou"] for r in selected)
    after_strict = sum(r["after_iou"] >= policy["strict_iou"] for r in selected)
    accepted = after-before >= policy["minimum_holdout_iou_gain"] and after_strict >= before_strict
    return {"accepted": accepted, "before_mean_iou": before, "after_mean_iou": after,
            "before_strict_matches": before_strict, "after_strict_matches": after_strict,
            "heldout_matches": len(selected), "matched_observed_objects": len(rows), "paired_predictions": paired,
            "policy": policy, "rows": rows, "proposal_binding": proposals["binding"],
            "limitation": "Detector previously trained on these images; this is a refiner gate, not an independent detector test."}


def evaluate_pair(checkpoint: Path, bundle: Path, output: Path, *, policy=POLICY):
    import torch
    from PIL import Image
    from rfdetr.training import RFDETRModelModule

    from coveragecv.compiler import _jsonl
    from coveragecv.training.evaluate import score_predictions
    from coveragecv.training.runner import checkpoint_config, configs
    from coveragecv.training.tiled import _tensor

    if not torch.cuda.is_available():
        raise RuntimeError("Geometry pilot inference requires a cloud GPU")
    torch.set_num_threads(2)
    manifest = verify(bundle)
    classes = read_json(bundle / "ontology.json")["classes"]
    if any(state not in ("exhaustive", "verified_absent")
           for row in _jsonl(bundle / "coverage.jsonl") if row["split"] == "valid" for state in row["states"]):
        raise ValueError("Scoring requires complete validation coverage")
    reference = read_json(bundle / "splits/valid.coco.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["class_names"] != classes:
        raise ValueError("Checkpoint class order differs from reference")
    mc = checkpoint_config(state, device="cuda")
    _, tc = configs(bundle, output)
    module = RFDETRModelModule(mc, tc).to("cuda").eval()
    module.model.load_state_dict(state["model"], strict=True)
    del state
    original, flipped = [], []
    for offset in range(0, len(reference["images"]), 2):
        chunk = reference["images"][offset:offset+2]
        tensors, sizes = [], []
        for im in chunk:
            with Image.open(safe_child(bundle, im["file_name"])) as image:
                tensors.append(_tensor(image, mc.resolution))
            sizes.append([im["height"], im["width"]])
        batch = torch.stack(tensors).to("cuda")
        with torch.inference_mode():
            raw = module.model(torch.cat([batch, batch.flip(-1)]))
            results = module.postprocess(raw, torch.tensor(sizes+sizes, device="cuda"))
        for i, im in enumerate(chunk):
            for mirror, result in ((False, results[i]), (True, results[i+len(chunk)])):
                for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(),
                                             result["scores"].tolist(), strict=True):
                    if label == len(classes):
                        continue
                    if not 0 <= label < len(classes):
                        raise ValueError("Unmapped class")
                    x1, y1, x2, y2 = box
                    if mirror:
                        x1, x2 = im["width"]-x2, im["width"]-x1
                    if x2 < x1 or y2 < y1:
                        raise ValueError("Invalid predicted box")
                    (flipped if mirror else original).append({"image_id": im["id"], "category_id": label+1,
                        "score": score, "bbox": [x1, y1, x2-x1, y2-y1]})
    corrected, paired = paired_geometry(original, flipped, policy)
    results = {}
    for name, predictions in (("original", original), ("geometry", corrected)):
        results[name] = {"checkpoint_sha256": file_digest(checkpoint), "bundle_digest": manifest["digest"],
                         "split": "valid", "device": "cuda", "resolution": mc.resolution,
                         "metrics": score_predictions(predictions, reference), "predictions": predictions,
                         "postprocess": "stock outputs" if name == "original" else "fixed geometry-only original/flip pairing",
                         "model_passes_per_image": 1 if name == "original" else 2,
                         "paired_predictions": paired, "policy": policy}
        write_json(output / f"{name}.json", results[name])
    del module
    torch.cuda.empty_cache()
    return {name: result["metrics"] for name, result in results.items()}
