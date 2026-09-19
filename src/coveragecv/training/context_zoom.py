"""Fixed wide-context detector reuse, gated on observed TRAIN geometry first.

Geometry helpers deliberately have no ML imports. Model execution is CUDA-only;
the detector, class decisions and confidence ordering stay unchanged.
"""
import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path

from coveragecv.artifacts import digest, file_digest, read_json, safe_child, verify, write_json

POLICY = {
    "minimum_score": .05, "maximum_crops_per_image": 32,
    "minimum_crop_side": 320, "maximum_crop_side": 512, "context_multiplier": 3.,
    "internal_edge_margin": 2., "minimum_pair_iou": .70,
    "crop_fraction": .5, "maximum_corner_fraction": .15,
    "holdout_fraction": .20, "seed": 20260919, "minimum_observed_matches": 100,
    "minimum_holdout_iou_gain": .003, "strict_iou": .90,
    "association": "Same class; maximum original-box IoU, then confidence, then original crop-output index",
    "preserve": "Original prediction count, ordering, image, class, confidence and every unmatched/zero-area box",
    "gate": "Observed TRAIN only; mean IoU gain, nondecreasing strict matches, no new IoU<.50 losses",
}


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0., min(ax+aw, bx+bw)-max(ax, bx)) * max(0., min(ay+ah, by+bh)-max(ay, by))
    return intersection / max(aw*ah+bw*bh-intersection, 1e-12)


def positive_geometry(row):
    box, score = row["bbox"], row["score"]
    if len(box) != 4 or not all(math.isfinite(v) for v in (*box, score)):
        raise ValueError("Nonfinite geometry or confidence")
    if min(box[2:]) < 0 or not 0 <= score <= 1:
        raise ValueError("Invalid geometry or confidence")
    return min(box[2:]) > 0


def crop_window(box, image_size, policy=POLICY):
    """Integer square shifted into the frame; never stretch a clipped rectangle."""
    x, y, w, h = box
    width, height = image_size
    if min(w, h) <= 0:
        return None
    side = math.ceil(max(policy["minimum_crop_side"], policy["context_multiplier"]*max(w, h)))
    if side > policy["maximum_crop_side"] or side > min(width, height):
        return None
    left = min(max(math.floor(x+w/2-side/2), 0), width-side)
    top = min(max(math.floor(y+h/2-side/2), 0), height-side)
    return left, top, left+side, top+side


def crop_plan(predictions, image_size, policy=POLICY):
    eligible = [i for i, row in enumerate(predictions)
                if positive_geometry(row) and row["score"] >= policy["minimum_score"]]
    eligible.sort(key=lambda i: (-predictions[i]["score"], i))
    planned = []
    for index in eligible[:policy["maximum_crops_per_image"]]:
        window = crop_window(predictions[index]["bbox"], image_size, policy)
        if window is not None:
            planned.append((index, window))
    return planned


def map_crop_box(box, window, image_size, margin=2.):
    """Reject internally truncated crop detections, preserve true frame edges."""
    x, y, w, h = box
    if min(w, h) <= 0:
        return None
    left, top, right, bottom = window
    width, height = image_size
    x2, y2 = x+w, y+h
    if ((left > 0 and x <= margin) or (top > 0 and y <= margin)
            or (right < width and x2 >= right-left-margin)
            or (bottom < height and y2 >= bottom-top-margin)):
        return None
    x1, y1 = max(0., x+left), max(0., y+top)
    x2, y2 = min(float(width), x2+left), min(float(height), y2+top)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2-x1, y2-y1]


def refine_anchor(anchor, candidates, window, image_size, policy=POLICY):
    """One crop can change only its original anchor's bounded geometry."""
    unchanged = {**anchor, "bbox": list(anchor["bbox"])}
    if not positive_geometry(anchor) or anchor["score"] < policy["minimum_score"]:
        return unchanged, False
    choices = []
    for index, row in enumerate(candidates):
        if (not positive_geometry(row) or row["score"] < policy["minimum_score"]
                or row["image_id"] != anchor["image_id"] or row["category_id"] != anchor["category_id"]):
            continue
        mapped = map_crop_box(row["bbox"], window, image_size, policy["internal_edge_margin"])
        if mapped is None:
            continue
        iou = overlap(anchor["bbox"], mapped)
        if iou >= policy["minimum_pair_iou"]:
            choices.append((iou, row["score"], -index, mapped))
    if not choices:
        return unchanged, False
    candidate = max(choices, key=lambda item: item[:3])[3]
    a, b = anchor["bbox"], candidate
    old, new = [a[0], a[1], a[0]+a[2], a[1]+a[3]], [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    sizes = [a[2], a[3], a[2], a[3]]
    corners = [x + policy["crop_fraction"] * max(-size*policy["maximum_corner_fraction"],
               min(size*policy["maximum_corner_fraction"], y-x))
               for x, y, size in zip(old, new, sizes, strict=True)]
    return {**unchanged, "bbox": [corners[0], corners[1], corners[2]-corners[0], corners[3]-corners[1]]}, True


def holdout_ids(images, manifest, policy=POLICY):
    selected = set()
    for im in images:
        image_sha = manifest["files"]["train/" + im["file_name"]]
        value = hashlib.sha256((str(policy["seed"])+image_sha).encode()).digest()
        if int.from_bytes(value[:4], "big") / 2**32 < policy["holdout_fraction"]:
            selected.add(im["id"])
    return selected


def assert_prediction_contract(original, zoom):
    if len(original) != len(zoom):
        raise ValueError("Zoom changed prediction cardinality")
    for before, after in zip(original, zoom, strict=True):
        if {k: v for k, v in before.items() if k != "bbox"} != {k: v for k, v in after.items() if k != "bbox"}:
            raise ValueError("Zoom changed a non-geometric detection decision")
        positive_geometry(before)
        positive_geometry(after)


def gate_predictions(original, zoom, annotations, selected_ids, policy=POLICY):
    """Compare the same original-to-observed-GT matches before and after zoom."""
    assert_prediction_contract(original, zoom)
    if any(p["image_id"] not in selected_ids for p in original):
        raise ValueError("Gate predictions include a non-holdout image")
    ground, annotation_ids = defaultdict(list), set()
    for ann in annotations:
        if ann.get("is_pseudo") or ann.get("ignore") or ann.get("iscrowd"):
            raise ValueError("Gate accepts only ordinary observed human annotations")
        if ann["id"] in annotation_ids:
            raise ValueError("Annotation IDs must be globally unique")
        annotation_ids.add(ann["id"])
        if ann["image_id"] in selected_ids:
            ground[(ann["image_id"], ann["category_id"])].append(ann)
    rows, used = [], set()
    for index in sorted(range(len(original)), key=lambda i: (-original[i]["score"], i)):
        pred = original[index]
        choices = [(overlap(pred["bbox"], ann["bbox"]), ann)
                   for ann in ground[(pred["image_id"], pred["category_id"])] if ann["id"] not in used]
        before, target = max(choices, key=lambda item: item[0], default=(0., None))
        if before < .5:
            continue
        used.add(target["id"])
        rows.append({"image_id": pred["image_id"], "annotation_id": target["id"],
                     "before_iou": before, "after_iou": overlap(zoom[index]["bbox"], target["bbox"])})
    count = len(rows)
    before = sum(r["before_iou"] for r in rows) / count if count else None
    after = sum(r["after_iou"] for r in rows) / count if count else None
    before_strict = sum(r["before_iou"] >= policy["strict_iou"] for r in rows)
    after_strict = sum(r["after_iou"] >= policy["strict_iou"] for r in rows)
    losses = sum(r["after_iou"] < .5 for r in rows)
    reasons = []
    if count < policy["minimum_observed_matches"]:
        reasons.append("insufficient_observed_matches")
    if before is None or after-before < policy["minimum_holdout_iou_gain"]:
        reasons.append("insufficient_mean_iou_gain")
    if after_strict < before_strict:
        reasons.append("strict_iou_matches_decreased")
    if losses:
        reasons.append("new_iou_below_half_losses")
    return {"accepted": not reasons, "rejection_reasons": reasons, "heldout_matches": count,
            "heldout_images": len(selected_ids), "before_mean_iou": before, "after_mean_iou": after,
            "before_strict_matches": before_strict, "after_strict_matches": after_strict,
            "new_iou_below_half_losses": losses, "rows": rows, "policy": policy,
            "limitation": "Detector saw these TRAIN images; this is a refinement gate, not an independent detector test."}


class _Predictor:
    """Lazy CUDA adapter; importing this module never imports torch."""
    def __init__(self, checkpoint, classes):
        import torch
        from rfdetr.training import RFDETRModelModule

        from coveragecv.training.runner import checkpoint_config

        if not torch.cuda.is_available():
            raise RuntimeError("Context zoom requires a cloud CUDA GPU")
        torch.set_num_threads(2)
        self.torch = torch
        self.classes = classes
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["class_names"] != classes:
            raise ValueError("Checkpoint and learner class layouts differ")
        self.variant = state.get("model_variant", "nano")
        self.mc = checkpoint_config(state, device="cuda")
        # Only the train configuration is needed; avoid another dataset lookup here.
        from rfdetr.config import TrainConfig
        tc = TrainConfig(dataset_dir="/unused", output_dir="/unused", dataset_file="roboflow", tensorboard=False, wandb=False,
                         mlflow=False, num_workers=0, pack_targets=False)
        self.module = RFDETRModelModule(self.mc, tc).to("cuda").eval()
        self.module.model.load_state_dict(state["model"], strict=True)
        self.reserved = 0

    def predict(self, images, image_ids):
        from coveragecv.training.tiled import _tensor
        torch = self.torch
        tensors = torch.stack([_tensor(image, self.mc.resolution) for image in images]).to("cuda")
        sizes = torch.tensor([[image.height, image.width] for image in images], device="cuda")
        with torch.inference_mode():
            outputs = self.module.postprocess(self.module.model(tensors), sizes)
        rows = []
        for image_id, result in zip(image_ids, outputs, strict=True):
            current = []
            for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(),
                                         result["scores"].tolist(), strict=True):
                if label == len(self.classes):
                    self.reserved += 1
                    continue
                if not 0 <= label < len(self.classes):
                    raise ValueError("Unmapped detector class")
                x1, y1, x2, y2 = box
                row = {"image_id": image_id, "category_id": label+1, "score": score,
                       "bbox": [x1, y1, x2-x1, y2-y1]}
                positive_geometry(row)  # Zero-area stock rows are valid unchanged outputs.
                current.append(row)
            rows.append(current)
        torch.cuda.synchronize()
        return rows

    def close(self):
        del self.module
        self.torch.cuda.empty_cache()


def infer_zoom(checkpoint, classes, images, image_root, policy=POLICY):
    from PIL import Image

    predictor = _Predictor(checkpoint, classes)
    original, zoom = [], []
    crop_count = refined = 0
    full_seconds = crop_seconds = 0.
    predictor.torch.cuda.synchronize()
    started = time.monotonic()
    try:
        for offset in range(0, len(images), 4):
            chunk = images[offset:offset+4]
            frames = []
            for im in chunk:
                with Image.open(safe_child(image_root, im["file_name"])) as source:
                    if source.size != (im["width"], im["height"]):
                        raise ValueError("Image dimensions disagree with manifest metadata")
                    frames.append(source.convert("RGB"))
            now = time.monotonic()
            predictions = predictor.predict(frames, [im["id"] for im in chunk])
            full_seconds += time.monotonic()-now
            for frame, im, base in zip(frames, chunk, predictions, strict=True):
                corrected = [{**row, "bbox": list(row["bbox"])} for row in base]
                plan = crop_plan(base, frame.size, policy)
                crop_count += len(plan)
                for start in range(0, len(plan), 4):
                    jobs = plan[start:start+4]
                    now = time.monotonic()
                    outputs = predictor.predict([frame.crop(window) for _, window in jobs], [im["id"]]*len(jobs))
                    crop_seconds += time.monotonic()-now
                    for (index, window), candidates in zip(jobs, outputs, strict=True):
                        corrected[index], changed = refine_anchor(base[index], candidates, window, frame.size, policy)
                        refined += changed
                assert_prediction_contract(base, corrected)
                original.extend(base)
                zoom.extend(corrected)
            for frame in frames:
                frame.close()
        predictor.torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        evidence = {"checkpoint_sha256": file_digest(checkpoint), "device": "cuda",
                    "resolution": predictor.mc.resolution, "model_variant": predictor.variant,
                    "images": len(images), "crop_count": crop_count,
                    "refined_boxes": refined, "elapsed_seconds": elapsed,
                    "full_frame_inference_seconds": full_seconds, "crop_inference_seconds": crop_seconds,
                    "reserved_slot_selections_all_passes": predictor.reserved,
                    "latency_scope": "Elapsed includes image IO, preprocessing, CUDA synchronization and association; excludes model load and metric scoring.",
                    "policy": policy}
        return original, zoom, evidence
    finally:
        predictor.close()


def _checkpoint(root, protocol, name):
    expected = protocol["checkpoints"][name]
    if isinstance(expected, dict):
        expected = expected["checkpoint_sha256"]
    path = root / "checkpoints" / f"{name}.pt"
    if file_digest(path) != expected:
        raise ValueError("Checkpoint differs from frozen protocol")
    return path


def run(root: Path, protocol: dict, output: Path) -> dict:
    if protocol.get("policy") != POLICY:
        raise ValueError("Context zoom policy differs from the fixed implementation")
    if (output / "result.json").exists():
        raise ValueError("Refusing to overwrite an existing zoom experiment")
    view = root / "view"
    manifest = verify(view)
    if manifest["kind"] != "training_view" or manifest["digest"] != protocol["view_digest"]:
        raise ValueError("Observed learner differs from protocol")
    data = read_json(view / "train/_annotations.coco.json")
    classes = read_json(view / "ontology.json")["classes"]
    selected = holdout_ids(data["images"], manifest)
    images = [im for im in data["images"] if im["id"] in selected]
    original, zoom, evidence = infer_zoom(_checkpoint(root, protocol, "aware"), classes, images, view / "train")
    gate = gate_predictions(original, zoom, data["annotations"], selected)
    gate.update(view_digest=manifest["digest"], inference=evidence, heldout_image_ids=sorted(selected))
    write_json(output / "train_selection.json", gate)
    for name, predictions in (("original", original), ("zoom", zoom)):
        write_json(output / "train" / f"{name}.json", {"predictions": predictions,
                   "view_digest": manifest["digest"], "split": "train", **evidence})
    result = {"status": "rejected_before_validation", "protocol_digest": digest(protocol),
              "gate": {k: v for k, v in gate.items() if k != "rows"}, "evaluations": {}, "failures": {}}
    if gate["accepted"]:
        # No complete-reference annotations are parsed until the TRAIN gate passes.
        from coveragecv.compiler import _jsonl
        from coveragecv.training.evaluate import score_predictions

        reference_root = root / "reference"
        reference_manifest = verify(reference_root)
        if reference_manifest["digest"] != protocol["reference_bundle_digest"]:
            raise ValueError("Complete reference differs from protocol")
        if read_json(reference_root / "ontology.json")["classes"] != classes:
            raise ValueError("Reference and learner ontology differ")
        if any(state not in ("exhaustive", "verified_absent")
               for row in _jsonl(reference_root / "coverage.jsonl") if row["split"] == "valid"
               for state in row["states"]):
            raise ValueError("Validation reference coverage is incomplete")
        reference = read_json(reference_root / "splits/valid.coco.json")
        for case in ("aware", "complete"):
            try:
                original, zoom, evidence = infer_zoom(_checkpoint(root, protocol, case), classes,
                                                       reference["images"], reference_root)
                result["evaluations"][case] = {}
                for name, predictions in (("original", original), ("zoom", zoom)):
                    metrics = score_predictions(predictions, reference)
                    passes = evidence["images"] + (evidence["crop_count"] if name == "zoom" else 0)
                    inference_seconds = evidence["full_frame_inference_seconds"] + (
                        evidence["crop_inference_seconds"] if name == "zoom" else 0)
                    payload = {"checkpoint_sha256": evidence["checkpoint_sha256"], "device": "cuda",
                               "resolution": evidence["resolution"], "model_variant": evidence["model_variant"],
                               "bundle_digest": reference_manifest["digest"], "split": "valid", "policy": POLICY,
                               "score_threshold": .25, "threshold_policy": "fixed before zoom pilot",
                               "postprocess": "stock outputs" if name == "original" else "fixed wide-context crop geometry",
                               "crop_count": 0 if name == "original" else evidence["crop_count"],
                               "refined_boxes": 0 if name == "original" else evidence["refined_boxes"],
                               "model_passes": passes, "inference_seconds": inference_seconds,
                               "mean_inference_seconds_per_image": inference_seconds/max(1, evidence["images"]),
                               "paired_collection_elapsed_seconds": evidence["elapsed_seconds"],
                               "latency_scope": "Own preprocessing and synchronized model inference; excludes image decoding, association, model load and metric scoring. Zoom includes crop creation.",
                               "metrics": metrics, "predictions": predictions}
                    write_json(output / case / f"{name}.json", payload)
                    result["evaluations"][case][name] = metrics
            except Exception as error:  # noqa: BLE001 -- retain the paid gate and other independent comparison
                result["failures"][case] = {"error_type": type(error).__name__, "error": str(error)}
        result["status"] = "evaluation_failed" if result["failures"] else "evaluated"
    write_json(output / "result.json", result)
    return result
