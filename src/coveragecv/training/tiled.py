"""Fixed, image-only sliced inference experiment; original evaluator stays unchanged.

Mechanism references: https://github.com/obss/sahi and Roboflow Supervision's
InferenceSlicer. This small adapter uses stock RF-DETR postprocessing and
torchvision NMS, retaining this project's checkpoint and COCO bindings.
"""
import math
import time
from pathlib import Path

import torch
from PIL import Image
from rfdetr.training import RFDETRModelModule
from torchvision.ops import nms
from torchvision.transforms import functional as TF

from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json
from coveragecv.compiler import _jsonl
from coveragecv.training.evaluate import score_predictions
from coveragecv.training.runner import checkpoint_config, configs

PROTOCOL = {
    "name": "construction-fixed-384-tiles-v1",
    "tile_size": 384, "stride": 256, "internal_edge_margin_pixels": 2,
    "score_floor": .001, "nms_iou": .5, "score_threshold": .25,
    "include_full_frame": True, "batch_size": 4,
    "resize": "PIL default RGB BICUBIC, matching original evaluator",
    "selection": "fixed before tiled scoring; no validation setting sweep",
    "boundary_rule": "reject tile detections touching internal tile edges within 2 original pixels",
    "merge": "class-aware torchvision CPU NMS; score descending, original-order ties",
    "references": ["https://github.com/obss/sahi",
                   "https://supervision.roboflow.com/latest/detection/tools/inference_slicer/"],
}


def tile_windows(width, height, *, size=384, stride=256):
    """Cover the original image, anchor final windows to its edges, never read labels."""
    if min(width, height, size, stride) <= 0 or stride > size:
        raise ValueError("positive dimensions and stride <= tile size required")

    def starts(length):
        if length <= size:
            return [0]
        return sorted({*range(0, length-size+1, stride), length-size})

    return [(x, y, min(x+size, width), min(y+size, height))
            for y in starts(height) for x in starts(width)]


def map_tile_box(box, window, image_size, *, margin=2):
    """Reject internally truncated boxes; retain boxes against true image boundaries."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError("finite xyxy box required")
    x1, y1, x2, y2 = box
    left, top, right, bottom = window
    width, height = image_size
    if x2 <= x1 or y2 <= y1:
        return None
    if ((left > 0 and x1 <= margin) or (top > 0 and y1 <= margin)
            or (right < width and x2 >= right-left-margin)
            or (bottom < height and y2 >= bottom-top-margin)):
        return None
    x1, x2 = max(0., x1+left), min(float(width), x2+left)
    y1, y2 = max(0., y1+top), min(float(height), y2+top)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2-x1, y2-y1]


def merge_predictions(predictions, *, score_floor=.001, iou_threshold=.5):
    """Deterministic CPU NMS, isolated by image AND category, preserving real scores."""
    if not 0 <= score_floor <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("score and IoU thresholds must lie in [0, 1]")
    groups = {}
    for index, pred in enumerate(predictions):
        box, score = pred["bbox"], pred["score"]
        if len(box) != 4 or not all(math.isfinite(v) for v in [*box, score]):
            raise ValueError("finite xywh box and score required")
        if box[2] < 0 or box[3] < 0 or not 0 <= score <= 1:
            raise ValueError("invalid box dimensions or confidence")
        if score >= score_floor and box[2] > 0 and box[3] > 0:
            groups.setdefault((pred["image_id"], pred["category_id"]), []).append((index, pred))
    kept = []
    for rows in groups.values():
        rows.sort(key=lambda row: (-row[1]["score"], row[0]))
        boxes = torch.tensor([[p["bbox"][0], p["bbox"][1], p["bbox"][0]+p["bbox"][2],
                               p["bbox"][1]+p["bbox"][3]] for _, p in rows], dtype=torch.float64)
        # NMS uses scores only for ordering. Unique ranks make equal-score order explicit.
        ranks = torch.arange(len(rows), 0, -1, dtype=torch.float64)
        kept.extend(rows[index] for index in nms(boxes, ranks, iou_threshold).tolist())
    kept.sort(key=lambda row: (row[1]["image_id"], -row[1]["score"], row[0]))
    return [dict(pred) for _, pred in kept]


def _tensor(image, resolution):
    return TF.normalize(TF.to_tensor(image.convert("RGB").resize((resolution, resolution))),
                        [.485, .456, .406], [.229, .224, .225])


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _inference(module, images, root, resolution, class_count, *, device, tiled):
    # Only image metadata enters this function; no annotation or coverage lookup.
    jobs = [(item, window) for item in images for window in
            (tile_windows(item["width"], item["height"]) if tiled else [None])]
    predictions, reserved, rejected, batches = [], 0, 0, 0
    _sync(device)
    started = time.monotonic()
    for offset in range(0, len(jobs), PROTOCOL["batch_size"]):
        chunk = jobs[offset:offset+PROTOCOL["batch_size"]]
        tensors, sizes = [], []
        for item, window in chunk:
            with Image.open(safe_child(root, item["file_name"])) as source:
                if source.size != (item["width"], item["height"]):
                    raise ValueError("image dimensions disagree with metadata")
                image = source.convert("RGB")
                if window:
                    image = image.crop(window)
                sizes.append([image.height, image.width])
                tensors.append(_tensor(image, resolution))
        with torch.no_grad():
            raw = module.model(torch.stack(tensors).to(device))
            results = module.postprocess(raw, torch.tensor(sizes, device=device))
        batches += 1
        for (item, window), result in zip(chunk, results, strict=True):
            for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(),
                                         result["scores"].tolist(), strict=True):
                if label == class_count:
                    reserved += 1
                    continue
                if not 0 <= label < class_count:
                    raise ValueError("unmapped semantic class")
                if window:
                    if score < PROTOCOL["score_floor"]:
                        continue
                    bbox = map_tile_box(box, window, (item["width"], item["height"]))
                    if bbox is None:
                        rejected += 1
                        continue
                else:
                    x1, y1, x2, y2 = box
                    bbox = [x1, y1, x2-x1, y2-y1]
                predictions.append({"image_id": item["id"], "category_id": label+1,
                                    "bbox": bbox, "score": score})
        if batches % 20 == 0:
            print({"inference": "tiles" if tiled else "full_frame", "passes": offset+len(chunk),
                   "total": len(jobs)}, flush=True)
    _sync(device)
    return predictions, {"elapsed_seconds": time.monotonic()-started, "image_passes": len(jobs),
                         "batches": batches, "reserved_slot_selections": reserved,
                         "tile_boxes_rejected_at_boundaries_or_zero_area": rejected}


def device_parity_smoke(checkpoint: Path, bundle: Path, output: Path, *, device="mps"):
    """Compare real CPU/device raw predictions on the same four images, not metric tuning."""
    torch.set_num_threads(min(8, torch.get_num_threads()))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mc = checkpoint_config(state, device="cpu")
    _, tc = configs(bundle, output.parent)
    module = RFDETRModelModule(mc, tc).eval()
    module.model.load_state_dict(state["model"], strict=True)
    images = read_json(bundle / "splits/valid.coco.json")["images"][:4]
    tensors = []
    for item in images:
        with Image.open(safe_child(bundle, item["file_name"])) as image:
            tensors.append(_tensor(image, mc.resolution))
    inputs = torch.stack(tensors)
    with torch.no_grad():
        cpu = module.model(inputs)
        module.to(device)
        other = module.model(inputs.to(device))
    differences = {}
    for key in ("pred_logits", "pred_boxes"):
        a, b = cpu[key].cpu(), other[key].cpu()
        differences[key] = {"max_absolute_difference": float((a-b).abs().max()),
                            "mean_absolute_difference": float((a-b).abs().mean()),
                            "allclose_atol_002_rtol_002": torch.allclose(a, b, atol=.002, rtol=.002)}
    result = {"checkpoint_sha256": file_digest(checkpoint), "images": [i["id"] for i in images],
              "device": device, "differences": differences,
              "passed": all(d["allclose_atol_002_rtol_002"] for d in differences.values())}
    write_json(output, result)
    if not result["passed"]:
        raise ValueError("device raw-output parity smoke failed; inspect evidence before evaluation")
    return result


def evaluate_tiled(checkpoint: Path, bundle: Path, output: Path, *, baseline_evaluation: Path, device="mps"):
    """Save raw full-frame, NMS-only, and full+tiles controls for one fixed checkpoint."""
    manifest = verify(bundle)
    baseline = read_json(baseline_evaluation)
    checkpoint_sha = file_digest(checkpoint)
    if (baseline["bundle_digest"] != manifest["digest"] or baseline["checkpoint_sha256"] != checkpoint_sha
            or baseline["split"] != "valid"):
        raise ValueError("historical baseline does not bind checkpoint and validation bundle")
    if any(state not in ("exhaustive", "verified_absent") for row in _jsonl(bundle / "coverage.jsonl")
           if row["split"] == "valid" for state in row["states"]):
        raise ValueError("complete reference coverage required")
    classes = read_json(bundle / "ontology.json")["classes"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["class_names"] != classes:
        raise ValueError("checkpoint class mapping differs")
    mc = checkpoint_config(state, device=device)
    _, tc = configs(bundle, output)
    module = RFDETRModelModule(mc, tc).to(device).eval()
    module.model.load_state_dict(state["model"], strict=True)
    reference = read_json(bundle / "splits/valid.coco.json")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    common = {"checkpoint_sha256": checkpoint_sha, "bundle_digest": manifest["digest"], "split": "valid",
              "baseline_evaluation_sha256": file_digest(baseline_evaluation), "device": device,
              "resolution": mc.resolution, "protocol": PROTOCOL, "score_threshold": .25}
    full, full_timing = _inference(module, reference["images"], bundle, mc.resolution, len(classes),
                                   device=device, tiled=False)
    full_metrics = score_predictions(full, reference)
    write_json(output / "full_frame.json", {**common, "predictions": full, "metrics": full_metrics,
                                            "timing": full_timing})
    full_nms = merge_predictions(full)
    nms_metrics = score_predictions(full_nms, reference)
    write_json(output / "full_frame_nms.json", {**common, "predictions": full_nms, "metrics": nms_metrics,
                                                "timing": full_timing})
    tiles, tile_timing = _inference(module, reference["images"], bundle, mc.resolution, len(classes),
                                    device=device, tiled=True)
    merge_started = time.monotonic()
    merged = merge_predictions(full+tiles)
    merge_seconds = time.monotonic()-merge_started
    metrics = score_predictions(merged, reference)
    write_json(output / "tiled.json", {**common, "predictions": merged, "metrics": metrics,
                                       "timing": {"full": full_timing, "tiles": tile_timing,
                                                  "merge_seconds": merge_seconds}})
    result = {**common, "historical_metrics": baseline["metrics"], "full_frame_metrics": full_metrics,
              "full_frame_nms_metrics": nms_metrics, "tiled_metrics": metrics,
              "tiled_minus_fresh_full_AP_points": 100*(metrics["AP"]-full_metrics["AP"]),
              "fresh_full_minus_historical_AP_points": 100*(full_metrics["AP"]-baseline["metrics"]["AP"]),
              "timing": {"full": full_timing, "tiles": tile_timing, "merge_seconds": merge_seconds},
              "predictions": {"full": len(full), "full_nms": len(full_nms), "tiled": len(merged)},
              "scope": "all validation images and classes; no test scoring; one setting and one seed"}
    write_json(output / "comparison.json", result)
    return result
