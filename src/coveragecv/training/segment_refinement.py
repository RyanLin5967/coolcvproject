"""Frozen SAM2 boundary proposals; geometry selection uses observed train boxes only."""
from collections import defaultdict
from pathlib import Path

from coveragecv.artifacts import read_json, safe_child, verify, write_json


def corners(boxes):
    import numpy as np
    values = np.asarray(boxes, dtype=np.float64).copy()
    values[:, 2:] += values[:, :2]
    return values


def corrected(original, segmented, bias, strength):
    import numpy as np
    size = np.tile(original[:, 2:]-original[:, :2], (1, 2))
    return original+strength*np.clip((segmented-original)/size+bias, -.15, .15)*size


def overlaps(first, second):
    import numpy as np
    intersection = np.maximum(0, np.minimum(first[:, 2:], second[:, 2:])-np.maximum(first[:, :2], second[:, :2])).prod(1)
    union = (first[:, 2:]-first[:, :2]).prod(1)+(second[:, 2:]-second[:, :2]).prod(1)-intersection
    return intersection/np.maximum(union, 1e-12)


class SegmentBoundaries:
    def __init__(self, model_dir):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.model = Sam2Model.from_pretrained(model_dir, local_files_only=True).to("cuda").eval()
        self.processor = Sam2Processor.from_pretrained(model_dir, local_files_only=True)

    def extract(self, rows, images, root):
        """Accept only image metadata and detector prompts; no target annotations."""
        import numpy as np
        from PIL import Image
        torch = self.torch
        original = corners([row["bbox"] for row in rows])
        output = original.copy()
        grouped = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[row["image_id"]].append(index)
        for ordinal, (image_id, indices) in enumerate(grouped.items()):
            item = images[image_id]
            with Image.open(safe_child(root, item["file_name"])) as source:
                image = source.convert("RGB")
            if image.size != (item["width"], item["height"]):
                raise ValueError("Image geometry mismatch")
            pixels = self.processor(images=image, return_tensors="pt").to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = self.model.get_image_embeddings(pixels["pixel_values"])
                for offset in range(0, len(indices), 16):
                    chunk = indices[offset:offset+16]
                    prompts = original[chunk].copy()
                    prompts[:, [0, 2]] = prompts[:, [0, 2]].clip(0, image.width)
                    prompts[:, [1, 3]] = prompts[:, [1, 3]].clip(0, image.height)
                    inputs = self.processor(input_boxes=[prompts.tolist()], original_sizes=pixels["original_sizes"],
                                            return_tensors="pt").to("cuda")
                    result = self.model(**inputs, image_embeddings=embeddings, multimask_output=True)
                    masks = self.processor.post_process_masks(result.pred_masks.float().cpu(), pixels["original_sizes"])[0]
                    best = result.iou_scores[0].argmax(-1).cpu()
                    masks = masks[torch.arange(len(chunk)), best].numpy()
                    for index, mask in zip(chunk, masks, strict=True):
                        ys, xs = np.where(mask)
                        if len(xs):
                            output[index] = [xs.min(), ys.min(), xs.max()+1, ys.max()+1]
            if (ordinal+1) % 25 == 0:
                print({"segmented_images": ordinal+1, "total": len(grouped)}, flush=True)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite segmentation geometry")
        return output


def run(root: Path, protocol, output: Path):
    import numpy as np
    import torch

    from coveragecv.training.evaluate import score_predictions
    from coveragecv.training.localization_cascade import matched_proposals, training_holdout
    torch.set_num_threads(2)
    view = root / "view"
    manifest = verify(view)
    rows, images, _, classes = matched_proposals(view, read_json(root / "proposals.json"))
    rows = [row for row in rows if row["view"] == "original"]
    held = training_holdout(rows, images, manifest)
    original = corners([row["bbox"] for row in rows])
    truth = corners([row["target"] for row in rows])
    segmenter = SegmentBoundaries("/sam2")
    # Strict boundary: training targets never enter the pretrained segmenter.
    segmented = segmenter.extract([{k: row[k] for k in ("image_id", "category_id", "bbox", "score")}
                                   for row in rows], images, view / "train")
    baseline = float(overlaps(original[held], truth[held]).mean())
    size = np.tile(original[:, 2:]-original[:, :2], (1, 2))
    residual = (truth-segmented)/size
    global_bias = np.clip(np.median(residual[~held], axis=0), -.15, .15)
    per_class = {c: global_bias.copy() for c in range(1, len(classes)+1)}
    for category in per_class:
        chosen = np.asarray([row["category_id"] == category for row in rows]) & ~held
        if chosen.sum() >= 15:
            per_class[category] = np.clip(np.median(residual[chosen], axis=0), -.15, .15)
    trials = []
    best = {"mean_iou": baseline, "strength": 0., "calibration": "none"}
    for calibration in ("none", "global", "class"):
        bias = (np.zeros_like(original) if calibration == "none" else np.broadcast_to(global_bias, original.shape)
                if calibration == "global" else np.stack([per_class[row["category_id"]] for row in rows]))
        for strength in (.25, .5, 1.):
            adjusted = corrected(original, segmented, bias, strength)
            mean = float(overlaps(adjusted[held], truth[held]).mean())
            trial = {"mean_iou": mean, "strength": strength, "calibration": calibration}
            trials.append(trial)
            if mean > best["mean_iou"]:
                best = trial
    accepted = best["mean_iou"] >= baseline+protocol["minimum_holdout_gain"]
    selection = {"baseline_holdout_iou": baseline, "best": best, "trials": trials,
                 "holdout_proposals": int(held.sum()), "fit_proposals": int((~held).sum()),
                 "status": "accepted_on_train_holdout" if accepted else "rejected_on_train_holdout",
                 "global_bias": global_bias.tolist(), "class_bias": {str(k): v.tolist() for k, v in per_class.items()},
                 "limitation": "Refiner holdout images were previously seen by the detector; no validation selection"}
    write_json(output / "train_selection.json", selection)
    print(selection, flush=True)
    if not accepted:
        return {"status": "rejected_on_train_holdout", "selection": selection}
    # This is the first access to validation labels, after the correction rule is frozen.
    bundle = Path("/reference")
    reference_manifest = verify(bundle)
    reference = read_json(bundle / "splits/valid.coco.json")
    train_hashes = {manifest["files"]["train/"+im["file_name"]] for im in images.values()}
    valid_hashes = {reference_manifest["files"][im["file_name"]] for im in reference["images"]}
    if train_hashes & valid_hashes:
        raise ValueError("Train/validation image overlap")
    val_images = {im["id"]: {k: im[k] for k in ("id", "file_name", "width", "height")}
                  for im in reference["images"]}
    results = {}
    for case, spec in protocol["evaluations"].items():
        baseline_eval = read_json(root / "evaluations" / f"{case}.json")
        if (baseline_eval["checkpoint_sha256"] != spec["checkpoint_sha256"]
                or baseline_eval["bundle_digest"] != reference_manifest["digest"] or baseline_eval["split"] != "valid"):
            raise ValueError("Validation prediction provenance mismatch")
        predictions = [dict(row) for row in baseline_eval["predictions"]]
        chosen = [i for i, row in enumerate(predictions) if row["score"] >= .05
                  and min(row["bbox"][2:]) > 0]
        prompts = [{k: predictions[i][k] for k in ("image_id", "category_id", "bbox", "score")} for i in chosen]
        proposal_boxes = corners([p["bbox"] for p in prompts])
        mask_boxes = segmenter.extract(prompts, val_images, bundle)
        bias = (np.zeros_like(proposal_boxes) if best["calibration"] == "none" else
                np.broadcast_to(global_bias, proposal_boxes.shape) if best["calibration"] == "global" else
                np.stack([per_class[p["category_id"]] for p in prompts]))
        revised = corrected(proposal_boxes, mask_boxes, bias, best["strength"])
        for i, box in zip(chosen, revised, strict=True):
            x1, y1, x2, y2 = box.tolist()
            predictions[i]["bbox"] = [x1, y1, x2-x1, y2-y1]
        result = {"checkpoint_sha256": spec["checkpoint_sha256"], "bundle_digest": reference_manifest["digest"],
                  "split": "valid", "device": "cuda", "metrics": score_predictions(predictions, reference),
                  "predictions": predictions, "selection": selection, "baseline_metrics": baseline_eval["metrics"]}
        write_json(output / f"{case}.json", result)
        results[case] = {k: v for k, v in result.items() if k not in ("predictions", "selection")}
        print({"case": case, "AP": result["metrics"]["AP"]}, flush=True)
    return {"status": "completed", "selection": selection, "results": results}
