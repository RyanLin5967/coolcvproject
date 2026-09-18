"""Collect actual detector proposals on TRAIN images for a second localization stage.

The verified learner view supplies image metadata only to inference. No validation,
test, or hidden annotations are parsed. Collections are checkpoint-bound and resumable.
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from rfdetr.training import RFDETRModelModule

from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json
from coveragecv.training.runner import checkpoint_config, configs
from coveragecv.training.tiled import _tensor


def collect(view, checkpoint, output, *, device="mps"):
    manifest = verify(view)
    if manifest.get("kind") != "training_view":
        raise ValueError("Only verified learner training views are allowed")
    classes = read_json(view / "ontology.json")["classes"]
    images = read_json(view / "train/_annotations.coco.json")["images"]
    binding = {"kind": "train_image_proposals", "view_digest": manifest["digest"],
               "checkpoint_sha256": file_digest(checkpoint), "device": device,
               "flip_views": True, "score_floor": .05, "classes": classes}
    if output.exists():
        saved = read_json(output)
        if saved["binding"] != binding:
            raise ValueError("Saved proposal collection has a different binding")
        if saved.get("status") == "completed":
            return saved
    else:
        saved = {"binding": binding, "status": "running", "completed_images": [], "predictions": []}
    torch.set_num_threads(4)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["class_names"] != classes:
        raise ValueError("Checkpoint ontology mismatch")
    mc = checkpoint_config(state, device=device)
    _, tc = configs(view, output.parent)
    module = RFDETRModelModule(mc, tc).to(device).eval()
    module.model.load_state_dict(state["model"], strict=True)
    del state
    done = set(saved["completed_images"])
    remaining = [im for im in images if im["id"] not in done]
    started = time.monotonic()
    for offset in range(0, len(remaining), 2):
        chunk = remaining[offset:offset+2]
        tensors, sizes = [], []
        for im in chunk:
            with Image.open(safe_child(view / "train", im["file_name"])) as image:
                if image.size != (im["width"], im["height"]):
                    raise ValueError("Image size mismatch")
                tensors.append(_tensor(image, mc.resolution))
                sizes.append([im["height"], im["width"]])
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            raw = module.model(torch.cat([batch, batch.flip(-1)]))
            results = module.postprocess(raw, torch.tensor(sizes+sizes, device=device))
        for i, im in enumerate(chunk):
            for mirrored, result in ((False, results[i]), (True, results[len(chunk)+i])):
                for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(),
                                              result["scores"].tolist(), strict=True):
                    if label == len(classes) or score < binding["score_floor"]:
                        continue
                    if not 0 <= label < len(classes):
                        raise ValueError("Unmapped semantic class")
                    x1, y1, x2, y2 = box
                    if mirrored:
                        x1, x2 = im["width"]-x2, im["width"]-x1
                    if x2 <= x1 or y2 <= y1:
                        continue
                    saved["predictions"].append({"image_id": im["id"], "category_id": label+1,
                        "bbox": [x1, y1, x2-x1, y2-y1], "score": score, "view": "flip" if mirrored else "original"})
            saved["completed_images"].append(im["id"])
        saved["last_session_seconds"] = time.monotonic()-started
        write_json(output, saved)
        if offset == 0 or (offset+len(chunk)) % 20 == 0 or offset+len(chunk) == len(remaining):
            print({"checkpoint": checkpoint.parent.name, "completed": len(saved["completed_images"]),
                   "total": len(images), "seconds": round(saved["last_session_seconds"], 1)}, flush=True)
    saved["status"] = "completed"
    write_json(output, saved)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    collect(args.view, args.checkpoint, args.output, device=args.device)
