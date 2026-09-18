"""Coverage-constrained teacher labels with horizontal-flip agreement.

The teacher sees only train images and observed labels in a verified learner view.
Pseudo boxes never turn unknown coverage into negative evidence. Human labels win
over conflicting predictions. Every accepted box retains its teacher provenance.
"""
import shutil
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from rfdetr.training import RFDETRModelModule
from torchvision.ops import batched_nms, box_iou
from torchvision.transforms import functional as TF

from coveragecv.artifacts import file_digest, publish, read_json, stage, verify, write_json
from coveragecv.compiler import _jsonl
from coveragecv.schema import NEGATIVE_ALLOWED
from coveragecv.training.runner import configs


def restore_teacher_view(original_view: Path, run_folder: Path, output: Path):
    """Reconstruct and verify the exact remote learner artifact from returned provenance."""
    original = verify(original_view)
    expected = read_json(run_folder / "learner_manifest.json")
    if expected.get("original_view_digest") != original["digest"]:
        raise ValueError("teacher learner was derived from a different observed dataset")
    staging = stage(output)
    try:
        for name in original["files"]:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_view / name, destination)
        write_json(staging / "train/_annotations.coco.json", read_json(run_folder / "derived-train.coco.json"))
        write_json(staging / "pseudo-labels.json", read_json(run_folder / "pseudo-labels.json"))
        for name, sha in expected["files"].items():
            if file_digest(staging / name) != sha:
                raise ValueError("reconstructed teacher data does not match the remote training manifest")
        metadata = {k: v for k, v in expected.items() if k not in ("files", "digest")}
        restored = publish(staging, output, metadata)
        if restored.name != expected["digest"]:
            raise ValueError("reconstructed teacher artifact has a different immutable identity")
        return restored
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def consensus_boxes(first, mirrored, observed, states, *, confidence=.7, agreement_iou=.6):
    """Match two original-coordinate predictions, suppress duplicates/conflicts."""
    def prepare(rows):
        valid = [p for p in rows if 0 <= p["label"] < len(states)
                 and states[p["label"]] not in NEGATIVE_ALLOWED and p["score"] >= confidence
                 and p["box"][2] > p["box"][0] and p["box"][3] > p["box"][1]]
        if not valid:
            return []
        keep = batched_nms(torch.tensor([p["box"] for p in valid], dtype=torch.float32),
                           torch.tensor([p["score"] for p in valid]),
                           torch.tensor([p["label"] for p in valid]), .5)
        return [valid[i] for i in keep.tolist()]

    a, b = prepare(first), prepare(mirrored)
    accepted, used = [], set()
    for p in a:
        candidates = [(float(box_iou(torch.tensor([p["box"]], dtype=torch.float32),
                                     torch.tensor([q["box"]], dtype=torch.float32))[0, 0]), j)
                      for j, q in enumerate(b) if j not in used and q["label"] == p["label"]]
        score, index = max(candidates, default=(0, -1))
        if score < agreement_iou:
            continue
        q = b[index]
        # Average locations only after both views agree on class and geometry.
        box = [(x*p["score"]+y*q["score"])/(p["score"]+q["score"])
               for x, y in zip(p["box"], q["box"])]
        occupied = observed+[row["box"] for row in accepted]
        if occupied and float(box_iou(torch.tensor([box]), torch.tensor(occupied, dtype=torch.float32)).max()) >= .5:
            continue
        used.add(index)
        accepted.append({"box": box, "label": p["label"], "score": min(p["score"], q["score"]),
                         "agreement_iou": score})
    return accepted


def mine_training_labels(view: Path, teacher: Path, output: Path, *, device="cpu",
                         confidence=.7, agreement_iou=.6):
    if not 0 < confidence <= 1 or not 0 < agreement_iou <= 1:
        raise ValueError("teacher confidence and agreement must be in (0, 1]")
    manifest = verify(view)
    if manifest["kind"] != "training_view":
        raise ValueError("pseudo-label mining accepts learner views only, never a complete reference bundle")
    classes = read_json(view / "ontology.json")["classes"]
    state = torch.load(teacher, map_location="cpu", weights_only=True)
    if state.get("class_names") != classes:
        raise ValueError("teacher and learner ontologies differ")
    mc, tc = configs(view, output)
    if "model_config" in state:
        from coveragecv.training.runner import checkpoint_config
        mc = checkpoint_config(state, device=device)
    model = RFDETRModelModule(mc, tc).eval()
    model.model.load_state_dict(state["model"], strict=True)
    model.to(device)
    coverage = {row["image_id"]: row["states"] for row in _jsonl(view / "coverage.jsonl")
                if row["split"] == "train"}
    data = read_json(view / "train/_annotations.coco.json")
    observed = {}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        observed.setdefault(ann["image_id"], []).append([x, y, x+w, y+h])
    accepted = []
    for offset in range(0, len(data["images"]), 4):
        chunk = data["images"][offset:offset+4]
        images, sizes = [], []
        for im in chunk:
            with Image.open(view / "train" / im["file_name"]) as image:
                tensor = TF.to_tensor(image.convert("RGB").resize((mc.resolution, mc.resolution)))
                images.append(TF.normalize(tensor, [.485, .456, .406], [.229, .224, .225]))
                sizes.append([im["height"], im["width"]])
        batch = torch.stack(images).to(device)
        with torch.inference_mode():
            raw = model.model(torch.cat([batch, batch.flip(-1)]))
            predictions = model.postprocess(raw, torch.tensor(sizes+sizes, device=device))
        for i, im in enumerate(chunk):
            variants = []
            for mirrored, pred in ((False, predictions[i]), (True, predictions[len(chunk)+i])):
                rows = []
                for box, label, score in zip(pred["boxes"].tolist(), pred["labels"].tolist(), pred["scores"].tolist()):
                    x1, y1, x2, y2 = box
                    if mirrored:
                        x1, x2 = im["width"]-x2, im["width"]-x1
                    clipped = [max(0., min(im["width"], x1)), max(0., min(im["height"], y1)),
                               max(0., min(im["width"], x2)), max(0., min(im["height"], y2))]
                    rows.append({"box": clipped, "label": label, "score": score})
                variants.append(rows)
            for row in consensus_boxes(*variants, observed.get(im["id"], []), coverage[im["id"]],
                                        confidence=confidence, agreement_iou=agreement_iou):
                accepted.append({"image_id": im["id"], **row})
        print(f"teacher inspected {min(offset+4,len(data['images']))}/{len(data['images'])} train images; "
              f"accepted {len(accepted)} boxes", flush=True)
    teacher_sha = file_digest(teacher)
    next_id = max((a["id"] for a in data["annotations"]), default=0)+1
    for index, row in enumerate(accepted):
        x1, y1, x2, y2 = row["box"]
        data["annotations"].append({"id": next_id+index, "image_id": row["image_id"],
            "category_id": row["label"]+1, "bbox": [x1, y1, x2-x1, y2-y1],
            "area": (x2-x1)*(y2-y1), "iscrowd": 0, "is_pseudo": True})
    audit = {"teacher_sha256": teacher_sha, "parent_view_digest": manifest["digest"],
             "policy": "coverage-constrained horizontal-flip consensus", "confidence": confidence,
             "agreement_iou": agreement_iou, "train_images": len(data["images"]),
             "accepted_boxes": len(accepted), "boxes_per_class": dict(Counter(classes[p["label"]] for p in accepted)),
             "coverage_promoted": False, "validation_or_test_images_used": False,
             "complete_reference_labels_used": False, "predictions": accepted}
    staging = stage(output)
    try:
        for name in manifest["files"]:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(view / name, target)
        write_json(staging / "train/_annotations.coco.json", data)
        write_json(staging / "pseudo-labels.json", audit)
        metadata = {k: v for k, v in manifest.items() if k not in ("files", "digest")}
        metadata.update(teacher_sha256=teacher_sha, original_view_digest=manifest["digest"],
                        supervision="observed-plus-teacher-pseudo-labels")
        return publish(staging, output, metadata)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
