"""A class-conditioned cascade learned from real, human-matched train proposals.

Unlike the first crop-refiner experiment, supervision comes from detector outputs
matched to observed human boxes. Unknown/unmatched classes never become negatives.
An image-level TRAIN holdout chooses regularization; validation is scored only
after fitting. Neither reference annotations nor scoring callbacks enter inference.
"""
import hashlib
import math
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

from coveragecv.artifacts import read_json, safe_child, verify
from coveragecv.training.evaluate import iou
from coveragecv.training.refiner import CropBoxRefiner, CropGeometry, _state_digest, xywh_to_xyxy

PROTOCOL = {
    "version": 1, "context": 1.5, "input_size": 224, "minimum_match_iou": .5,
    "minimum_confidence": .05, "proposals_per_object_view": 2,
    "projection_dimensions": 256, "seed": 20260918,
    "ridge_candidates": [1., 10., 100., 1000.], "strength_candidates": [0., .5, 1.],
    "holdout_fraction": .2, "max_corner_correction": .15,
    "selection": "mean matched IoU on deterministic image-level TRAIN holdout only",
    "score_or_class_changes": False, "validation_labels_used_for_training": False,
    "references": [
        "https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html",
        "https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html",
    ],
}


class ExactSpatialPool(nn.Module):
    """The same adaptive mean bins as PyTorch; works on MPS for 7 -> 4 pooling."""
    def forward(self, x):
        rows = []
        for i in range(4):
            cells = [x[..., math.floor(i*x.shape[-2]/4):math.ceil((i+1)*x.shape[-2]/4),
                       math.floor(j*x.shape[-1]/4):math.ceil((j+1)*x.shape[-1]/4)].mean((-2, -1))
                     for j in range(4)]
            rows.append(torch.stack(cells, -1))
        return torch.stack(rows, -2)


def matched_proposals(view, proposals):
    manifest = verify(view)
    if manifest.get("kind") != "training_view":
        raise ValueError("Matching requires a learner training view")
    if (proposals.get("status") != "completed" or proposals["binding"]["view_digest"] != manifest["digest"]):
        raise ValueError("Incomplete or mismatched proposal collection")
    data = read_json(view / "train/_annotations.coco.json")
    images = {im["id"]: im for im in data["images"]}
    classes = read_json(view / "ontology.json")["classes"]
    if proposals["binding"]["classes"] != classes:
        raise ValueError("Proposal ontology differs from learner")
    if set(proposals["completed_images"]) != set(images):
        raise ValueError("Proposal collection does not cover the learner train images")
    by_class = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("is_pseudo", False):
            raise ValueError("Only human-observed targets may supervise localization")
        if ann.get("ignore", 0) or ann.get("iscrowd", 0):
            raise ValueError("Ignored/crowd annotations are unsupported")
        xywh_to_xyxy(ann["bbox"])
        by_class[(ann["image_id"], ann["category_id"])].append(ann)
    matches = defaultdict(list)
    for p in proposals["predictions"]:
        if p["image_id"] not in images or not 1 <= p["category_id"] <= len(classes):
            raise ValueError("Proposal references an unknown image or class")
        xywh_to_xyxy(p["bbox"])
        if not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1:
            raise ValueError("Invalid proposal confidence")
        if p["score"] < PROTOCOL["minimum_confidence"]:
            continue
        candidates = by_class[(p["image_id"], p["category_id"])]
        overlap, ann = max(((iou(p["bbox"], a["bbox"]), a) for a in candidates),
                           key=lambda item: item[0], default=(0, None))
        if overlap >= PROTOCOL["minimum_match_iou"]:
            matches[(ann["id"], p["view"])].append({**p, "target": ann["bbox"],
                                                    "annotation_id": ann["id"], "input_iou": overlap})
    rows = []
    for key in sorted(matches):
        rows.extend(sorted(matches[key], key=lambda p: -p["score"])[:PROTOCOL["proposals_per_object_view"]])
    if not rows:
        raise ValueError("No human-matched train proposals")
    return rows, images, manifest, classes


def training_holdout(rows, images, manifest):
    """All views/proposals of the same exact image belong to the same fold."""
    held = {}
    for identity, image in images.items():
        sha = manifest["files"]["train/"+image["file_name"]]
        hashed = hashlib.sha256((str(PROTOCOL["seed"])+sha).encode()).digest()
        held[identity] = int.from_bytes(hashed[:4], "big") / 2**32 < PROTOCOL["holdout_fraction"]
    mask = np.asarray([held[row["image_id"]] for row in rows])
    if mask.sum() < 20 or (~mask).sum() < 100:
        raise ValueError("Not enough independent train/holdout proposals")
    return mask


class FeatureExtractor:
    def __init__(self, checkpoint, classes, *, device="mps"):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["metadata"]["final_parameters_sha256"] != _state_digest(state["model"]):
            raise ValueError("Feature encoder checkpoint failed integrity check")
        refiner = CropBoxRefiner(pretrained=False)
        refiner.load_state_dict(state["model"], strict=True)
        refiner.encoder.avgpool = ExactSpatialPool()
        self.encoder = refiner.encoder.to(device).eval()
        self.device, self.classes, self.metadata = device, classes, state["metadata"]
        generator = torch.Generator().manual_seed(PROTOCOL["seed"])
        self.projection = (torch.randn(8192, PROTOCOL["projection_dimensions"], generator=generator)
                           / math.sqrt(PROTOCOL["projection_dimensions"])).to(device)

    def extract(self, rows, images, root):
        """This boundary accepts images and proposals only, never reference boxes."""
        visual, geometrical = [], []
        cached_images = {}
        for offset in range(0, len(rows), 32):
            tensors, geometry = [], []
            for p in rows[offset:offset+32]:
                im = images[p["image_id"]]
                if im["id"] not in cached_images:
                    with Image.open(safe_child(root, im["file_name"])) as source:
                        if source.size != (im["width"], im["height"]):
                            raise ValueError("Inference image dimensions differ")
                        cached_images[im["id"]] = source.convert("RGB")
                box = xywh_to_xyxy(p["bbox"])
                crop = CropGeometry.around(box, PROTOCOL["context"]).extract(cached_images[im["id"]])
                crop = crop.resize((PROTOCOL["input_size"],)*2, Image.Resampling.BILINEAR)
                tensors.append(TF.normalize(TF.to_tensor(crop), [.485, .456, .406], [.229, .224, .225]))
                x, y, w, h = p["bbox"]
                g = np.array([(x+w/2)/im["width"], (y+h/2)/im["height"],
                              w/im["width"], h/im["height"], math.log(w/h), p["score"]])
                one_hot = np.eye(len(self.classes))[p["category_id"]-1]
                geometry.append(np.concatenate([g, g*g, one_hot, np.outer(one_hot, g).ravel()]))
            with torch.inference_mode():
                embedding = self.encoder(torch.stack(tensors).to(self.device)) @ self.projection
            visual.append(embedding.cpu().numpy())
            geometrical.extend(geometry)
            if offset % 320 == 0:
                print({"crop_features": offset+len(tensors), "total": len(rows)}, flush=True)
        return np.concatenate([np.concatenate(visual), np.asarray(geometrical)], axis=1).astype(np.float64)


def correction_targets(rows):
    result = []
    for row in rows:
        box = np.asarray(xywh_to_xyxy(row["bbox"]))
        target = np.asarray(xywh_to_xyxy(row["target"]))
        w, h = row["bbox"][2:]
        result.append((target-box)/np.asarray([w, h, w, h]))
    return np.asarray(result)


def apply_corrections(rows, corrections, strength):
    if len(rows) != len(corrections) or not np.isfinite(corrections).all() or not 0 <= strength <= 1:
        raise ValueError("Invalid correction predictions")
    result = []
    for row, delta in zip(rows, corrections, strict=True):
        box = np.asarray(xywh_to_xyxy(row["bbox"]))
        w, h = row["bbox"][2:]
        bounded = np.clip(delta, -PROTOCOL["max_corner_correction"], PROTOCOL["max_corner_correction"])
        x1, y1, x2, y2 = box+strength*bounded*np.asarray([w, h, w, h])
        result.append({**row, "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)]})
    return result


def fit_ridge(x, y, alpha):
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-5] = 1.
    z = np.concatenate([(x-center)/scale, np.ones((len(x), 1))], axis=1)
    regularization = np.eye(z.shape[1])*alpha
    regularization[-1, -1] = 1e-8
    weights = np.linalg.solve(z.T@z+regularization, z.T@y)
    return {"center": center, "scale": scale, "weights": weights, "alpha": alpha}


def predict_ridge(model, x):
    z = np.concatenate([(x-model["center"])/model["scale"], np.ones((len(x), 1))], axis=1)
    return z@model["weights"]


def train_cascade(features, rows, holdout):
    y = correction_targets(rows)
    selected = [row for row, held in zip(rows, holdout, strict=True) if held]
    baseline = float(np.mean([row["input_iou"] for row in selected]))
    trials, best = [], (baseline, 1., 0.)
    for alpha in PROTOCOL["ridge_candidates"]:
        fitted = fit_ridge(features[~holdout], y[~holdout], alpha)
        predictions = predict_ridge(fitted, features[holdout])
        for strength in PROTOCOL["strength_candidates"]:
            corrected = apply_corrections(selected, predictions, strength)
            quality = float(np.mean([iou(row["bbox"], row["target"]) for row in corrected]))
            trials.append({"alpha": alpha, "strength": strength, "holdout_mean_iou": quality})
            if quality > best[0]+1e-8:
                best = quality, alpha, strength
    model = fit_ridge(features, y, best[1])
    model["strength"] = best[2]
    return model, {"baseline_holdout_mean_iou": baseline, "selected_holdout_mean_iou": best[0],
                   "selected_alpha": best[1], "selected_strength": best[2], "trials": trials,
                   "train_proposals": int((~holdout).sum()), "holdout_proposals": int(holdout.sum())}
