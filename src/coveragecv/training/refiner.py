"""Experimental, class-agnostic localization from 224px image crops.

Training parses only human-observed train boxes in a verified learner view. A
proposal is a deterministically jittered human box; it is not a detector output
and therefore this experiment has a proposal-distribution shift to measure.
ResNet18 features and normalized proposal corners predict bounded center/log-size
corrections. Zero head output is exactly the identity, including padded crops.
Classification, confidence, prediction ordering, and low-score boxes are untouched.
"""
import hashlib
import math
import os
import tempfile
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import generalized_box_iou_loss
from torchvision.transforms import functional as TF

from coveragecv.artifacts import file_digest, read_json, safe_child, verify, write_json

MODEL_CONFIG = {"architecture": "torchvision.resnet18", "input_size": 224, "context": 1.5,
                "encoder_weights": "IMAGENET1K_V1", "center_limit": .5, "log_size_limit": .5,
                "head_width": 256, "spatial_pool": 4, "batch_norm": "frozen", "version": 1}


def _box(box):
    values = tuple(float(x) for x in box)
    if (len(values) != 4 or not all(math.isfinite(x) for x in values)
            or values[2] <= values[0] or values[3] <= values[1]):
        raise ValueError("boxes require finite, positive xyxy geometry")
    return values


def xywh_to_xyxy(box):
    if len(box) != 4:
        raise ValueError("COCO boxes require four coordinates")
    x, y, w, h = map(float, box)
    return _box((x, y, x+w, y+h))


@dataclass(frozen=True)
class CropGeometry:
    """Integer PIL crop bounds; outside-frame pixels are explicitly black padding."""
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def around(cls, proposal, context=1.5):
        x1, y1, x2, y2 = _box(proposal)
        if not math.isfinite(context) or context < 1:
            raise ValueError("crop context must be finite and at least one")
        cx, cy = (x1+x2)/2, (y1+y2)/2
        half_w, half_h = (x2-x1)*context/2, (y2-y1)*context/2
        crop = cls(math.floor(cx-half_w), math.floor(cy-half_h),
                   math.ceil(cx+half_w), math.ceil(cy+half_h))
        if crop.width*crop.height > 64_000_000 or max(crop.width, crop.height) > 32768:
            raise ValueError("proposal crop exceeds the allocation limit")
        return crop

    @property
    def width(self):
        return self.right-self.left

    @property
    def height(self):
        return self.bottom-self.top

    def normalize(self, box):
        x1, y1, x2, y2 = _box(box)
        return ((x1-self.left)/self.width, (y1-self.top)/self.height,
                (x2-self.left)/self.width, (y2-self.top)/self.height)

    def denormalize(self, box):
        x1, y1, x2, y2 = _box(box)
        return (x1*self.width+self.left, y1*self.height+self.top,
                x2*self.width+self.left, y2*self.height+self.top)

    def extract(self, image):
        # PIL crop uses exactly these integer bounds and pads with zero outside
        # the image; do not clip the rectangle because that changes the transform.
        return image.convert("RGB").crop((self.left, self.top, self.right, self.bottom))


def flip_box(box):
    """Horizontal flip of normalized xyxy corners in the crop coordinate system."""
    x1, y1, x2, y2 = _box(box)
    return (1-x2, y1, 1-x1, y2)


def prepare_crop(image, proposal, *, target=None, flipped=False):
    geometry = CropGeometry.around(proposal, MODEL_CONFIG["context"])
    crop = geometry.extract(image)
    proposal_in_crop = geometry.normalize(proposal)
    target_in_crop = geometry.normalize(target) if target is not None else None
    if flipped:
        crop = ImageOps.mirror(crop)
        proposal_in_crop = flip_box(proposal_in_crop)
        if target_in_crop is not None:
            target_in_crop = flip_box(target_in_crop)
    crop = crop.resize((MODEL_CONFIG["input_size"],)*2, Image.Resampling.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(crop), [.485, .456, .406], [.229, .224, .225])
    return (tensor, torch.tensor(proposal_in_crop, dtype=torch.float32),
            None if target_in_crop is None else torch.tensor(target_in_crop, dtype=torch.float32), geometry)


def bounded_deltas(raw):
    limits = raw.new_tensor([MODEL_CONFIG["center_limit"]]*2+[MODEL_CONFIG["log_size_limit"]]*2)
    return raw.tanh()*limits


def apply_deltas(proposals, deltas):
    """BoxCoder center/log-size deltas, expressed as additive edge changes.

    expm1(0) == 0 preserves proposal corners exactly for zero deltas. Coordinates
    may be normalized crop coordinates or original pixels; the algebra is affine
    equivariant. Bounded log-size deltas guarantee positive width and height.
    """
    wh = proposals[..., 2:]-proposals[..., :2]
    shift = deltas[..., :2]*wh
    half_change = .5*wh*torch.expm1(deltas[..., 2:])
    return proposals+torch.cat((shift-half_change, shift+half_change), dim=-1)


class CropBoxRefiner(nn.Module):
    def __init__(self, *, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet18(weights=weights)
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(512*4*4+4, MODEL_CONFIG["head_width"]), nn.ReLU(),
                                  nn.Linear(MODEL_CONFIG["head_width"], 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        # Keep ImageNet running statistics: correlated small object crops are
        # unsuitable for updating BatchNorm. Affine parameters can fine-tune.
        for layer in self.encoder.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def forward(self, crops, proposals):
        return bounded_deltas(self.head(torch.cat((self.encoder(crops), proposals), dim=-1)))


class HumanBoxDataset:
    """Stateless online jitter: sample index and seed fully specify each crop.

    Integrity verification hashes all learner payloads; only the train annotation
    file is parsed. No validation, reference, oracle, or test labels are consumed.
    """
    def __init__(self, view: Path, *, seed=20260917, pseudo_policy="reject"):
        if pseudo_policy not in ("reject", "filter"):
            raise ValueError("pseudo_policy must be reject or filter")
        self.view = Path(view)
        self.manifest = verify(self.view)
        if self.manifest.get("kind") != "training_view":
            raise ValueError("refiner training requires a verified learner training_view")
        data = read_json(self.view / "train/_annotations.coco.json")
        self.images = _image_index(data["images"])
        self.seed, self.pseudo_policy = seed, pseudo_policy
        self._images_cache = OrderedDict()
        self.instances, self.pseudo_boxes_filtered = [], 0
        annotation_ids = set()
        for ann in data["annotations"]:
            if ann["id"] in annotation_ids:
                raise ValueError("duplicate annotation ID")
            annotation_ids.add(ann["id"])
            if not isinstance(ann.get("is_pseudo", False), bool):
                raise TypeError("is_pseudo provenance must be a boolean")
            if ann.get("is_pseudo", False):
                if pseudo_policy == "reject":
                    raise ValueError("pseudo annotations are forbidden for human-only refiner training")
                self.pseudo_boxes_filtered += 1
                continue
            if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                raise ValueError("refiner training does not support crowd or ignored boxes")
            if ann["image_id"] not in self.images:
                raise ValueError("annotation refers to an unknown image")
            self.instances.append((ann["image_id"], xywh_to_xyxy(ann["bbox"])))
        if not self.instances:
            raise ValueError("refiner training requires at least one human-observed train box")
        for image in self.images.values():
            safe_child(self.view / "train", image["file_name"])

    def image(self, image_id):
        if image_id in self._images_cache:
            self._images_cache.move_to_end(image_id)
            return self._images_cache[image_id]
        metadata = self.images[image_id]
        with Image.open(safe_child(self.view / "train", metadata["file_name"])) as source:
            if source.size != (metadata["width"], metadata["height"]):
                raise ValueError("training image dimensions differ from their annotation metadata")
            image = source.convert("RGB")
        self._images_cache[image_id] = image
        if len(self._images_cache) > 32:
            self._images_cache.popitem(last=False)
        return image

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed+index)
        image_id, target = self.instances[int(torch.randint(len(self.instances), (), generator=generator))]
        x1, y1, x2, y2 = target
        w, h = x2-x1, y2-y1
        random_values = torch.rand(6, generator=generator).tolist()
        cx = (x1+x2)/2+(2*random_values[0]-1)*.12*w
        cy = (y1+y2)/2+(2*random_values[1]-1)*.12*h
        jw, jh = w*math.exp((2*random_values[2]-1)*.15), h*math.exp((2*random_values[3]-1)*.15)
        proposal = (cx-jw/2, cy-jh/2, cx+jw/2, cy+jh/2)
        # Already-correct detections must not be moved merely because the
        # synthetic training proposals were always perturbed.
        if random_values[5] < .25:
            proposal = target
        crop, proposal, target, _ = prepare_crop(self.image(image_id), proposal, target=target,
                                                flipped=random_values[4] < .5)
        return crop, proposal, target


def _image_index(images):
    result = {}
    for im in images:
        if im["id"] in result:
            raise ValueError("duplicate image ID")
        if any(isinstance(im[k], bool) or not isinstance(im[k], int) or im[k] <= 0
               for k in ("width", "height")):
            raise ValueError("image dimensions must be positive integers")
        result[im["id"]] = im
    return result


def _state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _save_checkpoint(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".refiner-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def train_refiner(view: Path, output: Path, *, device="cpu", seed=20260917, steps=1000, batch_size=32,
                  encoder_lr=1e-5, head_lr=1e-3, pseudo_policy="reject", max_seconds=None):
    """Train a fixed budget; return a receipt binding checkpoint and learner digest.

    Loss = 5 * mean SmoothL1(beta=1/9, normalized crop corners) + 2 * mean GIoU.
    No early stopping or validation selection. Only the final fixed-step model is
    saved. max_seconds aborts without publishing a completed checkpoint.
    """
    if (steps < 1 or batch_size < 2 or not all(math.isfinite(x) and x > 0 for x in (encoder_lr, head_lr))
            or (max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0))):
        raise ValueError("positive steps, learning rates, timeout, and batch_size >= 2 are required")
    dataset = HumanBoxDataset(view, seed=seed, pseudo_policy=pseudo_policy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("refiner.pt", "refiner_run.json", "refiner_protocol.json")):
        raise ValueError("refiner output already contains an experiment; choose a fresh output directory")
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("refiner device must be cpu, cuda, or mps")
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    protocol = {"kind": "class_agnostic_crop_refiner", "model_config": MODEL_CONFIG,
                "training_view_digest": dataset.manifest["digest"],
                "training_parent_digest": dataset.manifest.get("parent_digest"),
                "human_box_count": len(dataset.instances), "pseudo_policy": pseudo_policy,
                "training_images_sha256": sorted({
                    dataset.manifest["files"]["train/"+dataset.images[image_id]["file_name"]]
                    for image_id, _ in dataset.instances}),
                "pseudo_boxes_filtered": dataset.pseudo_boxes_filtered, "seed": seed, "steps": steps,
                "batch_size": batch_size, "device": str(target_device), "encoder_lr": encoder_lr,
                "head_lr": head_lr, "optimizer": "AdamW, weight_decay=0.0001, cosine to zero",
                "sampling": "uniform human boxes with replacement; stateless seed+sample_index",
                "jitter": {"center_fraction": .12, "log_size": .15, "horizontal_flip_probability": .5,
                           "identity_proposal_probability": .25},
                "loss": "5*mean SmoothL1(beta=1/9, normalized crop xyxy) + 2*mean GIoU",
                "validation_labels_used": False, "test_labels_used": False,
                "determinism": "fixed initialization/sample seeds; device kernels may be nondeterministic",
                "encoder_weights_url": ResNet18_Weights.IMAGENET1K_V1.url,
                "proposal_distribution": "synthetic jitter of observed human boxes; detector shift not validated"}
    write_json(output / "refiner_protocol.json", protocol)
    model = CropBoxRefiner(pretrained=True).to(target_device).train()
    protocol["initial_parameters_sha256"] = _state_digest(model.state_dict())
    protocol["initial_encoder_sha256"] = _state_digest(model.encoder.state_dict())
    write_json(output / "refiner_protocol.json", protocol)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": encoder_lr},
                                  {"params": model.head.parameters(), "lr": head_lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    started, curve = time.monotonic(), []
    for step in range(steps):
        if max_seconds is not None and time.monotonic()-started > max_seconds:
            raise TimeoutError("fixed-step refiner budget exceeded; no completed checkpoint published")
        samples = [dataset[step*batch_size+i] for i in range(batch_size)]
        crops, proposals, targets = [torch.stack(items).to(target_device) for items in zip(*samples)]
        deltas = model(crops, proposals)
        decoded = apply_deltas(proposals, deltas)
        l1 = F.smooth_l1_loss(decoded, targets, beta=1/9)
        giou = generalized_box_iou_loss(decoded, targets, reduction="mean")
        loss = 5*l1+2*giou
        if not torch.isfinite(loss):
            raise ValueError("nonfinite refiner training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step+1) % 25 == 0 or step+1 == steps:
            event = {"step": step+1, "loss": float(loss.detach().cpu()),
                     "smooth_l1": float(l1.detach().cpu()), "giou": float(giou.detach().cpu()),
                     "elapsed_seconds": time.monotonic()-started}
            curve.append(event)
            write_json(output / "refiner_progress.json", event)
            print(f"refiner {step+1}/{steps}: loss={event['loss']:.5f}", flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    metadata = {**protocol, "final_parameters_sha256": _state_digest(state)}
    checkpoint = output / "refiner.pt"
    _save_checkpoint(checkpoint, {"model": state, "metadata": metadata})
    receipt = {**metadata, "status": "completed", "checkpoint_sha256": file_digest(checkpoint),
               "checkpoint": str(checkpoint), "elapsed_seconds": time.monotonic()-started, "curve": curve}
    write_json(output / "refiner_run.json", receipt)
    return receipt


def refine_predictions(checkpoint: Path, images, image_root: Path, predictions, *, device="cpu",
                       confidence=.05, top_k=100, batch_size=32):
    """Refine eligible geometry with no reference annotations in this interface.

    The highest-scoring top_k eligible boxes per image are processed; ties retain
    input order. Boxes below confidence or outside top_k retain byte-equivalent
    JSON geometry. Finite zero-area detector boxes are also preserved, including
    high-confidence boxes, and never consume a refinement slot. Classes, scores,
    metadata, and input ordering are preserved.
    """
    if not math.isfinite(confidence) or not .05 <= confidence <= 1 or not 1 <= top_k <= 100 or batch_size < 1:
        raise ValueError("invalid refinement selection policy")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = state.get("metadata", {})
    if metadata.get("model_config") != MODEL_CONFIG:
        raise ValueError("unsupported refiner model configuration")
    if metadata.get("final_parameters_sha256") != _state_digest(state["model"]):
        raise ValueError("refiner parameter digest mismatch")
    model = CropBoxRefiner(pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    by_id = _image_index(images)
    grouped = defaultdict(list)
    result = [{**p, "bbox": list(p["bbox"])} for p in predictions]
    for index, p in enumerate(predictions):
        if p["image_id"] not in by_id or not math.isfinite(p["score"]) or not 0 <= p["score"] <= 1:
            raise ValueError("prediction references an unknown image or has an invalid score")
        box = p["bbox"]
        if (len(box) != 4 or not all(math.isfinite(value) for value in box)
                or box[2] < 0 or box[3] < 0):
            raise ValueError("prediction boxes require four finite coordinates and nonnegative width/height")
        # RF-DETR can emit finite degenerate boxes. They remain part of the
        # baseline prediction stream and scorer; do not drop or try to crop them.
        if p["score"] >= confidence and box[2] > 0 and box[3] > 0:
            grouped[p["image_id"]].append(index)
    for image_id, indices in grouped.items():
        im = by_id[image_id]
        with Image.open(safe_child(Path(image_root), im["file_name"])) as source:
            if source.size != (im["width"], im["height"]):
                raise ValueError("inference image dimensions differ from metadata")
            image = source.convert("RGB")
        chosen = sorted(indices, key=lambda i: -predictions[i]["score"])[:top_k]
        for offset in range(0, len(chosen), batch_size):
            chunk = chosen[offset:offset+batch_size]
            originals = [xywh_to_xyxy(predictions[i]["bbox"]) for i in chunk]
            samples = [prepare_crop(image, box) for box in originals]
            crops = torch.stack([s[0] for s in samples]).to(device)
            proposals = torch.stack([s[1] for s in samples]).to(device)
            with torch.inference_mode():
                deltas = model(crops, proposals).cpu()
            if not torch.isfinite(deltas).all():
                raise ValueError("refiner returned nonfinite corrections")
            # Decode in original pixel coordinates at float64 to avoid an
            # unnecessary normalized->pixel rounding cycle, including padding.
            refined = apply_deltas(torch.tensor(originals, dtype=torch.float64), deltas.double()).tolist()
            for i, delta, box in zip(chunk, deltas, refined):
                if not torch.count_nonzero(delta):
                    continue
                x1, y1, x2, y2 = _box(box)
                result[i]["bbox"] = [x1, y1, x2-x1, y2-y1]
    return result


def evaluate_refiner(checkpoint: Path, baseline_evaluation: Path, detector_checkpoint: Path,
                     bundle: Path, output: Path, *, device="cpu", confidence=.05, top_k=100):
    """Labels enter only here, after refinement, through the common COCO scorer."""
    from coveragecv.compiler import _jsonl
    from coveragecv.training.evaluate import score_predictions

    manifest = verify(bundle)
    baseline = read_json(baseline_evaluation)
    split = baseline.get("split")
    detector_sha = file_digest(detector_checkpoint)
    if (split not in ("valid", "test") or baseline.get("bundle_digest") != manifest["digest"]
            or baseline.get("checkpoint_sha256") != detector_sha):
        raise ValueError("baseline evaluation, detector, and reference binding mismatch")
    if any(state not in ("exhaustive", "verified_absent")
           for row in _jsonl(bundle / "coverage.jsonl") if row["split"] == split for state in row["states"]):
        raise ValueError("refiner evaluation requires complete reference coverage")
    reference = read_json(bundle / f"splits/{split}.coco.json")
    images = [{k: im[k] for k in ("id", "file_name", "width", "height")} for im in reference["images"]]
    refiner_metadata = torch.load(checkpoint, map_location="cpu", weights_only=True).get("metadata", {})
    train_hashes = refiner_metadata.get("training_images_sha256")
    if not train_hashes or not refiner_metadata.get("training_view_digest"):
        raise ValueError("refiner is missing training-data provenance")
    evaluation_hashes = {manifest["files"][im["file_name"]] for im in images}
    if evaluation_hashes.intersection(train_hashes):
        raise ValueError("refiner training images overlap evaluation images by exact file hash")
    refinement_started = time.perf_counter()
    refined = refine_predictions(checkpoint, images, bundle, baseline["predictions"], device=device,
                                 confidence=confidence, top_k=top_k)
    refinement_seconds = time.perf_counter()-refinement_started
    threshold = baseline.get("score_threshold", .25)
    baseline_metrics = score_predictions(baseline["predictions"], reference, threshold=threshold)
    result = {"kind": "experimental_crop_refinement", "bundle_digest": manifest["digest"], "split": split,
              "checkpoint_sha256": detector_sha, "refiner_checkpoint_sha256": file_digest(checkpoint),
              "baseline_evaluation_sha256": file_digest(baseline_evaluation),
              "refiner_training_view_digest": refiner_metadata["training_view_digest"],
              "refiner_human_box_count": refiner_metadata["human_box_count"],
              "train_evaluation_exact_image_overlap": False,
              "refinement_confidence": confidence, "refinement_top_k": top_k, "score_threshold": threshold,
              "refinement_device": str(device), "refinement_seconds": refinement_seconds,
              "refinement_timing_scope": "One complete pass including checkpoint load and crop preparation; excludes detector and COCO scoring",
              "evaluation_images": len(images),
              "changed_boxes": sum(a["bbox"] != b["bbox"] for a, b in zip(baseline["predictions"], refined)),
              "baseline_metrics": baseline_metrics, "metrics": score_predictions(refined, reference,
                                                                                     threshold=threshold),
              "predictions": refined, "classification_and_scores_changed": False,
              "interpretation": "Separate postprocessing experiment with added crop inference cost; not a seed mean"}
    write_json(output, result)
    return result
