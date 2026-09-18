"""Immutable partial-label-derived sampling plans for matched detector training.

Plans mix exactly half uniform full frames and half class-balanced observed-anchor
crops. All arms replay the same image pixels, rectangles, and flips. The complete
arm may retain additional labels, but those labels cannot influence the plan.
"""
import math
from collections import defaultdict
from pathlib import Path
from types import MethodType

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ToDtype, ToImage

from coveragecv.artifacts import digest, read_json, safe_child, verify

POLICY = {"version": 1, "full_frame_fraction": .5, "anchor_sampling": "uniform class then uniform human box",
          "anchor_target_pixels": 96, "minimum_crop_side": 160, "center_shift_fraction": .1,
          "horizontal_flip_probability": .5, "oversized_anchor": "full-frame fallback",
          "extra_random_scale_or_crop": False}


def _training_data(view):
    view = Path(view)
    manifest = verify(view)
    if manifest.get("kind") != "training_view":
        raise ValueError("object crop planning requires a verified learner training_view")
    data = read_json(view / "train/_annotations.coco.json")
    classes = read_json(view / "ontology.json")["classes"]
    expected_categories = [{"id": i+1, "name": name} for i, name in enumerate(classes)]
    if [{"id": c["id"], "name": c["name"]} for c in data["categories"]] != expected_categories:
        raise ValueError("training categories differ from the canonical ontology")
    images = []
    seen = set()
    for im in sorted(data["images"], key=lambda row: row["id"]):
        if im["id"] in seen:
            raise ValueError("duplicate training image ID")
        seen.add(im["id"])
        if any(isinstance(im[key], bool) or not isinstance(im[key], int) or im[key] < 1
               for key in ("width", "height")):
            raise ValueError("image dimensions must be positive integers")
        path = safe_child(view / "train", im["file_name"])
        with Image.open(path) as image:
            if image.size != (im["width"], im["height"]):
                raise ValueError("training image pixels differ from their dimension metadata")
        filename = path.relative_to(view.resolve()).as_posix()
        images.append({key: im[key] for key in ("id", "file_name", "width", "height")}
                      | {"sha256": manifest["files"][filename]})
    if not images:
        raise ValueError("object crops require training images")
    return manifest, data, classes, images


def _human_anchors(data, images):
    image_index = {im["id"]: im for im in images}
    category_ids = {cat["id"] for cat in data["categories"]}
    seen, by_class = set(), defaultdict(list)
    for ann in sorted(data["annotations"], key=lambda row: row["id"]):
        if ann["id"] in seen:
            raise ValueError("duplicate anchor annotation ID")
        seen.add(ann["id"])
        if not isinstance(ann.get("is_pseudo", False), bool):
            raise TypeError("is_pseudo provenance must be boolean")
        if ann.get("is_pseudo", False):
            raise ValueError("object crop anchors must contain human labels only; pseudo labels are forbidden")
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            raise ValueError("object crop anchors do not support crowd or ignored labels")
        if ann["image_id"] not in image_index or ann["category_id"] not in category_ids:
            raise ValueError("anchor refers to an unknown image or category")
        box = ann["bbox"]
        if len(box) != 4 or not all(math.isfinite(x) for x in box) or box[2] <= 0 or box[3] <= 0:
            raise ValueError("anchor boxes require finite positive geometry")
        im = image_index[ann["image_id"]]
        x, y, w, h = map(float, box)
        if x < -1e-6 or y < -1e-6 or x+w > im["width"]+1e-6 or y+h > im["height"]+1e-6:
            raise ValueError("anchor box exceeds original image bounds")
        by_class[ann["category_id"]].append(ann)
    if not by_class:
        raise ValueError("object crops require at least one human-observed anchor")
    return by_class


def _randint(high, generator):
    return int(torch.randint(high, (), generator=generator))


def _anchor_crop(image, annotation, resolution, generator):
    width, height = image["width"], image["height"]
    x, y, w, h = map(float, annotation["bbox"])
    x1, y1 = max(0., x), max(0., y)
    x2, y2 = min(float(width), x+w), min(float(height), y+h)
    side = min(max(math.ceil(max(w, h)*resolution/POLICY["anchor_target_pixels"]),
                   POLICY["minimum_crop_side"]), width, height)
    # Rectangular images may contain a tall/wide anchor that fits no in-frame
    # square. A full-frame fallback keeps that anchor intact rather than clipping it.
    if x2-x1 > side or y2-y1 > side:
        return [0, 0, height, width], "anchor_larger_than_square"
    low_x, high_x = max(0, math.ceil(x2-side)), min(math.floor(x1), width-side)
    low_y, high_y = max(0, math.ceil(y2-side)), min(math.floor(y1), height-side)
    if low_x > high_x or low_y > high_y:
        return [0, 0, height, width], "no_integer_square_contains_anchor"
    shifts = (torch.rand(2, generator=generator)*2-1).tolist()
    left = round((x1+x2-side)/2+shifts[0]*POLICY["center_shift_fraction"]*side)
    top = round((y1+y2-side)/2+shifts[1]*POLICY["center_shift_fraction"]*side)
    left, top = max(low_x, min(high_x, left)), max(low_y, min(high_y, top))
    return [top, left, side, side], None


def build_crop_plan(partial_view: Path, samples=8000, seed=20260917, resolution=512):
    """Read only partial human train labels; freeze every geometry/sampling choice."""
    if (isinstance(samples, bool) or not isinstance(samples, int) or samples < 2 or samples % 2
            or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            or resolution != 512):
        raise ValueError("crop plans require a positive even sample count, integer seed, and resolution 512")
    manifest, data, classes, images = _training_data(partial_view)
    by_class = _human_anchors(data, images)
    image_index = {im["id"]: im for im in images}
    categories = sorted(by_class)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for kind in ("full_frame", "anchor_crop"):
        for _ in range(samples//2):
            anchor_id = category_id = fallback = None
            if kind == "full_frame":
                im = images[_randint(len(images), generator)]
                rectangle = [0, 0, im["height"], im["width"]]
            else:
                category_id = categories[_randint(len(categories), generator)]
                anchors = by_class[category_id]
                ann = anchors[_randint(len(anchors), generator)]
                anchor_id, im = ann["id"], image_index[ann["image_id"]]
                rectangle, fallback = _anchor_crop(im, ann, resolution, generator)
            rows.append({"image_id": im["id"], "kind": kind, "anchor_annotation_id": anchor_id,
                         "anchor_category_id": category_id, "crop_tlhw": rectangle, "fallback": fallback,
                         "flip": bool(torch.rand((), generator=generator) < .5)})
    order = torch.randperm(samples, generator=generator).tolist()
    rows = [{"sample_index": i, **rows[j]} for i, j in enumerate(order)]
    blueprint = {"kind": "partial_human_object_crop_plan", "policy": POLICY, "seed": seed,
                 "resolution": resolution, "samples": samples, "source_view_digest": manifest["digest"],
                 "classes": classes, "images": images,
                 "anchor_counts": {str(i+1): len(by_class.get(i+1, [])) for i in range(len(classes))},
                 "train_only": True, "validation_or_test_labels_used": False, "rows": rows}
    return {**blueprint, "digest": digest(blueprint)}


class ObjectCropDataset(Dataset):
    """Replay a verified plan; supervision comes from this arm's original dataset."""
    def __init__(self, dataset, plan):
        from rfdetr.datasets._torchvision import RandomHorizontalFlip, Resize
        from rfdetr.datasets.transforms import Normalize

        self.dataset, self.plan = dataset, plan
        self.indices = {image_id: i for i, image_id in enumerate(dataset.ids)}
        self.images = {im["id"]: im for im in plan["images"]}
        self.dataset._transforms = None
        self.dataset._draft_size = None
        self.resize = Resize((plan["resolution"], plan["resolution"]))
        self.flip = RandomHorizontalFlip(p=1.)
        self.to_image, self.to_float = ToImage(), ToDtype(torch.float32, scale=True)
        self.normalize = Normalize()

    def __len__(self):
        return self.plan["samples"]

    def __getitem__(self, index):
        from rfdetr.datasets._torchvision import crop

        row = self.plan["rows"][index]
        image, target = self.dataset[self.indices[row["image_id"]]]
        metadata = self.images[row["image_id"]]
        if image.size != (metadata["width"], metadata["height"]):
            raise ValueError("upstream decode changed the original crop coordinate system")
        if int(target["image_id"].item()) != row["image_id"]:
            raise ValueError("upstream preparation changed the source image ID")
        image, target = crop(image, target, *row["crop_tlhw"])
        if row["flip"]:
            # p=1 is deterministic, but upstream still draws one random value.
            # Restore global CPU RNG so augmentation does not change model RNG.
            with torch.random.fork_rng(devices=[]):
                image, target = self.flip(image, target)
        image, target = self.resize(image, target)
        image, target = self.normalize(self.to_float(self.to_image(image)), target)
        if int(target["image_id"].item()) != row["image_id"]:
            raise ValueError("object crop transform changed the coverage lookup ID")
        return image, target


def _planned_train_dataloader(dm):
    world_size = getattr(dm.trainer, "world_size", 1) if dm.trainer else 1
    if world_size != 1 or dm.train_config.grad_accum_steps != 1:
        raise ValueError("fixed crop replay currently supports one device and accumulation one")
    dataset = dm._dataset_train
    batch = dm._resolve_batch_size()
    if len(dataset) % batch:
        raise ValueError("crop plan length must be divisible by batch size")
    return DataLoader(dataset, batch_size=batch, shuffle=False, drop_last=False, num_workers=0,
                      collate_fn=dm._collate_fn, pin_memory=dm._pin_memory,
                      generator=torch.Generator().manual_seed(dataset.plan["seed"]))


def install_object_crops(dm, view: Path, partial_view: Path, plan):
    """Install deterministic replay on one already-set-up RF-DETR DataModule.

    Verification regenerates the plan from its declared partial train source. The
    current arm must have exactly the same train images and ontology. Its additional
    labels never enter sampling. Validation/test datasets and coverage tables stay
    untouched. Call after the runner's existing assert_data_contract check.
    """
    if dm._dataset_train is None or isinstance(dm._dataset_train, ObjectCropDataset):
        raise ValueError("install object crops exactly once after dm.setup('fit')")
    if plan.get("digest") != digest({key: value for key, value in plan.items() if key != "digest"}):
        raise ValueError("crop plan digest mismatch")
    expected = build_crop_plan(partial_view, samples=plan["samples"], seed=plan["seed"],
                               resolution=plan["resolution"])
    if expected != plan:
        raise ValueError("crop plan differs from its verified partial-label source")
    _, data, classes, images = _training_data(view)
    if images != plan["images"] or classes != plan["classes"]:
        raise ValueError("training arm image identities, content hashes, dimensions, or ontology differ from crop plan")
    # Every observed anchor must exist unmodified in all arms, including complete
    # labels. Annotation IDs can differ across views; compare semantic geometry.
    _, partial, _, _ = _training_data(partial_view)
    present = {(a["image_id"], a["category_id"], tuple(a["bbox"])) for a in data["annotations"]
               if not a.get("is_pseudo", False)}
    if any((a["image_id"], a["category_id"], tuple(a["bbox"])) not in present for a in partial["annotations"]):
        raise ValueError("training arm is missing or altered a partial human anchor")
    if set(dm._dataset_train.ids) != {im["id"] for im in images}:
        raise ValueError("upstream training dataset image IDs differ from verified view")
    expected_mapping = {i+1: i for i in range(len(classes))}
    if dm._dataset_train.cat2label != expected_mapping:
        raise ValueError("upstream training category mapping differs from verified view")
    # Bind the actual dataset being adapted, not merely the supplied view path.
    if Path(dm._dataset_train.root).resolve() != (Path(view)/"train").resolve():
        raise ValueError("upstream training dataset root differs from verified view")
    for im in data["images"]:
        actual = dm._dataset_train.coco.imgs[im["id"]]
        if any(actual[key] != im[key] for key in ("file_name", "width", "height")):
            raise ValueError("upstream training metadata differs from verified view")
    wrapper = ObjectCropDataset(dm._dataset_train, plan)
    dm._dataset_train = wrapper
    dm.train_dataloader = MethodType(_planned_train_dataloader, dm)
    return wrapper
