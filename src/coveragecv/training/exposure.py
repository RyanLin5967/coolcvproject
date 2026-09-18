"""Fixed image exposure derived only from a partial learner's observed labels."""
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from coveragecv.artifacts import digest, read_json, verify


def build_exposure_plan(view: Path, *, samples: int, seed: int, repeat: bool, threshold=.20):
    manifest = verify(view)
    if manifest["kind"] != "training_view" or samples < 1 or not 0 < threshold <= 1:
        raise ValueError("Exposure planning requires a training view and positive sample budget")
    data = read_json(view / "train/_annotations.coco.json")
    ids = sorted(im["id"] for im in data["images"])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Training images must have unique IDs")
    present = defaultdict(set)
    for ann in data["annotations"]:
        if ann.get("is_pseudo", False) or ann["image_id"] not in ids:
            raise ValueError("Exposure must use observed training labels only")
        present[ann["image_id"]].add(ann["category_id"])
    counts = Counter(c for cats in present.values() for c in cats)
    factors = {c: max(1., math.sqrt(threshold * len(ids) / count)) for c, count in counts.items()}
    weights = [max((factors[c] for c in present[iid]), default=1.) if repeat else 1. for iid in ids]
    # Same replacement mechanism and independent RNG for uniform and weighted controls.
    sequence = random.Random(seed).choices(ids, weights=weights, k=samples)
    image_hashes = {str(im["id"]): manifest["files"]["train/" + im["file_name"]] for im in data["images"]}
    plan = {"source_view_digest": manifest["digest"], "source_image_hashes": image_hashes,
            "sampling": "repeat_factor" if repeat else "uniform_replacement", "seed": seed,
            "threshold": threshold, "observed_image_counts": dict(counts),
            "class_repeat_factors": factors, "sample_ids": sequence,
            "selection": "Training observed labels only; frequencies measure supervision exposure, not true prevalence.",
            "augmentation": "Unchanged stock augmentation; no targeted object crops"}
    # Canonical JSON requires string keys.
    plan["observed_image_counts"] = {str(k): v for k, v in counts.items()}
    plan["class_repeat_factors"] = {str(k): v for k, v in factors.items()}
    return {**plan, "digest": digest(plan)}


def bind_exposure(dm, view, plan):
    """Replay identical source images in partial and full arms, without changing labels."""
    from types import MethodType

    from torch.utils.data import DataLoader, Subset

    if digest({k: v for k, v in plan.items() if k != "digest"}) != plan["digest"]:
        raise ValueError("Exposure plan digest mismatch")
    manifest = verify(view)
    data = read_json(Path(view) / "train/_annotations.coco.json")
    hashes = {str(im["id"]): manifest["files"]["train/" + im["file_name"]] for im in data["images"]}
    if hashes != plan["source_image_hashes"]:
        raise ValueError("Control images differ from the partial-derived exposure plan")
    dataset = dm._dataset_train
    index = {iid: i for i, iid in enumerate(dataset.ids)}
    indices = [index[iid] for iid in plan["sample_ids"]]
    if len(indices) % dm._resolve_batch_size():
        raise ValueError("Exposure budget must contain complete batches")

    def loader(self):
        return DataLoader(Subset(dataset, indices), batch_size=self._resolve_batch_size(),
                          shuffle=False, drop_last=False, num_workers=0,
                          collate_fn=self._collate_fn, pin_memory=self._pin_memory)

    dm.train_dataloader = MethodType(loader, dm)
