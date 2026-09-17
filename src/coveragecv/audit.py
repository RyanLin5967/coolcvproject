"""Freeze board-configuration groups before any model comparison.

This controls repeated positions, not camera/domain independence: all images
still come from the same board setup. Reference annotations are used only by
the experiment preparer to form groups and split policy, never by the learner.
"""
import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .artifacts import digest, read_json, write_json
from .compiler import compile_bundle, materialize_view
from .datasets import ATTRIBUTION, CLASSES, REVISION, _project_split

PROTOCOL = "coverage-chess-group-v1"


def group_chess(root: Path, output: Path):
    root = root.resolve()
    original = root / "original"
    records = []
    for split in ("train", "valid", "test"):
        data, _ = _project_split(original, split)
        by_id = {im["id"]: [] for im in data["images"]}
        for a in data["annotations"]:
            by_id[a["image_id"]].append(a)
        for im in data["images"]:
            label = original / split / "labels" / (Path(im["file_name"]).stem+".txt")
            full = np.array([list(map(float, line.split())) for line in label.read_text().splitlines()
                             if line.strip()], dtype=float).reshape(-1, 5)
            records.append({"path": f"{split}/images/{im['file_name']}", "original_split": split,
                            "image": im, "boxes": by_id[im["id"]], "full": full})
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for i, a in enumerate(records):
        for j in range(i+1, len(records)):
            b = records[j]
            if len(a["full"]) != len(b["full"]) or Counter(a["full"][:, 0]) != Counter(b["full"][:, 0]):
                continue
            distances = []
            for label in set(a["full"][:, 0]):
                x, y = a["full"][a["full"][:, 0] == label, 1:3], b["full"][b["full"][:, 0] == label, 1:3]
                costs = np.linalg.norm(x[:, None, :]-y[None, :, :], axis=-1)
                row, col = linear_sum_assignment(costs)
                distances.extend(costs[row, col])
            if not distances or (max(distances) <= 0.035 and np.mean(distances) <= 0.018):
                parent[find(j)] = find(i)
                edges.append({"a": a["path"], "b": b["path"],
                              "max_center_distance": float(max(distances, default=0)),
                              "mean_center_distance": float(np.mean(distances)) if distances else 0})
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(find(i), []).append(record)
    assigned = {}
    group_rows = []
    for stratum in ("both", "one", "neither"):
        pool = []
        for items in groups.values():
            classes = {a["category_id"] for r in items for a in r["boxes"]}
            s = "both" if len(classes) == 2 else "one" if classes else "neither"
            if s == stratum:
                key = digest(sorted(r["path"] for r in items))
                pool.append((key, items))
        pool.sort(key=lambda p: hashlib.sha256((PROTOCOL+"\0"+"20260917\0"+p[0]).encode()).hexdigest())
        total = sum(len(items) for _, items in pool)
        targets = {"train": total*.7, "valid": total*.2, "test": total*.1}
        counts = {s: 0 for s in targets}
        for key, items in pool:
            split = max(targets, key=lambda s: targets[s]-counts[s])
            counts[split] += len(items)
            for record in items:
                assigned[record["path"]] = split
            group_rows.append({"group": key, "stratum": stratum, "split": split,
                               "original_splits": sorted({r["original_split"] for r in items}),
                               "paths": sorted(r["path"] for r in items)})
    grouped = root / "grouped"
    grouped.mkdir(exist_ok=True)
    categories = [{"id": i+1, "name": c} for i, c in enumerate(CLASSES)]
    split_data = {s: {"images": [], "annotations": [], "categories": categories} for s in ("train", "valid", "test")}
    for iid, record in enumerate(sorted(records, key=lambda r: r["path"])):
        split = assigned[record["path"]]
        dest = split_data[split]
        dest["images"].append({**record["image"], "id": iid, "file_name": record["path"]})
        for ann in record["boxes"]:
            dest["annotations"].append({**ann, "id": len(dest["annotations"])+1, "image_id": iid})
    train = split_data["train"]
    order = sorted(train["images"], key=lambda im: hashlib.sha256(
        (PROTOCOL+"\0"+"20260917\0"+im["file_name"]).encode()).hexdigest())
    partial_sources = []
    visible = 0
    for index, name in enumerate(("black", "white")):
        ids = {im["id"] for im in order[index::2]}
        data = {**train, "images": [im for im in train["images"] if im["id"] in ids],
                "annotations": [a for a in train["annotations"] if a["image_id"] in ids and
                                a["category_id"] == index+1]}
        write_json(grouped / f"{name}.json", data)
        visible += len(data["annotations"])
        partial_sources.append(_source(name, f"{name}.json", "train",
                               {c: "exhaustive" if i == index else "unknown" for i, c in enumerate(CLASSES)}))
    full_sources = []
    for split, data in split_data.items():
        write_json(grouped / f"{split}.json", data)
        full_sources.append(_source(split, f"{split}.json", split, {c: "exhaustive" for c in CLASSES}))
    for name, sources in (("partial", partial_sources+full_sources[1:]), ("complete", full_sources)):
        write_json(grouped / f"{name}-spec.json", {"classes": CLASSES, "sources": sources})
    bundles = {name: compile_bundle(grouped / f"{name}-spec.json", output / "bundles")
               for name in ("partial", "complete")}
    views = {name: materialize_view(bundle, output / "views") for name, bundle in bundles.items()}
    audit = {"protocol": PROTOCOL, "grouping": "same full-label class counts and Hungarian center-distance thresholds",
             "max_center_distance": 0.035, "mean_center_distance": 0.018,
             "groups": group_rows, "similarity_edges": edges,
             "groups_crossing_original_splits": sum(len(g["original_splits"]) > 1 for g in group_rows),
             "group_count": len(group_rows), "new_group_split_collisions": 0,
             "limitation": "configuration grouping is heuristic; same board/camera domain remains shared",
             "splits": {s: {"images": len(d["images"]), "boxes": len(d["annotations"]),
                            "boxes_per_class": dict(Counter(str(a["category_id"]) for a in d["annotations"]))}
                        for s, d in split_data.items()}}
    write_json(output / "audit/scene_groups.json", audit)
    old = output / "chess_experiment.json"
    if old.exists() and read_json(old).get("protocol") == "coverage-chess-v1":
        shutil.copyfile(old, output / "chess_original_experiment.json")
    experiment = {"protocol": PROTOCOL, "seed": 20260917, "classes": CLASSES, "attribution": ATTRIBUTION,
                  "visible_boxes": visible, "withheld_boxes": len(train["annotations"])-visible,
                  **{f"{name}_bundle": str(path.resolve()) for name, path in bundles.items()},
                  **{f"{name}_view": str(path.resolve()) for name, path in views.items()},
                  "scene_independence": audit["limitation"], "split_counts": audit["splits"],
                  "accuracy_status": "not run"}
    write_json(output / "chess_experiment.json", experiment)
    return experiment


def _source(name, annotations, split, coverage):
    return {"id": name, "revision": REVISION+":"+PROTOCOL, "annotations": annotations,
            "images": "../original", "split": split, "class_map": {c: c for c in CLASSES},
            "coverage": coverage, "evidence": "Frozen configuration groups and deterministic class withholding",
            "attribution": ATTRIBUTION}
