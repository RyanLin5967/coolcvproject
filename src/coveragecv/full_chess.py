"""A 13-class task on the same frozen scene groups, preserving the source ontology."""
from collections import Counter
from pathlib import Path

from PIL import Image

from .artifacts import read_json, write_json
from .compiler import compile_bundle, materialize_view
from .datasets import ATTRIBUTION, REVISION

CLASSES = ["bishop", "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
           "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook"]


def prepare_full_chess(root=Path("data/chess"), output=Path("artifacts/full_chess")):
    root, output = root.resolve(), output.resolve()
    frozen = read_json(output.parent / "audit/scene_groups.json")
    assignments = {path: group["split"] for group in frozen["groups"] for path in group["paths"]}
    black_source = {im["file_name"] for im in read_json(root / "grouped/black.json")["images"]}
    categories = [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]
    data = {s: {"images": [], "annotations": [], "categories": categories} for s in ("train", "valid", "test")}
    corrections = []
    for iid, path in enumerate(sorted(assignments)):
        split = assignments[path]
        image_path = root / "original" / path
        with Image.open(image_path) as image:
            width, height = image.size
        original_split = path.split("/")[0]
        label_path = root / "original" / original_split / "labels" / (image_path.stem+".txt")
        data[split]["images"].append({"id": iid, "file_name": path, "width": width, "height": height})
        seen = set()
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            label, cx, cy, w, h = map(float, line.split())
            if not label.is_integer() or not 0 <= label < len(CLASSES):
                raise ValueError("unexpected source category in the pinned chess dataset")
            x1, y1 = max(0, (cx-w/2)*width), max(0, (cy-h/2)*height)
            x2, y2 = min(width, (cx+w/2)*width), min(height, (cy+h/2)*height)
            identity = (int(label), x1, y1, x2-x1, y2-y1)
            if identity in seen:
                corrections.append({"image": path, "label_file": str(label_path.relative_to(root)),
                    "line": line_number, "original_label": line, "action": "remove_exact_duplicate_box",
                    "reason": "Explicit prepared-dataset variant; identical class and geometry already retained"})
                continue
            seen.add(identity)
            data[split]["annotations"].append({"id": len(data[split]["annotations"])+1, "image_id": iid,
                "category_id": int(label)+1, "bbox": [x1, y1, x2-x1, y2-y1], "iscrowd": 0,
                "area": (x2-x1)*(y2-y1)})
    folder = root / "all-pieces"
    folder.mkdir(exist_ok=True)

    def source(name, annotations, split, covered):
        return {"id": name, "revision": REVISION+":coverage-chess-all-pieces-v1", "annotations": annotations,
            "images": "../original", "split": split, "class_map": dict(zip(CLASSES, CLASSES)),
            "coverage": {c: "exhaustive" if c in covered else "unknown" for c in CLASSES},
            "evidence": "Frozen board-configuration splits; deliberate source/class withholding; exact duplicate boxes explicitly removed with source_corrections.json. The generic bishop class is preserved without guessing its color.",
            "attribution": ATTRIBUTION.replace("two-pawn projection", "full 13-class ontology")}

    complete_sources = []
    for split, content in data.items():
        write_json(folder / f"{split}.json", content)
        complete_sources.append(source(split, f"{split}.json", split, CLASSES))
    partial_sources, visible = [], 0
    for name, predicate, covered in (("black-source", lambda im: im["file_name"] in black_source,
                                      [c for c in CLASSES if c.startswith("black-") or c == "bishop"]),
                                     ("white-source", lambda im: im["file_name"] not in black_source,
                                      [c for c in CLASSES if c.startswith("white-")])):
        selected = [im for im in data["train"]["images"] if predicate(im)]
        ids = {im["id"] for im in selected}
        annotations = [a for a in data["train"]["annotations"] if a["image_id"] in ids and CLASSES[a["category_id"]-1] in covered]
        visible += len(annotations)
        write_json(folder / f"{name}.json", {"images": selected, "annotations": annotations, "categories": categories})
        partial_sources.append(source(name, f"{name}.json", "train", covered))
    bundles, views = {}, {}
    write_json(output / "source_corrections.json", {"corrections": corrections, "count": len(corrections),
                                                   "original_download_modified": False})
    for name, sources in (("partial", partial_sources+complete_sources[1:]), ("complete", complete_sources)):
        spec = folder / f"{name}-spec.json"
        write_json(spec, {"classes": CLASSES, "sources": sources})
        bundles[name] = compile_bundle(spec, output / "bundles")
        views[name] = materialize_view(bundles[name], output / "views")
    experiment = {"protocol": "coverage-chess-all-pieces-v1", "seed": 20260917, "classes": CLASSES,
        "partial_spec": str(folder / "partial-spec.json"),
        "visible_boxes": visible, "withheld_boxes": len(data["train"]["annotations"])-visible,
        "attribution": complete_sources[0]["attribution"],
        "limitation": "Same camera/board domain as the pawn pilot, not an independent dataset or domain.",
        "split_counts": {s: {"images": len(d["images"]), "boxes": len(d["annotations"]),
                             "class_support": dict(Counter(CLASSES[a["category_id"]-1] for a in d["annotations"]))}
                         for s, d in data.items()},
        **{f"{name}_bundle": str(path) for name, path in bundles.items()},
        **{f"{name}_view": str(path) for name, path in views.items()}}
    write_json(output / "experiment.json", experiment)
    return experiment
