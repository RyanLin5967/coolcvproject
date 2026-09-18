"""Pinned, independently sourced industrial detection task with grouped splits."""
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image

from coveragecv.artifacts import file_digest, safe_child, write_json
from coveragecv.compiler import compile_bundle, materialize_view

REPO = "LibreYOLO/construction-safety-gsnvb"
REVISION = "342e545489a6b5f76d6c8225f1ef2629c5a4770a"
CLASSES = ["helmet", "no-helmet", "no-vest", "person", "vest"]
ATTRIBUTION = ("Construction Safety / Roboflow100 dataset1, CC BY4.0; "
               "https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1 ; "
               "pinned LibreYOLO mirror; coveragecv adds grouped split repair, COCO conversion and deliberate class withholding.")


def download(root: Path):
    original = root / "original"
    original.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
        entries = []
        while url:
            response = client.get(url)
            response.raise_for_status()
            entries += response.json()
            url = response.links.get("next", {}).get("url")
        files = [e for e in entries if e["type"] == "file"]
        if len(files) > 3000 or sum(e["size"] for e in files) > 200_000_000:
            raise ValueError("dataset inventory exceeds the reviewed download envelope")
        write_json(root / "source_inventory.json", {"repository": REPO, "revision": REVISION,
                                                   "entries": files})

        def fetch(entry):
            path = safe_child(original, entry["path"])
            if not path.exists():
                response = client.get(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(entry['path'])}")
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix+".part")
                temp.write_bytes(response.content)
                temp.replace(path)
            data = path.read_bytes()
            if entry.get("lfs"):
                actual, expected = hashlib.sha256(data).hexdigest(), entry["lfs"]["oid"]
            else:
                actual = hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
                expected = entry["oid"]
            if actual != expected or len(data) != entry["size"]:
                raise ValueError(f"source content hash mismatch: {entry['path']}")
            return {"path": entry["path"], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(fetch, files))
    write_json(root / "download_manifest.json", {"repository": REPO, "revision": REVISION, "files": records})
    return original


def prepare(root=Path("data/construction"), output=Path("artifacts/construction"), *, fetch=True):
    root, output = root.resolve(), output.resolve()
    original = download(root) if fetch else root / "original"
    images, parents, corrections = [], {}, []
    for path in sorted(original.glob("*/images/*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        relative = path.relative_to(original).as_posix()
        with Image.open(path) as decoded:
            decoded.load()
            width, height = decoded.size
            hashes = []
            for im in (decoded, decoded.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
                pixels = np.asarray(im.convert("L").resize((9, 8)))
                bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
                hashes.append(sum(int(bit) << i for i, bit in enumerate(bits)))
        labels = []
        label_path = path.parent.parent / "labels" / (path.stem+".txt")
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            label, cx, cy, w, h = map(float, line.split())
            if not label.is_integer() or not 0 <= label < len(CLASSES):
                raise ValueError("unexpected construction category")
            x1, y1 = max(0., (cx-w/2)*width), max(0., (cy-h/2)*height)
            x2, y2 = min(width, (cx+w/2)*width), min(height, (cy+h/2)*height)
            item = [int(label)+1, x1, y1, x2-x1, y2-y1]
            if x2 <= x1 or y2 <= y1 or item in labels:
                corrections.append({"path": str(label_path.relative_to(original)), "line": line_number,
                                    "label": line, "action": "remove_exact_duplicate_or_degenerate_box"})
                continue
            labels.append(item)
        images.append({"path": relative, "split": relative.split("/")[0], "width": width, "height": height,
                       "sha": file_digest(path), "name": path.name.split(".rf.")[0],
                       "hashes": hashes, "labels": labels})
    if len(images) != 1206:
        raise ValueError("unexpected source image count")
    parents = list(range(len(images)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    relations = Counter()
    for i, a in enumerate(images):
        for j in range(i):
            b = images[j]
            why = ("exact_bytes" if a["sha"] == b["sha"] else "original_filename" if a["name"] == b["name"] else
                   "near_duplicate_dhash" if min((x^y).bit_count() for x in a["hashes"] for y in b["hashes"]) <= 4 else None)
            if why:
                parents[find(i)] = find(j)
                relations[why] += 1
    groups = {}
    for i, row in enumerate(images):
        groups.setdefault(find(i), []).append(row)
    priority = {"train": 0, "valid": 1, "test": 2}
    chosen, audit_groups = [], []
    seen = {}
    for group in groups.values():
        split = max((row["split"] for row in group), key=priority.get)
        audit_groups.append({"split": split, "original_splits": sorted({r["split"] for r in group}),
                             "paths": [r["path"] for r in group]})
        for row in group:
            if row["sha"] in seen:
                earlier = seen[row["sha"]]
                if sorted(row["labels"]) != sorted(earlier["labels"]):
                    raise ValueError("Identical source image bytes have conflicting annotations; review before training")
                corrections.append({"path": row["path"], "action": "deduplicate_identical_image",
                                    "retained": earlier["path"]})
                continue
            seen[row["sha"]] = row
            chosen.append({**row, "split": split})
    categories = [{"id": i+1, "name": c} for i, c in enumerate(CLASSES)]
    data = {split: {"images": [], "annotations": [], "categories": categories} for split in priority}
    for iid, row in enumerate(sorted(chosen, key=lambda r: r["path"])):
        split = row["split"]
        data[split]["images"].append({"id": iid, "file_name": row["path"], "width": row["width"], "height": row["height"]})
        for label, x, y, w, h in row["labels"]:
            data[split]["annotations"].append({"id": len(data[split]["annotations"])+1, "image_id": iid,
                "category_id": label, "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0})
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    def source(name, split, covered):
        return {"id": name, "revision": REVISION+":grouped-v1", "annotations": name+".json",
                "images": "../original", "split": split, "class_map": dict(zip(CLASSES, CLASSES)),
                "coverage": {c: "exhaustive" if c in covered else "unknown" for c in CLASSES},
                "evidence": "Deliberate class-family withholding; groups prevent recognized exact/near duplicates crossing splits. Reference completeness is inherited from the source, not independently certified.",
                "attribution": ATTRIBUTION}

    complete = []
    for split, rows in data.items():
        write_json(prepared / f"{split}.json", rows)
        complete.append(source(split, split, CLASSES))
    order = sorted(data["train"]["images"], key=lambda im: hashlib.sha256(("construction-coverage-v1\0"+im["file_name"]).encode()).hexdigest())
    partial, visible = [], 0
    for index, (name, covered) in enumerate((("head-source", {"helmet", "no-helmet"}),
                                            ("vest-source", {"vest", "no-vest"}),
                                            ("person-source", {"person"}))):
        selected = order[index::3]
        ids = {im["id"] for im in selected}
        annotations = [a for a in data["train"]["annotations"] if a["image_id"] in ids and CLASSES[a["category_id"]-1] in covered]
        visible += len(annotations)
        write_json(prepared / f"{name}.json", {"images": selected, "annotations": annotations, "categories": categories})
        partial.append(source(name, "train", covered))
    paths = {}
    for name, sources in (("partial", partial+complete[1:]), ("complete", complete)):
        spec = prepared / f"{name}-spec.json"
        write_json(spec, {"classes": CLASSES, "sources": sources})
        bundle = compile_bundle(spec, output / "bundles")
        paths[f"{name}_bundle"] = str(bundle)
        paths[f"{name}_view"] = str(materialize_view(bundle, output / "views"))
    write_json(output / "group_audit.json", {"group_count": len(groups), "relations": dict(relations),
        "groups_crossing_original_splits": sum(len(g["original_splits"]) > 1 for g in audit_groups),
        "policy": "Group source filenames, exact bytes and normal/mirrored dHash distance <=4; retain highest holdout split test>valid>train.",
        "limitation": "Heuristic near-duplicate grouping, not independently established scene/camera isolation.",
        "groups": audit_groups, "corrections": corrections})
    experiment = {"protocol": "construction-coverage-v1", "classes": CLASSES, "attribution": ATTRIBUTION,
                  "partial_spec": str(prepared / "partial-spec.json"), "visible_boxes": visible,
                  "withheld_boxes": len(data["train"]["annotations"])-visible,
                  "split_counts": {s: {"images": len(d["images"]), "boxes": len(d["annotations"])} for s, d in data.items()},
                  **paths}
    write_json(output / "experiment.json", experiment)
    return experiment
