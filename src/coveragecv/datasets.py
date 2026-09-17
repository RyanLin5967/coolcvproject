"""Pinned, attributed chess pilot with reproducible class withholding."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx
from PIL import Image, ImageDraw, ImageOps

from .artifacts import file_digest, write_json
from .compiler import compile_bundle, materialize_view

REPO = "LibreYOLO/chess-pieces-mjzgj"
REVISION = "17e0d3e7c76bea701ad623b0f7b13bec8859ff80"
CLASSES = ["black-pawn", "white-pawn"]
ATTRIBUTION = ("Chess Pieces by Joseph Nelson and Brad Dwyer; RF100 / Roboflow version 1; "
               "CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ ; "
               "two-pawn projection, COCO conversion and deliberate class withholding by coveragecv.")


def download_chess(root: Path):
    original = root / "original"
    original.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
        entries = []
        while url:
            response = client.get(url)
            response.raise_for_status()
            entries.extend(response.json())
            url = response.links.get("next", {}).get("url")
        files = [e for e in entries if e["type"] == "file"]
        if len(files) != 583 or sum(e["size"] for e in files) != 13_776_642:
            raise ValueError("pinned source inventory differs from the reviewed dataset")

        def fetch(entry):
            relative = entry["path"]
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("unsafe dataset path")
            path = original / relative
            if not path.exists():
                response = client.get(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(relative)}")
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            data = path.read_bytes()
            if len(data) != entry["size"]:
                raise ValueError(f"source size mismatch: {relative}")
            if entry.get("lfs"):
                actual = hashlib.sha256(data).hexdigest()
                expected = entry["lfs"]["oid"]
            else:
                actual = hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
                expected = entry["oid"]
            if actual != expected:
                raise ValueError(f"source digest mismatch: {relative}")
            return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

        rows = list(ThreadPoolExecutor(max_workers=8).map(fetch, files))
    write_json(root / "download_manifest.json", {"repository": REPO, "revision": REVISION,
                                                 "files": sorted(rows, key=lambda r: r["path"])})
    return original


def _project_split(original: Path, split: str):
    images, annotations, paths = [], [], {}
    for iid, image in enumerate(sorted((original / split / "images").glob("*"))):
        with Image.open(image) as decoded:
            decoded.load()
            width, height = decoded.size
        images.append({"id": iid, "file_name": image.name, "width": width, "height": height})
        label = original / split / "labels" / f"{image.stem}.txt"
        paths[iid] = label.relative_to(original).as_posix()
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = map(float, line.split())
            if int(c) not in (4, 10):
                continue
            box = [(cx-w/2)*width, (cy-h/2)*height, w*width, h*height]
            annotations.append({"id": len(annotations)+1, "image_id": iid,
                                "category_id": 1 if int(c) == 4 else 2,
                                "bbox": box, "area": box[2]*box[3], "iscrowd": 0})
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i+1, "name": c} for i, c in enumerate(CLASSES)]}, paths


def prepare_chess(root: Path, output: Path, *, download=True):
    root = root.resolve()
    original = download_chess(root) if download else root / "original"
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    train, paths = _project_split(original, "train")
    order = sorted(paths, key=lambda i: hashlib.sha256(
        ("coverage-chess-v1\0"+"20260917\0"+paths[i]).encode()).hexdigest())
    sources = []
    visible = 0
    for group, keep_class in ((0, 1), (1, 2)):
        ids = set(order[group::2])
        partial = {**train, "images": [im for im in train["images"] if im["id"] in ids],
                   "annotations": [a for a in train["annotations"] if a["image_id"] in ids and
                                   a["category_id"] == keep_class]}
        name = "source-black" if group == 0 else "source-white"
        write_json(prepared / f"{name}.json", partial)
        visible += len(partial["annotations"])
        sources.append(_source(name, f"prepared/{name}.json", "train",
                       {c: "exhaustive" if i+1 == keep_class else "unknown" for i, c in enumerate(CLASSES)}))
    references = []
    for split in ("valid", "test"):
        data, _ = _project_split(original, split)
        write_json(prepared / f"{split}.json", data)
        references.append(_source(split, f"prepared/{split}.json", split, {c: "exhaustive" for c in CLASSES}))
    write_json(prepared / "complete-train.json", train)
    write_json(root / "partial.json", {"schema_version": 1, "classes": CLASSES, "sources": sources+references})
    write_json(root / "complete.json", {"schema_version": 1, "classes": CLASSES,
        "sources": [_source("complete-train", "prepared/complete-train.json", "train",
                            {c: "exhaustive" for c in CLASSES})]+references})
    if len(train["images"]) != 202 or visible != 480 or len(train["annotations"])-visible != 490:
        raise ValueError("experiment does not match the frozen withholding inventory")
    partial_bundle = compile_bundle(root / "partial.json", output / "bundles")
    complete_bundle = compile_bundle(root / "complete.json", output / "bundles")
    partial_view = materialize_view(partial_bundle, output / "views")
    complete_view = materialize_view(complete_bundle, output / "views")
    experiment = {"protocol": "coverage-chess-v1", "seed": 20260917, "classes": CLASSES,
                  "attribution": ATTRIBUTION, "visible_boxes": visible, "withheld_boxes": 490,
                  "partial_bundle": str(partial_bundle.resolve()), "complete_bundle": str(complete_bundle.resolve()),
                  "partial_view": str(partial_view.resolve()), "complete_view": str(complete_view.resolve()),
                  "scene_independence": "not yet audited", "accuracy_status": "not run"}
    write_json(output / "chess_experiment.json", experiment)
    make_contact_sheets(original, output / "audit")
    return experiment


def _source(name, annotations, split, coverage):
    return {"id": name, "revision": REVISION, "annotations": annotations,
            "images": f"original/{split}/images", "split": split,
            "class_map": {c: c for c in CLASSES}, "coverage": coverage,
            "evidence": "Deterministic class withholding from pinned reference annotations; visual completeness unverified",
            "attribution": ATTRIBUTION}


def make_contact_sheets(original, output):
    output.mkdir(parents=True, exist_ok=True)
    indexed = []
    for split in ("train", "valid", "test"):
        paths = sorted((original / split / "images").glob("*"))
        for page in range((len(paths)+47)//48):
            chunk = paths[page*48:(page+1)*48]
            sheet = Image.new("RGB", (8*144, 6*166), "#f1f3f5")
            draw = ImageDraw.Draw(sheet)
            for j, p in enumerate(chunk):
                with Image.open(p) as im:
                    sheet.paste(ImageOps.contain(im.convert("RGB"), (140, 140)), ((j%8)*144, (j//8)*166))
                    gray = im.convert("L").resize((9, 8))
                    pixels = list(gray.get_flattened_data())
                    bits = sum(int(pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x)
                               for y in range(8) for x in range(8))
                    indexed.append({"split": split, "path": str(p), "dhash": f"{bits:016x}",
                                    "sha256": file_digest(p)})
                draw.text(((j%8)*144+3, (j//8)*166+143), f"{split} {page*48+j}", fill="#202530")
            sheet.save(output / f"{split}-{page}.jpg", quality=90)
    pairs = []
    for i, a in enumerate(indexed):
        for b in indexed[i+1:]:
            if a["split"] == b["split"]:
                continue
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= 12:
                pairs.append({"distance": distance, "a": a["path"], "b": b["path"]})
    write_json(output / "image_index.json", indexed)
    write_json(output / "near_duplicates.json", sorted(pairs, key=lambda p: (p["distance"], p["a"], p["b"])))
