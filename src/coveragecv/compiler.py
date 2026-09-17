"""Compile COCO observations and explicit source policy into an immutable contract."""
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from . import __version__
from .artifacts import (
    canonical,
    digest,
    file_digest,
    publish,
    read_json,
    safe_child,
    stage,
    verify,
    write_json,
    write_jsonl,
)
from .schema import NEGATIVE_ALLOWED, CompileSpec, DiagnosticError

SPLITS = ("train", "valid", "test")


def _fail(code, message, source, **context):
    raise DiagnosticError(code, message, source=source.id, **context)


def _integer(value):
    return type(value) is int


def _read_source(source, classes, base):
    data = read_json((base / source.annotations).resolve())
    image_root = (base / source.images).resolve()
    categories = {}
    for cat in data["categories"]:
        if not _integer(cat["id"]) or cat["id"] in categories:
            _fail("INVALID_CATEGORY", "category IDs must be unique integers", source)
        if cat["name"] not in source.class_map:
            _fail("ONTOLOGY_MAPPING_REQUIRED", "source category has no explicit mapping", source,
                  category=cat["name"])
        categories[cat["id"]] = classes.index(source.class_map[cat["name"]])
    images = {}
    filenames = set()
    for image in data["images"]:
        iid = image["id"]
        if not _integer(iid) or iid in images or image["file_name"] in filenames:
            _fail("INVALID_IMAGE_ID", "image IDs and filenames must be unique", source)
        p = safe_child(image_root, image["file_name"])
        with Image.open(p) as decoded:
            decoded.load()
            width, height = decoded.size
            if decoded.getexif().get(274, 1) != 1:
                _fail("UNSUPPORTED_ORIENTATION", "normalize image orientation and boxes before import", source)
        if not _integer(image["width"]) or not _integer(image["height"]) or (width, height) != (
            image["width"], image["height"]
        ):
            _fail("IMAGE_DIMENSION_MISMATCH", "declared dimensions differ from decoded image", source,
                  image=image["file_name"])
        states = {name: source.coverage.get(name, "unknown") for name in classes}
        states.update(source.overrides.get(image["file_name"], {}))
        content_sha = file_digest(p)
        images[iid] = {"path": p, "file_name": image["file_name"], "width": width, "height": height,
                       "content_sha256": content_sha, "split": source.split,
                       "coverage": [states[c] for c in classes], "boxes": [],
                       "observations": [{"source": source.id, "revision": source.revision,
                                         "original_file": image["file_name"], "evidence": source.evidence,
                                         "attribution": source.attribution}]}
        filenames.add(image["file_name"])
    unknown_overrides = set(source.overrides) - filenames
    if unknown_overrides:
        _fail("INVALID_COVERAGE_OVERRIDE", "coverage override names an absent image", source,
              images=sorted(unknown_overrides))
    annotation_ids = set()
    for ann in data["annotations"]:
        if not _integer(ann["id"]) or ann["id"] in annotation_ids:
            _fail("INVALID_ANNOTATION", "annotation IDs must be unique integers", source)
        annotation_ids.add(ann["id"])
        if (not _integer(ann["image_id"]) or not _integer(ann["category_id"]) or
                ann["image_id"] not in images or ann["category_id"] not in categories):
            _fail("INVALID_ANNOTATION", "annotation refers to an absent image/category", source)
        if ann.get("iscrowd", 0) or ann.get("ignore", 0) or ann.get("segmentation"):
            _fail("UNSUPPORTED_ANNOTATION", "crowd, ignore regions and segmentation are unsupported", source)
        box = ann["bbox"]
        if (not isinstance(box, list) or len(box) != 4 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in box
        )):
            _fail("INVALID_BOX", "bbox must contain four finite numbers", source)
        x, y, w, h = map(float, box)
        image = images[ann["image_id"]]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image["width"]+1e-5 or y+h > image["height"]+1e-5:
            _fail("INVALID_BOX", "box is empty or outside the image", source, annotation=ann["id"])
        label = categories[ann["category_id"]]
        if image["coverage"][label] == "verified_absent":
            _fail("COVERAGE_ABSENCE_CONTRADICTION", "positive box contradicts verified absence", source,
                  image=image["file_name"], category=classes[label])
        image["boxes"].append({"category_id": label+1, "bbox": [x, y, w, h]})
    for image in images.values():
        image["boxes"].sort(key=canonical)
        if len({canonical(b) for b in image["boxes"]}) != len(image["boxes"]):
            _fail("DUPLICATE_ANNOTATION", "identical box observations require explicit source correction", source)
    return list(images.values())


def compile_bundle(spec_path: Path, output: Path) -> Path:
    spec = CompileSpec.model_validate_json(spec_path.read_text())
    samples = {}
    coalesced = 0
    for source in sorted(spec.sources, key=lambda s: s.id):
        for image in _read_source(source, spec.classes, spec_path.parent):
            sha = image["content_sha256"]
            if sha in samples:
                other = samples[sha]
                if image["split"] != other["split"]:
                    _fail("IMAGE_SPLIT_COLLISION", "identical image bytes occur in different splits", source,
                          image=image["file_name"])
                if image["boxes"] != other["boxes"] or image["coverage"] != other["coverage"]:
                    _fail("DUPLICATE_OBSERVATION_CONFLICT", "identical image has conflicting labels/coverage", source)
                other["observations"].extend(image["observations"])
                coalesced += 1
            else:
                samples[sha] = image
    ordered = sorted(samples.values(), key=lambda x: (SPLITS.index(x["split"]), x["content_sha256"]))
    categories = [{"id": i+1, "name": c, "supercategory": "none"} for i, c in enumerate(spec.classes)]
    ontology = {"classes": spec.classes, "categories": categories, "reserved_model_index": len(spec.classes)}
    ontology_sha = digest(ontology)
    coco = {s: {"images": [], "annotations": [], "categories": categories} for s in SPLITS}
    coverage, provenance = [], []
    annotation_id = 1
    staging = stage(output)
    try:
        for iid, image in enumerate(ordered):
            image["observations"].sort(key=canonical)
            sample_key = digest({"content": image["content_sha256"], "observations": image["observations"]})
            name = f"images/{image['content_sha256']}{image['path'].suffix.lower()}"
            dest = staging / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image["path"], dest)
            split = image["split"]
            coco[split]["images"].append({"id": iid, "file_name": name, "width": image["width"],
                                          "height": image["height"]})
            for box in image["boxes"]:
                coco[split]["annotations"].append({"id": annotation_id, "image_id": iid, **box,
                    "area": box["bbox"][2]*box["bbox"][3], "iscrowd": 0})
                annotation_id += 1
            coverage.append({"image_id": iid, "sample_key": sample_key, "split": split,
                             "states": image["coverage"]})
            provenance.append({"image_id": iid, "content_sha256": image["content_sha256"],
                               "sample_key": sample_key, "observations": image["observations"]})
        diagnostics = {
            "images": len(ordered), "annotations": annotation_id-1, "coalesced_observations": coalesced,
            "splits": {s: {"images": len(coco[s]["images"]), "annotations": len(coco[s]["annotations"])}
                       for s in SPLITS},
            "coverage_counts": {c: dict(Counter(r["states"][i] for r in coverage))
                                for i, c in enumerate(spec.classes)},
            "no_task_supervision": [iid for iid, im in enumerate(ordered) if not im["boxes"] and
                                    not any(s in NEGATIVE_ALLOWED for s in im["coverage"])],
        }
        write_json(staging / "ontology.json", ontology)
        for split in SPLITS:
            write_json(staging / f"splits/{split}.coco.json", coco[split])
        write_jsonl(staging / "coverage.jsonl", coverage)
        write_jsonl(staging / "provenance.jsonl", provenance)
        write_json(staging / "diagnostics.json", diagnostics)
        return publish(staging, output, {"schema_version": 1, "kind": "coverage_bundle",
                       "compiler_version": __version__, "ontology_digest": ontology_sha,
                       "image_count": len(ordered)})
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_view(bundle: Path, output: Path) -> Path:
    manifest = verify(bundle)
    if manifest["kind"] != "coverage_bundle":
        raise DiagnosticError("INVALID_ARTIFACT_KIND", "training view requires a source bundle")
    staging = stage(output)
    try:
        selected = set()
        for split in ("train", "valid"):
            data = read_json(bundle / f"splits/{split}.coco.json")
            for image in data["images"]:
                selected.add(image["id"])
                src = safe_child(bundle, image["file_name"])
                filename = src.name
                dst = staging / split / filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                image["file_name"] = filename
            write_json(staging / split / "_annotations.coco.json", data)
        rows = [row for row in _jsonl(bundle / "coverage.jsonl") if row["image_id"] in selected]
        write_jsonl(staging / "coverage.jsonl", rows)
        shutil.copyfile(bundle / "ontology.json", staging / "ontology.json")
        return publish(staging, output, {"schema_version": 1, "kind": "training_view",
                       "parent_digest": manifest["digest"], "ontology_digest": manifest["ontology_digest"],
                       "index_space_size": manifest["image_count"], "compiler_version": __version__})
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _jsonl(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines() if line]
