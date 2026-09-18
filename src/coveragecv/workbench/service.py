"""Application operations: imports, revision analysis and reproducible comparisons."""
import copy
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path

from coveragecv.artifacts import read_json, safe_child, verify
from coveragecv.compiler import _jsonl, materialize_view
from coveragecv.schema import NEGATIVE_ALLOWED, CompileSpec


def import_spec(store, spec_path: Path, name: str, *, reference=None, import_root=None):
    spec_path = spec_path.resolve()
    spec = CompileSpec.model_validate_json(spec_path.read_text()).model_dump(mode="json")
    if import_root is not None:
        for source in spec["sources"]:
            for key in ("annotations", "images"):
                target = (spec_path.parent / source[key]).resolve()
                if not target.is_relative_to(import_root.resolve()):
                    raise ValueError("Dataset sources must stay inside the uploaded archive")
    return store.create_project(name.strip(), spec, spec_path.parent, str(reference) if reference else None)


def extract_dataset(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as z:
            entries = z.infolist()
            if len(entries) > 20000 or sum(e.file_size for e in entries) > 500*1024*1024:
                raise ValueError("The archive exceeds 20,000 entries or 500 MB unpacked")
            seen = set()
            for entry in entries:
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Symlinks are not accepted in dataset archives")
                path = safe_child(destination, entry.filename.rstrip("/"))
                if path in seen:
                    raise ValueError("Archive contains duplicate paths")
                seen.add(path)
                if entry.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target)
        specs = list(destination.rglob("coverage.json"))
        if len(specs) != 1:
            raise ValueError("Include exactly one coverage.json spec in the dataset ZIP")
        return specs[0]
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def revision_summary(revision):
    if revision["status"] != "ready":
        return {**revision, "classes": revision["spec"]["classes"]}
    bundle = Path(revision["bundle"])
    manifest = verify(bundle)
    coverage = _jsonl(bundle / "coverage.jsonl")
    provenance = {r["image_id"]: r for r in _jsonl(bundle / "provenance.jsonl")}
    classes = read_json(bundle / "ontology.json")["classes"]
    samples = []
    class_stats = {name: {"boxes": 0, "validation_boxes": 0, "test_boxes": 0,
                          "known_images": 0, "unknown_images": 0} for name in classes}
    by_id = {r["image_id"]: r for r in coverage}
    for split in ("train", "valid", "test"):
        data = read_json(bundle / f"splits/{split}.coco.json")
        counts = Counter(a["image_id"] for a in data["annotations"])
        key = {"train": "boxes", "valid": "validation_boxes", "test": "test_boxes"}[split]
        for ann in data["annotations"]:
            class_stats[classes[ann["category_id"]-1]][key] += 1
        for im in data["images"]:
            row = by_id[im["id"]]
            sources = sorted({o["source"] for o in provenance[im["id"]]["observations"]})
            samples.append({"id": im["id"], "split": split, "states": row["states"], "boxes": counts[im["id"]],
                            "sources": sources, "sha": provenance[im["id"]]["content_sha256"]})
            if split == "train":
                for name, state in zip(classes, row["states"]):
                    class_stats[name]["known_images" if state in NEGATIVE_ALLOWED else "unknown_images"] += 1
    return {**revision, "manifest": manifest, "classes": classes, "samples": samples, "class_stats": class_stats}


def change_policy(store, project_id, base_revision, changes):
    previous = store.get("revisions", base_revision)
    if not previous or previous["project_id"] != project_id:
        raise ValueError("Base revision does not belong to this project")
    spec = copy.deepcopy(previous["spec"])
    sources = {s["id"]: s for s in spec["sources"]}
    for change in changes:
        if change["source"] not in sources or change["class_name"] not in spec["classes"]:
            raise ValueError("Policy change refers to an unknown source or class")
        sources[change["source"]]["coverage"][change["class_name"]] = change["state"]
    CompileSpec.model_validate(spec)
    return store.create_revision(project_id, spec, previous["base_dir"], expected_revision=base_revision)


def revision_diff(before, after):
    a, b = revision_summary(before), revision_summary(after)
    if "samples" not in a or "samples" not in b:
        raise ValueError("Both revisions must compile successfully before comparing")
    if a["classes"] != b["classes"]:
        raise ValueError("Ontology changes require an explicit class migration before coverage comparison")
    ai, bi = ({s["sha"]: s for s in row["samples"]} for row in (a, b))
    changes, changed_images = [], set()
    before_negative = after_negative = 0
    for sha in ai.keys() & bi.keys():
        old, new = ai[sha], bi[sha]
        for name, x, y in zip(a["classes"], old["states"], new["states"]):
            if old["split"] == "train":
                before_negative += x in NEGATIVE_ALLOWED
            if new["split"] == "train":
                after_negative += y in NEGATIVE_ALLOWED
            if x != y:
                changed_images.add(sha)
                changes.append({"image_id": new["id"], "class_name": name, "before": x, "after": y,
                                "split": new["split"]})
    return {"before_digest": a["manifest"]["digest"], "after_digest": b["manifest"]["digest"],
            "added_images": len(bi.keys()-ai.keys()), "removed_images": len(ai.keys()-bi.keys()),
            "changed_images": len(changed_images), "changed_class_cells": len(changes),
            "negative_training_cells_before": before_negative, "negative_training_cells_after": after_negative,
            "changes": changes}


def queue_training(store, revision, parameters):
    if revision["status"] != "ready":
        raise ValueError("Compile this revision successfully before training")
    view, bundle = Path(revision["view"]), Path(revision["bundle"])
    manifest = verify(view)
    train = read_json(view / "train/_annotations.coco.json")
    valid = read_json(view / "valid/_annotations.coco.json")
    if len(train["images"]) < parameters["batch"] or not train["annotations"] or not valid["images"]:
        raise ValueError("Training needs enough images for one batch, observed boxes and validation images")
    if any(state not in NEGATIVE_ALLOWED for row in _jsonl(view / "coverage.jsonl")
           if row["split"] == "valid" for state in row["states"]):
        raise ValueError("Validation must declare complete coverage for every class")
    payload = {**parameters, "view": str(view), "view_digest": manifest["digest"], "reference": str(bundle)}
    if "complete_reference" in parameters["arms"]:
        project = store.get("projects", revision["project_id"])
        if not project["reference_bundle"]:
            raise ValueError("This project has no independent complete-label training reference")
        reference = Path(project["reference_bundle"])
        rm = verify(reference)
        if rm["ontology_digest"] != manifest["ontology_digest"]:
            raise ValueError("Complete reference uses a different ontology")
        for split in ("train", "valid"):
            a = read_json(bundle / f"splits/{split}.coco.json")["images"]
            b = read_json(reference / f"splits/{split}.coco.json")["images"]
            if a != b:
                raise ValueError("Complete reference must have exactly the same images and splits")
        payload.update(reference=str(reference), complete_view=str(materialize_view(reference, store.root / "views")))
    return store.create_job(revision["project_id"], revision["id"], "train", payload)


def import_saved_experiment(store, project_id, revision_id, folder, reference_bundle):
    """Adopt real existing measurements into history without claiming a new execution."""
    from coveragecv.artifacts import file_digest
    from coveragecv.report import validate_run_evidence
    arms = {}
    shared = None
    revision = store.get("revisions", revision_id)
    for arm in ("naive", "aware", "complete_reference"):
        path = folder / arm
        if not (path / "evaluation.json").exists():
            continue
        run, evaluation = read_json(path / "run.json"), read_json(path / "evaluation.json")
        expected = verify(Path(revision["view"]))["digest"]
        if arm == "complete_reference":
            expected = verify(materialize_view(reference_bundle, store.root / "views"))["digest"]
        validate_run_evidence(arm, run, evaluation, view_digest=expected,
                              bundle_digest=verify(reference_bundle)["digest"],
                              checkpoint_digest=file_digest(path / "detector.pt"))
        comparison = {key: run[key] for key in ("seed", "steps", "batch", "epochs", "device", "initial_parameter_digest")}
        if shared is not None and comparison != shared:
            raise ValueError("comparison arms differ in initialization or execution budget")
        shared = comparison
        arms[arm] = {"run": run, "metrics": evaluation["metrics"], "evaluation_path": str(path / "evaluation.json"),
                     "checkpoint_path": str(path / "detector.pt")}
    if not arms:
        raise ValueError("No completed verified measurements were found")
    job = store.create_job(project_id, revision_id, "imported", {"source": str(folder), "arms": list(arms)})
    store.finish(job["id"], {"status": "completed", "result": {"arms": arms,
        "reference_bundle": str(reference_bundle), "imported": True}})
    return store.get("jobs", job["id"])
