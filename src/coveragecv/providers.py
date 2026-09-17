"""Provider access with credentials outside the repository and redacted failures."""
import contextlib
import io
import json
import os
from pathlib import Path

from .artifacts import write_json


def credentials():
    path = Path.home() / ".config/coveragecv/credentials.json"
    values = json.loads(path.read_text()) if path.exists() else {}
    for key in ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        if os.getenv(key):
            values[key] = os.environ[key]
    return values


def redact(value):
    secrets = [v for k, v in credentials().items() if any(s in k for s in ("KEY", "TOKEN")) and v]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if not any(s in k.lower() for s in ("api_key", "token_secret", "signed_url"))}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
    return value


@contextlib.contextmanager
def safe_sdk():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            yield
        except Exception as exc:  # noqa: BLE001 -- redact arbitrary SDK errors before exposing them
            raise RuntimeError(redact(str(exc))) from None


def roboflow_workspace():
    from roboflow import Roboflow
    c = credentials()
    with safe_sdk():
        return Roboflow(api_key=c["ROBOFLOW_API_KEY"]).workspace(c["ROBOFLOW_WORKSPACE"])


def roboflow_preflight(output=Path("artifacts/provider")):
    workspace = roboflow_workspace()
    with safe_sdk():
        plan = workspace.get_plan()
    try:
        with safe_sdk():
            usage = workspace.get_usage()
    except RuntimeError as exc:
        usage = {"available": False, "reason": str(exc)}
    result = redact({"workspace": workspace.url, "plan": plan, "usage": usage})
    write_json(output / "roboflow_preflight.json", result)
    return result


def upload_demo_view(view: Path, output=Path("artifacts/provider"), project_name="coveragecv-chess-mvp"):
    """Upload exact train/validation observations, with resumable per-image receipts."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .artifacts import read_json, verify
    manifest = verify(view)
    workspace = roboflow_workspace()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "roboflow_upload.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {
        "view_digest": manifest["digest"], "project_name": project_name, "images": {}}
    if ledger["view_digest"] != manifest["digest"]:
        raise ValueError("upload ledger belongs to a different immutable view")
    # Read the project listing first so network failures are not mistaken for absence.
    projects = workspace.project_list
    exists = any(p["id"].rsplit("/", 1)[-1] == project_name for p in projects)
    with safe_sdk():
        project = (workspace.project(project_name) if exists else
                   workspace.create_project(project_name, "object-detection", "CC BY 4.0", "pawns"))
    ledger["project_id"] = project.id
    write_json(ledger_path, ledger)
    tasks = []
    for split in ("train", "valid"):
        data = read_json(view / split / "_annotations.coco.json")
        for image in data["images"]:
            key = f"{split}/{image['id']}"
            if ledger["images"].get(key, {}).get("annotation_format") != "voc-one-based-v1":
                # The per-image endpoint accepts VOC; COCO is a dataset-level import format.
                annotation = voc_annotation(image, [a for a in data["annotations"]
                                                    if a["image_id"] == image["id"]], data["categories"])
                tasks.append((key, split, image, annotation))
    def upload(task):
        key, split, image, annotation = task
        try:
            if key in ledger["images"]:
                image_id = ledger["images"][key]["image_id"]
                saved, _, _ = project.save_annotation(annotation_path=annotation, image_id=image_id,
                                                       annotation_overwrite=True)
                result = {"image": {"success": True, "id": image_id}, "annotation": saved}
            else:
                result = project.single_upload(str(view / split / image["file_name"]), annotation_path=annotation,
                    split=split, num_retry_uploads=0, annotation_overwrite=True, batch_name="coveragecv-group-v1",
                    metadata={"coveragecv_view": manifest["digest"], "coveragecv_image_id": image["id"]})
            image_result = result["image"]
            if (not (image_result.get("success") or image_result.get("duplicate")) or
                    not image_result.get("id") or not result.get("annotation", {}).get("success")):
                raise RuntimeError("image or annotation upload did not return success")
            return key, {"image_id": result["image"]["id"], "split": split,
                         "filename": image["file_name"], "status": "uploaded",
                         "annotation_format": "voc-one-based-v1"}
        except Exception as exc:  # noqa: BLE001 -- redact arbitrary SDK errors before exposing them
            raise RuntimeError(redact(str(exc))) from None
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(upload, task) for task in tasks]
        for future in as_completed(futures):
            key, result = future.result()
            ledger["images"][key] = result
            write_json(ledger_path, ledger)
            if len(ledger["images"]) % 25 == 0:
                print(f"Roboflow uploaded {len(ledger['images'])} images", flush=True)
    if "version" not in ledger or ledger.get("version_annotation_format") != "voc-one-based-v1":
        if "version" in ledger:
            ledger.setdefault("superseded_versions", []).append(ledger["version"])
        with safe_sdk():
            ledger["version"] = project.generate_version({"preprocessing": {}, "augmentation": {}})
        ledger["version_annotation_format"] = "voc-one-based-v1"
        write_json(ledger_path, ledger)
    ledger["url"] = f"https://app.roboflow.com/{workspace.url}/{project_name}/{ledger['version']}"
    write_json(ledger_path, ledger)
    return ledger


def voc_annotation(image, annotations, categories):
    from xml.etree.ElementTree import Element, SubElement, tostring
    root = Element("annotation")
    SubElement(root, "filename").text = image["file_name"]
    size = SubElement(root, "size")
    for key, value in (("width", image["width"]), ("height", image["height"]), ("depth", 3)):
        SubElement(size, key).text = str(value)
    names = {c["id"]: c["name"] for c in categories}
    for ann in annotations:
        obj = SubElement(root, "object")
        SubElement(obj, "name").text = names[ann["category_id"]]
        bounds = SubElement(obj, "bndbox")
        x, y, width, height = ann["bbox"]
        # Roboflow's VOC parser subtracts one from the origin; COCO is zero-based.
        x, y = x + 1, y + 1
        for key, value in (("xmin", x), ("ymin", y), ("xmax", x+width), ("ymax", y+height)):
            SubElement(bounds, key).text = str(value)
    return {"name": "annotation.xml", "rawText": tostring(root, encoding="unicode")}


def verify_roboflow_export(view: Path, exported: Path, output: Path):
    """Check platform round-trip identity, splits, classes and subpixel geometry."""
    from .artifacts import file_digest, read_json, verify
    manifest = verify(view)
    classes = read_json(view / "ontology.json")["classes"]
    counts, max_error = {}, 0.0
    for split in ("train", "valid"):
        local = read_json(view / split / "_annotations.coco.json")
        exported_file = exported / split / "_annotations.coco.json"
        remote = read_json(exported_file)
        expected_images = {im["id"]: im for im in local["images"]}
        names = {c["id"]: c["name"] for c in remote["categories"]}
        seen = set()
        for image in remote["images"]:
            metadata = image["extra"]["user_metadata"]
            iid = metadata["coveragecv_image_id"]
            if iid in seen or iid not in expected_images or metadata["coveragecv_view"] != manifest["digest"]:
                raise ValueError("platform image identity/split differs from the compiled view")
            seen.add(iid)
            expected = expected_images[iid]
            if (image["width"], image["height"]) != (expected["width"], expected["height"]):
                raise ValueError("platform changed image dimensions")
            remaining = [(a["category_id"], a["bbox"]) for a in local["annotations"] if a["image_id"] == iid]
            for annotation in (a for a in remote["annotations"] if a["image_id"] == image["id"]):
                category = classes.index(names[annotation["category_id"]]) + 1
                candidates = [(max(abs(x-y) for x, y in zip(box, annotation["bbox"])), j)
                              for j, (label, box) in enumerate(remaining) if label == category]
                error, index = min(candidates, default=(float("inf"), -1))
                if error > 1e-4:
                    raise ValueError(f"platform changed box geometry or class for image {iid}: {error} pixels")
                max_error = max(max_error, error)
                remaining.pop(index)
            if remaining:
                raise ValueError("platform dropped annotations")
        if seen != set(expected_images):
            raise ValueError("platform dropped images")
        counts[split] = {"images": len(seen), "boxes": len(remote["annotations"]),
                         "export_annotations_sha256": file_digest(exported_file)}
    result = {"status": "passed", "view_digest": manifest["digest"], "splits": counts,
              "max_coordinate_error_px": max_error, "tolerance_px": 1e-4,
              "method": "Downloaded COCO export; bijective image/box mapping through preserved metadata"}
    write_json(output, result)
    return result
