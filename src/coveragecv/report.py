"""A standalone, inspectable visual report backed only by saved evidence."""
import base64
import json
from pathlib import Path

from .artifacts import file_digest, read_json, safe_child, verify
from .compiler import _jsonl


def build_report(experiment: Path, runs: Path, output: Path):
    ex = read_json(experiment)
    partial, complete = Path(ex["partial_bundle"]), Path(ex["complete_bundle"])
    pm, cm = verify(partial), verify(complete)
    rows = {row["image_id"]: row for row in _jsonl(partial / "coverage.jsonl")}
    provenance = {row["image_id"]: row for row in _jsonl(partial / "provenance.jsonl")}
    observed = read_json(partial / "splits/train.coco.json")
    full = read_json(complete / "splits/train.coco.json")
    reference_by_image = {}
    for a in full["annotations"]:
        reference_by_image.setdefault(a["image_id"], []).append(a)
    samples = []
    # Chosen from reference class support, not from a model's successes/failures.
    images = sorted(full["images"], key=lambda im: (
        -len({a["category_id"] for a in reference_by_image.get(im["id"], [])}), im["id"]))[:16]
    for im in images:
        iid = im["id"]
        samples.append({**im, "source": provenance[iid]["observations"][0]["source"],
                        "coverage": rows[iid]["states"], "image": _image_url(safe_child(complete, im["file_name"])),
                        "observed": [a for a in observed["annotations"] if a["image_id"] == iid],
                        "reference": reference_by_image.get(iid, [])})
    evaluations = {}
    shared = None
    for arm in ("naive", "aware", "complete_reference"):
        folder = runs / arm
        if (folder / "evaluation.json").exists():
            run, evaluation = read_json(folder / "run.json"), read_json(folder / "evaluation.json")
            expected_view = verify(Path(ex["complete_view"] if arm == "complete_reference" else ex["partial_view"]))
            validate_run_evidence(arm, run, evaluation, view_digest=expected_view["digest"],
                                  bundle_digest=cm["digest"], checkpoint_digest=file_digest(folder / "detector.pt"))
            matched = {key: run[key] for key in ("seed", "steps", "batch", "epochs", "max_steps", "device",
                                                "initialization_sha256", "initial_parameter_digest")}
            if shared is not None and matched != shared:
                raise ValueError("comparison runs do not share initialization and training budget")
            shared = matched
            evaluations[arm] = {"run": run, "evaluation": evaluation}
    validation = read_json(complete / "splits/valid.coco.json")
    val_ids_with_boxes = {a["image_id"] for a in validation["annotations"]}
    val_images = sorted(validation["images"], key=lambda im: (im["id"] not in val_ids_with_boxes, im["id"]))[:12]
    validation_samples = [{**im, "image": _image_url(safe_child(complete, im["file_name"])),
                           "reference": [a for a in validation["annotations"] if a["image_id"] == im["id"]]}
                          for im in val_images]
    platform_file = experiment.parent / "provider/roboflow_upload.json"
    platform = read_json(platform_file) if platform_file.exists() else {}
    audit_file = experiment.parent / "audit/scene_groups.json"
    audit = read_json(audit_file) if audit_file.exists() else {}
    roundtrip_file = experiment.parent / "provider/roundtrip_verification.json"
    roundtrip = read_json(roundtrip_file) if roundtrip_file.exists() else {}
    if roundtrip and roundtrip["view_digest"] != verify(Path(ex["partial_view"]))["digest"]:
        raise ValueError("platform verification belongs to a different dataset")
    deployment_file = experiment.parent / "provider/model/server_status.json"
    deployment = read_json(deployment_file) if deployment_file.exists() else {}
    data = {"experiment": ex, "bundle": pm["digest"], "reference_bundle": cm["digest"],
            "diagnostics": read_json(partial / "diagnostics.json"), "samples": samples,
            "validation": validation_samples, "evaluations": evaluations,
            "platform": {k: platform.get(k) for k in ("url", "project_id", "version")},
            "uploaded_images": len(platform.get("images", {})),
            "roundtrip": roundtrip, "deployment": deployment,
            "audit": {k: audit.get(k) for k in ("group_count", "groups_crossing_original_splits", "limitation")}}
    payload = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent / "report_template.html").read_text()
    output.write_text(template.replace("__REPORT_DATA__", payload))
    return output.resolve()


def validate_run_evidence(arm, run, evaluation, *, view_digest, bundle_digest, checkpoint_digest):
    """Reject stale metrics, mislabeled runs and mismatched data before rendering."""
    if run["arm"] != arm or run["status"] != "completed":
        raise ValueError("reported arm does not match a completed training ledger")
    if run["view_digest"] != view_digest:
        raise ValueError("run used a different learner view")
    if run["detector_sha256"] != checkpoint_digest or evaluation["checkpoint_sha256"] != checkpoint_digest:
        raise ValueError("evaluation is not bound to the saved detector")
    if evaluation["bundle_digest"] != bundle_digest or evaluation["split"] != "valid":
        raise ValueError("comparison requires the same complete validation reference")


def _image_url(path):
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()
