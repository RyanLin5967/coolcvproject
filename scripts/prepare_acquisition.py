"""Freeze both review policies before revealing any published training annotations."""
import argparse
import math
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from coveragecv.artifacts import digest, file_digest, read_json, verify, write_json
from coveragecv.training.acquisition import apply_reviews, plan_reviews

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/acquisition"


def freeze_plans():
    experiment = read_json(ROOT / "artifacts/construction/experiment.json")
    view = Path(experiment["partial_view"])
    raw_path = OUTPUT / "training_predictions.json"
    raw = read_json(raw_path)
    manifest = verify(view)
    checkpoint = ROOT / "artifacts/gpu/construction/20260917/aware/detector.pt"
    if (raw["split"] != "train" or raw["view_digest"] != manifest["digest"]
            or raw["checkpoint_sha256"] != file_digest(checkpoint)):
        raise ValueError("acquisition predictions do not match the observed training contract")
    data = read_json(view / "train/_annotations.coco.json")
    images = {im["id"]: im for im in data["images"]}
    cleaned, clipped, degenerate = [], 0, 0
    for p in raw["predictions"]:
        im = images[p["image_id"]]
        x, y, w, h = p["bbox"]
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w < 0 or h < 0:
            raise ValueError("invalid detector geometry")
        x1, y1 = min(im["width"], max(0., x)), min(im["height"], max(0., y))
        x2, y2 = min(im["width"], max(0., x+w)), min(im["height"], max(0., y+h))
        if x2 <= x1 or y2 <= y1:
            degenerate += 1
            continue
        box = [x1, y1, x2-x1, y2-y1]
        clipped += box != p["bbox"]
        cleaned.append({"image_id": p["image_id"], "category_id": p["category_id"],
                        "bbox": box, "score": p["score"]})
    preprocessing = {"raw_prediction_file_sha256": file_digest(raw_path), "raw_predictions": len(raw["predictions"]),
        "retained_predictions": len(cleaned), "clipped_predictions": clipped, "dropped_zero_area": degenerate,
        "rule": "Clip xyxy to original image bounds; discard zero-area after clipping; no reference labels used"}
    plans = {}
    for mode in ("guided", "random"):
        plan = plan_reviews(view, cleaned, mode=mode, per_class=30, seed=20260917,
                            checkpoint_sha256=raw["checkpoint_sha256"])
        path = OUTPUT / f"{mode}-plan.json"
        if path.exists() and read_json(path) != plan:
            raise ValueError("existing query plan differs; do not overwrite frozen selection")
        if not path.exists():
            write_json(path, plan)
        plans[mode] = plan["digest"]
    frozen = {"plans": plans, "partial_view_digest": manifest["digest"], "preprocessing": preprocessing,
        "selection": "30 image-class reviews per class; guided max model confidence versus uniform seeded random",
        "reference_labels_read_for_selection": False, "test_labels_used": False,
        "study": "Simulated acquisition from published TRAIN annotations, not actual human work",
        "budget": "150 image-class review units per arm; acquired box counts can differ"}
    path = OUTPUT / "selection_frozen.json"
    if path.exists():
        if read_json(path)["declaration"] != frozen:
            raise ValueError("frozen acquisition declaration changed")
    else:
        write_json(path, {"frozen_at": datetime.now(UTC).isoformat(), "declaration": frozen,
                          "declaration_digest": digest(frozen)})
    print({"status": "both_plans_frozen", "plans": plans, "preprocessing": preprocessing}, flush=True)


def reveal_and_pack():
    frozen_path = OUTPUT / "selection_frozen.json"
    frozen = read_json(frozen_path)
    declaration = frozen["declaration"]
    if digest(declaration) != frozen["declaration_digest"]:
        raise ValueError("selection declaration digest mismatch")
    # All query lists are checked before the oracle function receives its reference argument.
    plans = {mode: read_json(OUTPUT / f"{mode}-plan.json") for mode in ("guided", "random")}
    if any(plan["digest"] != declaration["plans"][mode]
           or digest({k: v for k, v in plan.items() if k != "digest"}) != plan["digest"]
           for mode, plan in plans.items()):
        raise ValueError("a query list changed after freezing")
    experiment = read_json(ROOT / "artifacts/construction/experiment.json")
    partial, complete = Path(experiment["partial_view"]), Path(experiment["complete_view"])
    views = {mode: apply_reviews(partial, complete, plan, OUTPUT / "views") for mode, plan in plans.items()}
    views["complete_standard"] = complete
    archive_path = OUTPUT / "input.zip"
    if archive_path.exists():
        raise ValueError("cloud input archive already exists; do not overwrite a frozen experiment")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for case, view in views.items():
            manifest = verify(view)
            for name in ("manifest.json", *sorted(manifest["files"])):
                archive.write(view / name, f"{case}/{name}")
    parents = {}
    for arm in ("aware", "complete_reference"):
        folder = ROOT / "artifacts/gpu/construction/20260917" / arm
        parents[arm] = {"checkpoint_sha256": file_digest(folder / "detector.pt"),
                        "call_id": read_json(folder / "call.json")["call_id"]}
    protocol = {"kind": "fixed_budget_coverage_acquisition_simulation", "seed": 20260917,
        "steps": 2000, "batch": 4, "resolution": 512, "recipe": "augmented", "total_training_steps": 6000,
        "optimizer": "restarted AdamW, cosine, LR5e-5 encoder5e-6", "validation_during_training": False,
        "selection_frozen_file_sha256": file_digest(frozen_path), "dataset_sha256": file_digest(archive_path),
        "parents": parents, "reference_bundle_digest": verify(Path(experiment["complete_bundle"]))["digest"],
        "views": {case: {"path": str(view), "digest": verify(view)["digest"],
                         "acquisition": read_json(view / "acquisition-audit.json") if case != "complete_standard" else None}
                  for case, view in views.items()},
        "primary_comparison": ["guided", "random"], "review_units_per_acquisition_arm": 150,
        "zero_review_control": {"folder": "artifacts/object-crops/construction/20260917/aware_standard",
            "checkpoint_sha256": file_digest(ROOT / "artifacts/object-crops/construction/20260917/aware_standard/detector.pt")},
        "limitations": ["Additional published TRAIN annotations are revealed; not the original fixed-label-budget experiment.",
            "Equal image-class reviews can acquire different box counts; report both.",
            "Published annotation oracle can contain errors; no new human adjudication is claimed.",
            "One fixed seed, validation-guided design, unchanged published validation labels; test remains unused."],
        "cloud_reservation": {"calls": 3, "each_usd": 1.10, "max_usd": 3.30, "timeout_seconds": 1500,
                               "max_containers": 3, "retries": 0, "volumes": []}}
    write_json(OUTPUT / "protocol.json", protocol)
    print({"status": "published", "views": {k: str(v) for k,v in views.items()},
           "archive_sha256": protocol["dataset_sha256"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "apply"))
    args = parser.parse_args()
    freeze_plans() if args.stage == "plan" else reveal_and_pack()
