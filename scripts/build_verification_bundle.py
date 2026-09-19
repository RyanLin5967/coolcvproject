"""Build the browser-verifiable evidence bundle for the public demo.

Every published score on the Benchmarks page is a stored constant a visitor cannot
check. This script exports the *saved validation predictions* behind selected runs,
plus the reference labels they were scored against, so the site can recompute
AP50:95 in the visitor's browser and show it landing on the published value.

Two reductions keep the payload small. Both are asserted lossless here, against the
published metrics, before anything is written:

  1. Only the four fields pycocotools uses for bbox evaluation are kept
     (image_id, category_id, bbox, score). `COCO.loadRes` overwrites area/id/iscrowd
     and derives segmentation, so the rest cannot affect a score.
  2. Detections are truncated to the top 100 per (image, category), which is the
     `maxDets=100` limit COCOeval already applies, and quantised to float32, which
     is the precision the detector emitted.

A run is only exported if the reduced predictions reproduce its published metrics
dict exactly -- all keys, bit for bit.
"""
import argparse
import hashlib
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coveragecv.training.evaluate import score_predictions

SNAPSHOT = ROOT / "public-demo/data/snapshot.json"
OUTPUT = ROOT / "src/coveragecv/workbench/static/verify"
MAGIC = b"CVB1"

# One matched cohort per entry: same recipe, same update budget, one row per seed.
# `arms` maps the published method name to the role the UI shows it as.
COHORTS = [
    {
        "key": "pawns-base",
        "task": "pawns",
        "name": "Chess pawns",
        "detail": "2 classes · 58 validation images",
        "recipe": "RF-DETR Nano · 384px · 2,000 updates",
        "headline": True,
        "arms": {"naive": "naive", "aware": "aware", "complete_reference": "complete_reference"},
    },
    {
        "key": "pawns-augmented",
        "task": "pawns",
        "name": "Chess pawns · stronger recipe",
        "detail": "2 classes · 58 validation images",
        "recipe": "RF-DETR Nano · 512px · 4,000 updates · augmentation",
        "arms": {"naive_augmented_512": "naive", "aware_augmented_512": "aware",
                 "complete_augmented_512": "complete_reference"},
    },
    {
        "key": "all-pieces-base",
        "task": "all-pieces",
        "name": "All chess pieces",
        "detail": "13 classes · 58 validation images",
        "recipe": "RF-DETR Nano · 384px · 2,000 updates",
        "arms": {"naive": "naive", "aware": "aware", "complete_reference": "complete_reference"},
    },
    {
        "key": "construction",
        "task": "construction",
        "name": "Construction safety",
        "detail": "5 classes · 120 validation images",
        "recipe": "RF-DETR Nano · 512px · 4,000 updates",
        "arms": {"naive": "naive", "aware": "aware", "complete_reference": "complete_reference"},
    },
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def index_evaluations():
    """Map a published AP (exact float64) to the saved evaluation that produced it.

    Matching on the full 17-digit double is an identity check, so the mapping from a
    number on the site to the file behind it is established rather than assumed.
    """
    index = defaultdict(list)
    for path in ROOT.joinpath("artifacts").rglob("evaluation.json"):
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("AP"), float):
            index[metrics["AP"]].append(path)
    return index


def find_ground_truth(digest):
    for candidate in ROOT.joinpath("artifacts").rglob(f"bundles/{digest}/splits/valid.coco.json"):
        return candidate
    return None


def reduce_predictions(predictions):
    """Keep the four scored fields, quantise to float32, cap at maxDets per class."""
    grouped = defaultdict(list)
    for entry in predictions:
        box = np.asarray(entry["bbox"], dtype=np.float32).astype(float).tolist()
        grouped[(entry["image_id"], entry["category_id"])].append(
            {"image_id": entry["image_id"], "category_id": entry["category_id"],
             "bbox": box, "score": float(np.float32(entry["score"]))})
    reduced = []
    for key in sorted(grouped):
        rows = grouped[key]
        rows.sort(key=lambda row: (-row["score"], row["bbox"]))
        reduced.extend(rows[:100])
    return reduced


def encode(predictions):
    count = len(predictions)
    image_ids = np.array([p["image_id"] for p in predictions], dtype="<i4")
    categories = np.array([p["category_id"] for p in predictions], dtype="<i4")
    scores = np.array([p["score"] for p in predictions], dtype="<f4")
    boxes = np.array([p["bbox"] for p in predictions], dtype="<f4").reshape(-1)
    header = MAGIC + struct.pack("<III", 1, count, 0)
    return header + image_ids.tobytes() + categories.tobytes() + scores.tobytes() + boxes.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Verify the committed bundle still reproduces every number; write nothing.")
    args = parser.parse_args()

    snapshot = json.loads(SNAPSHOT.read_text())
    tasks = snapshot["routes"]["/research"]["tasks"]
    index = index_evaluations()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    ground_truths, cohorts, failures, written = {}, [], [], []
    for spec in COHORTS:
        task = tasks.get(spec["task"])
        if not task:
            failures.append(f"{spec['key']}: task absent from snapshot")
            continue
        by_method = defaultdict(list)
        for run in task.get("runs", []):
            if run.get("method") in spec["arms"]:
                by_method[run["method"]].append(run)
        seeds = None
        for method in spec["arms"]:
            found = {run["seed"] for run in by_method.get(method, [])}
            seeds = found if seeds is None else (seeds & found)
        seeds = sorted(seeds or [])
        if not seeds:
            failures.append(f"{spec['key']}: no seed is present in every arm")
            continue

        runs = []
        for seed in seeds:
            for method, role in spec["arms"].items():
                run = next(r for r in by_method[method] if r["seed"] == seed)
                published = run["metrics"]
                candidates = index.get(published["AP"], [])
                exact = []
                for path in candidates:
                    payload = json.loads(path.read_text())
                    if all(payload["metrics"].get(k) == v for k, v in published.items()):
                        exact.append((path, payload))
                if not exact:
                    failures.append(f"{spec['key']} seed {seed} {method}: no saved evaluation matches the published metrics")
                    continue
                path, payload = exact[0]
                digest = payload["bundle_digest"]
                gt_path = find_ground_truth(digest)
                if gt_path is None:
                    failures.append(f"{spec['key']} seed {seed} {method}: reference bundle {digest[:12]} not found")
                    continue
                reference = json.loads(gt_path.read_text())

                reduced = reduce_predictions(payload["predictions"])
                rescored = score_predictions(reduced, reference, threshold=payload.get("score_threshold", .25))
                if rescored != payload["metrics"]:
                    differing = [k for k in payload["metrics"] if rescored.get(k) != payload["metrics"][k]]
                    failures.append(f"{spec['key']} seed {seed} {method}: reduction is not lossless ({', '.join(differing)})")
                    continue

                if digest not in ground_truths:
                    gt_payload = {"images": reference["images"], "categories": reference["categories"],
                                  "annotations": [{"image_id": a["image_id"], "category_id": a["category_id"],
                                                   "bbox": a["bbox"], "area": a["area"],
                                                   "iscrowd": a.get("iscrowd", 0)} for a in reference["annotations"]]}
                    gt_bytes = json.dumps(gt_payload, separators=(",", ":"), sort_keys=True).encode()
                    name = f"labels-{digest[:16]}.json"
                    if not args.check:
                        (OUTPUT / name).write_bytes(gt_bytes)
                        written.append(name)
                    ground_truths[digest] = {
                        "path": name, "sha256": sha256_bytes(gt_bytes),
                        "source_sha256": sha256_bytes(gt_path.read_bytes()),
                        "images": len(reference["images"]), "boxes": len(reference["annotations"]),
                        "categories": [c["name"] for c in sorted(reference["categories"], key=lambda c: c["id"])]}

                blob = encode(reduced)
                run_id = f"{spec['key']}-{seed}-{spec['arms'][method]}"
                name = f"{run_id}.bin"
                if not args.check:
                    (OUTPUT / name).write_bytes(blob)
                    written.append(name)
                runs.append({
                    "id": run_id, "role": role, "method": method, "seed": seed,
                    "steps": run.get("steps"), "resolution": run.get("resolution"),
                    "bundle_digest": digest,
                    "checkpoint_sha256": payload.get("checkpoint_sha256"),
                    "score_threshold": payload.get("score_threshold", .25),
                    "source_evaluation_sha256": sha256_bytes(path.read_bytes()),
                    "source_evaluation_path": str(path.relative_to(ROOT)),
                    "detections_saved": len(payload["predictions"]),
                    "predictions": {"path": name, "sha256": sha256_bytes(blob),
                                    "count": len(reduced), "bytes": len(blob)},
                    "expected": rescored,
                })
        if runs:
            # A cohort is only a fair comparison if every arm shares the training
            # conditions. Assert it here so the site can render these as a matched
            # comparison without re-deriving the guarantee at display time.
            for field in ("steps", "resolution"):
                values = {run[field] for run in runs}
                if len(values) != 1:
                    failures.append(f"{spec['key']}: arms disagree on {field} ({sorted(values)})")
            roles = [run["role"] for run in runs]
            for role in ("naive", "aware", "complete_reference"):
                if roles.count(role) != len(seeds):
                    failures.append(f"{spec['key']}: {role} has {roles.count(role)} runs for {len(seeds)} seeds")
            cohorts.append({**{k: v for k, v in spec.items() if k != "arms"}, "seeds": seeds,
                            "steps": runs[0]["steps"], "resolution": runs[0]["resolution"], "runs": runs})

    if failures:
        for line in failures:
            print("FAILED:", line, file=sys.stderr)
        raise SystemExit(f"{len(failures)} run(s) could not be exported as verifiable evidence")
    if not cohorts:
        raise SystemExit("No cohort was exported; refusing to write an empty evidence bundle")

    manifest = {
        "version": 1,
        "metric": "COCO AP50:95 on complete validation labels",
        "scorer": {
            "implementation": "pycocotools COCOeval (bbox)",
            "iou_thresholds": [round(.5 + .05 * i, 2) for i in range(10)],
            "recall_thresholds": 101, "max_detections": 100, "area_range": "all",
            "operating_point_iou": .5,
        },
        "reduction": [
            "Only image_id, category_id, bbox and score are kept; COCO.loadRes overwrites the rest.",
            "Detections truncated to the top 100 per image and class, matching COCOeval maxDets=100.",
            "Coordinates and scores quantised to float32, the precision the detector emitted.",
            "Every run below was re-scored after reduction and reproduced its published metrics exactly.",
        ],
        "binary_format": {
            "magic": "CVB1", "header_bytes": 16, "byte_order": "little-endian",
            "layout": "int32 image_id[n], int32 category_id[n], float32 score[n], float32 bbox[4n] as x,y,w,h",
        },
        "ground_truth": ground_truths,
        "cohorts": cohorts,
    }
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    total = sum(run["predictions"]["bytes"] for cohort in cohorts for run in cohort["runs"])
    if args.check:
        existing = (OUTPUT / "manifest.json").read_bytes()
        if sha256_bytes(existing) != sha256_bytes(body):
            raise SystemExit("Committed manifest differs from a fresh build of the same evidence")
        for cohort in cohorts:
            for run in cohort["runs"]:
                path = OUTPUT / run["predictions"]["path"]
                if sha256_bytes(path.read_bytes()) != run["predictions"]["sha256"]:
                    raise SystemExit(f"Committed {path.name} does not match its rebuilt digest")
        print(f"check passed: {sum(len(c['runs']) for c in cohorts)} runs reproduce their published metrics exactly")
        return
    (OUTPUT / "manifest.json").write_bytes(body)
    print(f"Wrote {len(written) + 1} files to {OUTPUT.relative_to(ROOT)}: "
          f"{sum(len(c['runs']) for c in cohorts)} runs across {len(cohorts)} cohorts, "
          f"{total/1e6:.2f} MB of predictions, all re-scored losslessly.")


if __name__ == "__main__":
    main()
