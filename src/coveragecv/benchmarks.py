"""Verified multi-seed measurements; all variants and regressions stay visible."""
import statistics
from functools import lru_cache
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, write_json

LABELS = {"naive": "Naive merge", "aware": "Coverage-aware", "complete_reference": "Complete labels",
          "aware_continue_384": "Coverage-aware · longer training",
          "aware_augmented_512": "Coverage-aware · augmentation + 512px",
          "aware_teacher_512": "Coverage-aware · teacher + augmentation + 512px",
          "aware_teacher_cautious_512": "Teacher · 0.1× pseudo box regression",
          "aware_teacher_classonly_512": "Teacher · classification-only pseudo labels",
          "naive_augmented_512": "Naive · augmentation + 512px",
          "complete_augmented_512": "Complete labels · augmentation + 512px",
          "naive_large_704_ema": "Naive · Large + 704px + EMA",
          "aware_large_704_ema": "Coverage-aware · Large + 704px + EMA",
          "complete_reference_large_704_ema": "Complete labels · Large + 704px + EMA"}


@lru_cache(maxsize=256)
def _verified_checkpoint(path, modified_ns, size):
    return file_digest(Path(path))


def measured_run(folder, *, reference_digest, method, seed, expected_steps):
    if not (folder / "receipt.json").exists():
        return None
    run, evaluation = read_json(folder / "run.json"), read_json(folder / "evaluation.json")
    checkpoint = folder / "detector.pt"
    stat = checkpoint.stat()
    sha = _verified_checkpoint(str(checkpoint), stat.st_mtime_ns, stat.st_size)
    if (sha != run["detector_sha256"] or sha != evaluation["checkpoint_sha256"]
            or evaluation["bundle_digest"] != reference_digest or evaluation["split"] != "valid"
            or run["seed"] != seed or run["steps"] != expected_steps or run["status"] != "completed"
            or run.get("method", run["arm"]) != method):
        raise ValueError(f"Experiment binding failed: {folder}")
    audit = read_json(folder / "pseudo-labels.json") if (folder / "pseudo-labels.json").exists() else None
    return {"method": method, "seed": seed, "metrics": evaluation["metrics"],
            "checkpoint_sha256": sha, "initial_parameter_digest": run["initial_parameter_digest"],
            "start_parameter_digest": run.get("start_parameter_digest", run["initial_parameter_digest"]),
            "warm_start_sha256": run.get("warm_start_sha256"), "view_digest": run["view_digest"],
            "original_view_digest": run.get("original_view_digest", run["view_digest"]),
            "steps": run.get("total_training_steps", run["steps"]), "batch": run["batch"],
            "resolution": run["model_config"]["resolution"], "recipe": run.get("recipe", "pilot"),
            "elapsed_seconds": run["elapsed_seconds"], "device": run["device"],
            "pseudo_labels": {k: v for k, v in audit.items() if k != "predictions"} if audit else None,
            "folder": str(folder.resolve())}


def stats(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "min": min(values), "max": max(values)}


def summarize(rows):
    if len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("Repeated seeds cannot increase the sample count")
    budget = {(r["steps"], r["batch"], r["resolution"], r["recipe"], r["device"], r["original_view_digest"])
              for r in rows}
    if len(budget) != 1:
        raise ValueError("Cannot pool measurements with different training contracts")
    metrics = {k: stats([r["metrics"][k] for r in rows])
               for k in ("AP", "AP50", "AP75", "recall_at_threshold", "precision_at_threshold",
                         "false_positives_per_image")}
    return {"label": LABELS[rows[0]["method"]], "seeds": sorted(r["seed"] for r in rows),
            "n": len(rows), "metrics": metrics, "steps": rows[0]["steps"], "resolution": rows[0]["resolution"],
            "per_class_AP": {k: stats([r["metrics"]["per_class_AP"][k] for r in rows
                                      if r["metrics"]["per_class_AP"][k] is not None])
                             for k, v in rows[0]["metrics"]["per_class_AP"].items() if v is not None}}


def paired_delta(rows, left, right):
    a, b = ({r["seed"]: r for r in rows if r["method"] == method} for method in (left, right))
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return None
    for seed in shared:
        if (a[seed]["initial_parameter_digest"] != b[seed]["initial_parameter_digest"]
                or a[seed]["steps"] != b[seed]["steps"] or a[seed]["batch"] != b[seed]["batch"]
                or a[seed]["device"] != b[seed]["device"]
                or a[seed]["resolution"] != b[seed]["resolution"]):
            raise ValueError("Paired comparison has unmatched seeds, budgets or resolutions")
        if left.startswith("aware_teacher") and a[seed]["warm_start_sha256"] != b[seed]["warm_start_sha256"]:
            raise ValueError("Teacher ablation must share its exact starting checkpoint")
    return {"left": left, "right": right, "seeds": shared,
            "AP_points": stats([100*(a[s]["metrics"]["AP"]-b[s]["metrics"]["AP"]) for s in shared])}


def build_summary(workspace: Path):
    artifacts = workspace / "artifacts"
    tasks, completed, submitted, issues = {}, 0, 0, []
    for task, experiment in (("pawns", artifacts / "chess_experiment.json"),
                             ("all-pieces", artifacts / "full_chess/experiment.json"),
                             ("construction", artifacts / "construction/experiment.json")):
        if not experiment.exists():
            continue
        ex = read_json(experiment)
        reference_digest = read_json(Path(ex["complete_bundle"]) / "manifest.json")["digest"]
        rows, methods = [], {}
        selected_methods = list(LABELS) if task != "construction" else ["naive", "aware", "complete_reference"]
        for method in selected_methods:
            stage = "capacity" if method.endswith("large_704_ema") else (
                "gpu" if method in ("naive", "aware", "complete_reference") else "improved")
            base = artifacts / stage / task
            for seed in (20260917, 20260918, 20260919):
                folder = base / str(seed) / method
                submitted += (folder / "call.json").exists()
                try:
                    result = measured_run(folder, reference_digest=reference_digest, method=method,
                                           seed=seed, expected_steps=4000 if task == "construction" else 2000)
                except (ValueError, OSError, KeyError) as exc:
                    issues.append({"task": task, "seed": seed, "method": method, "error": str(exc)})
                    continue
                if result:
                    rows.append(result)
            group = [r for r in rows if r["method"] == method]
            if group:
                methods[method] = summarize(group)
        completed += len(rows)
        pairs = [paired_delta(rows, "aware", "naive"),
                 paired_delta(rows, "aware_augmented_512", "naive_augmented_512"),
                 paired_delta(rows, "aware_teacher_512", "aware_augmented_512"),
                 paired_delta(rows, "aware_teacher_cautious_512", "aware_teacher_512"),
                 paired_delta(rows, "aware_teacher_classonly_512", "aware_teacher_512"),
                 paired_delta(rows, "aware_large_704_ema", "naive_large_704_ema")]
        tasks[task] = {"name": {"pawns": "Two pawn classes", "all-pieces": "All 13 chess classes",
                               "construction": "Construction safety"}[task],
                       "independent_domain": task == "construction",
                       "classes": read_json(Path(ex["partial_view"]) / "ontology.json")["classes"],
                       "methods": methods, "runs": rows, "comparisons": [p for p in pairs if p],
                       "reference_digest": reference_digest}
        if task == "construction":
            tasks[task]["reference_quality"] = {
                "status": "published_annotations_with_known_defects",
                "source": "Unchanged published validation annotations",
                "independently_verified_exhaustive": False,
                "labels_splits_and_scores_unchanged": True,
                "audit_document": "docs/CONSTRUCTION_LABEL_AUDIT.md",
                "findings": [
                    {"image_id": 998, "finding": "A full-person box is labeled helmet alongside a tight helmet box.",
                     "annotation_ids": [2132, 2134], "box_IoU": 0.097872, "area_ratio": 10.2174},
                    {"image_id": 1024, "finding": "One visible helmet has two differently sized helmet references.",
                     "annotation_ids": [2275, 2276], "box_IoU": 0.451589, "area_ratio": 2.2144},
                    {"image_id": 1002, "finding": "Only two of five visible people have reference annotations; no ignore regions mark the others.",
                     "visible_people": 5, "annotated_people": 2,
                     "fixed_threshold_no_helmet_false_positives_on_unannotated_faces": 3},
                ],
                "interpretation": "Targeted visual audit, not a defect-prevalence estimate. Known reference defects do not explain all model errors or the overall AP gap.",
            }
    result = {"tasks": tasks, "completed_runs": completed, "submitted_runs": submitted, "planned_runs": 64,
              "issues": issues, "metric": "Complete-validation COCO AP50:95",
              "uncertainty": "Mean ± sample standard deviation across training seeds, not a confidence interval.",
              "limitations": ["The two chess tasks share one small board/camera domain. Construction safety is a separate dataset.",
                              "The full ontology retains a generic bishop class with zero training positives.",
                              "Stage-two recipes are exploratory; validation guides engineering and is not an untouched final test.",
                              "The separate Roboflow-hosted baseline has used the original test split.",
                              "Construction splits are repaired using filename, exact-byte and perceptual-hash groups; scene independence is not guaranteed.",
                              "Construction uses unchanged published validation annotations with known inconsistent boxes and missing annotations. They are not independently verified as exhaustive; a targeted audit does not establish defect prevalence or explain the full AP gap.",
                              "Stage two restarts the optimizer. Teacher mining adds forward-pass compute beyond matched updates.",
                              "Large/704px/EMA is a separate one-seed combined-recipe pilot, not an isolated model-size ablation.",
                              "CUDA nondeterministic operators warn; exact bitwise reproduction is not guaranteed."]}
    incident = artifacts / "cloud_incident.json"
    if incident.exists():
        record = read_json(incident)
        result["cloud_status"] = record["status"]
        result["stopped_runs"] = len(record["cancelled"])
        result["limitations"].append("Cloud jobs were interrupted after conflicting billing readings; three Large controls were explicitly authorized to restart. The nine-run construction test plan remains incomplete and unexecuted.")
    write_json(artifacts / "research_summary.json", result)
    return result
