"""Export all completed outcomes with exact checkpoint and evaluation provenance."""
from datetime import UTC, datetime
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_v2"


def main():
    protocol = read_json(OUT / "protocol.json")
    scoring = read_json(OUT / "evaluation_protocol.json")
    protocol_sha, scoring_sha = file_digest(OUT / "protocol.json"), file_digest(OUT / "evaluation_protocol.json")
    tasks = {}
    for task, dataset in protocol["datasets"].items():
        reference = read_json(Path(dataset["local_reference"]) / "splits/valid.coco.json")
        rows = {}

        def checked_evaluation(path, checkpoint_sha, *, task=task, reference=reference):
            result = read_json(path)
            metrics = result["metrics"]
            class_ap = [v for v in metrics["per_class_AP"].values() if v is not None]
            if (result["checkpoint_sha256"] != checkpoint_sha or result["evaluation_protocol_sha256"] != scoring_sha
                    or result["bundle_digest"] != scoring["references"][task]["bundle_digest"]
                    or result["split"] != "valid" or result["device"] != "cuda"
                    or metrics["images"] != len(reference["images"])
                    or metrics["boxes"] != len(reference["annotations"])
                    or abs(sum(class_ap)/len(class_ap)-metrics["AP"]) > 1e-10):
                raise ValueError(f"Evaluation provenance or metric audit failed for {path}")
            return {"checkpoint_sha256": checkpoint_sha, "evaluation_sha256": file_digest(path),
                    "metrics": metrics, "resolution": result["resolution"], "model_variant": result["model_variant"]}

        for arm, parent in dataset["parents"].items():
            rows[f"parent_{arm}"] = checked_evaluation(OUT / "parent_evaluations" / f"{task}-{arm}.json",
                                                      parent["checkpoint_sha256"])
        for case, spec in protocol["cases"].items():
            if spec["task"] != task:
                continue
            folder = OUT / "runs" / case
            receipt, run = read_json(folder / "receipt.json"), read_json(folder / "run.json")
            checkpoint_sha = file_digest(folder / "detector.pt")
            if (receipt["protocol_sha256"] != protocol_sha or receipt["spec"] != spec
                    or receipt["checkpoint_sha256"] != checkpoint_sha or run["status"] != "completed"
                    or run["steps"] != spec["steps"] or run["detector_sha256"] != checkpoint_sha):
                raise ValueError(f"Training receipt failed for {case}")
            row = checked_evaluation(folder / "evaluation.json", checkpoint_sha)
            row.update(spec=spec, annotation_count=run["annotation_count"],
                       total_training_steps=run["total_training_steps"], training_seconds=run["elapsed_seconds"],
                       parent_sha256=run["warm_start_sha256"], view_digest=run["view_digest"])
            rows[case.removeprefix(task+"-")] = row
        control = rows["control"]["metrics"]["AP"]
        for name in ("exclusive", "exclusive_power"):
            rows[name]["delta_vs_continuation_control_points"] = 100*(rows[name]["metrics"]["AP"]-control)
        tasks[task] = rows
        print(task, {name: round(row["metrics"]["AP"]*100, 4) for name, row in rows.items()})
    cascade = read_json(ROOT / "artifacts/localization_v2/all-pieces/aware/train_selection.json")
    pilot = read_json(ROOT / "artifacts/localization_v2/quality_matrix_results.json")
    result = {"status": "completed", "generated_at": datetime.now(UTC).isoformat(),
              "training_protocol_sha256": protocol_sha, "evaluation_protocol_sha256": scoring_sha,
              "training_source_files": protocol["source_files"], "evaluation_source_files": scoring["source_files"],
              "metric": "Unchanged complete-validation COCO AP50:95; metrics are fractions, deltas percentage points",
              "single_seed": True, "test_set_evaluated": False, "tasks": tasks,
              "rejected_cascade_train_selection": cascade, "local_pilot": pilot,
              "limitations": ["One continuation seed; validation used repeatedly across development.",
                  "Chess tasks share one dataset/domain; construction reference has documented annotation defects.",
                  "Construction aware variants share the same previously acquired 265 additional train boxes.",
                  "Parents and new runs are rescored on CUDA; older published CPU results may differ numerically.",
                  "Complete labels are a controlled reference, not a frontier benchmark or guaranteed upper bound."]}
    tiled_path = OUT / "tiled_results.json"
    if tiled_path.exists():
        tiled = read_json(tiled_path)
        if tiled["status"] != "completed" or tiled["failures"] or len(tiled["results"]) != 6:
            raise ValueError("Tiled follow-up is incomplete")
        result["secondary_tiled"] = tiled
        result["tiling_protocol_sha256"] = file_digest(OUT / "tiling_protocol.json")
    segment_path = ROOT / "artifacts/segment_refinement/result.json"
    if segment_path.exists():
        result["segmentation_refinement"] = read_json(segment_path)
    write_json(ROOT / "docs/RESEARCH_V2_RESULTS.json", result)


if __name__ == "__main__":
    main()
