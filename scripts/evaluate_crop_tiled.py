"""Preregister then evaluate fixed crop-training/sliced-inference interaction.

Declare before continuation scores: python scripts/evaluate_crop_tiled.py --declare-only
Run after all original tiled and continuation primary controls finish: omit the flag.
"""
import argparse
from datetime import UTC, datetime
from pathlib import Path

from coveragecv.artifacts import digest, file_digest, read_json, write_json
from coveragecv.training.tiled import PROTOCOL, evaluate_tiled

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/crop-tiled/construction/20260917"
PARENTS = ROOT / "artifacts/object-crops/construction/20260917"
CASES = ["aware_object_crops", "aware_standard", "naive_object_crops", "complete_reference_object_crops"]


def declaration():
    sources = [ROOT / "src/coveragecv/training/tiled.py", Path(__file__)]
    return {"kind": "crop_training_sliced_inference_interaction", "protocol": PROTOCOL,
            "device": "cpu", "cases": CASES,
            "continuation_protocol_sha256": file_digest(ROOT / "artifacts/object-crops/protocol.json"),
            "source_files": {p.relative_to(ROOT).as_posix(): file_digest(p) for p in sources},
            "pending_checkpoints": {case: (PARENTS / case / "detector.pt").relative_to(ROOT).as_posix()
                                    for case in CASES},
            "checkpoint_hashes": "bound after collection, before inference; unavailable at declaration",
            "comparison": "same frozen slicing on all four controls; no parameter search or class removal",
            "primary_crop_comparison_unchanged": "full-frame complete-validation AP50:95",
            "inference_cost": "five image passes per 640x640 image versus one; no cloud compute",
            "scope": "one-seed exploratory interaction; all120validationimages/all5classes; test unused"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    expected = declaration()
    frozen_path = ROOT / "artifacts/crop-tiled/declaration.json"
    if not frozen_path.exists():
        if not args.declare_only:
            raise ValueError("declare interaction before scores first")
        # Fail if any primary score is already present: do not backdate preregistration.
        if any((PARENTS / case / "evaluation.json").exists() for case in CASES):
            raise ValueError("a continuation score already exists; cannot claim declaration preceded scores")
        write_json(frozen_path, {"frozen_at": datetime.now(UTC).isoformat(), "declaration": expected,
                                "declaration_sha256": digest(expected)})
    elif read_json(frozen_path)["declaration"] != expected:
        raise ValueError("frozen interaction protocol or implementation changed")
    if args.declare_only:
        print({"declared": str(frozen_path.relative_to(ROOT)), "checkpoints": "pending"}, flush=True)
        return
    original = ROOT / "artifacts/tiled/construction/20260917/results.json"
    if not original.exists() or len(read_json(original)["runs"]) != 3:
        raise ValueError("finish all three original tiled controls first")
    primary_path = PARENTS / "comparison.json"
    if not primary_path.exists() or read_json(primary_path)["status"] != "completed":
        raise ValueError("all four primary continuation evaluations must be complete")
    primary = read_json(primary_path)
    parents = {}
    for case in CASES:
        folder = PARENTS / case
        receipt, evaluation = read_json(folder / "receipt.json"), read_json(folder / "evaluation.json")
        sha = file_digest(folder / "detector.pt")
        if (receipt["status"] != "completed" or receipt["evaluation_status"] != "completed"
                or receipt["checkpoint_sha256"] != sha or evaluation["checkpoint_sha256"] != sha
                or primary["results"][case]["checkpoint_sha256"] != sha
                or evaluation["split"] != "valid" or evaluation["device"] != "cpu"):
            raise ValueError("durable checkpoint, primary evaluation, and completed cohort must agree")
        parents[case] = {"checkpoint_sha256": sha, "evaluation_sha256": file_digest(folder / "evaluation.json")}
    binding = {"declaration_sha256": file_digest(frozen_path), "parents": parents,
               "primary_comparison_sha256": file_digest(primary_path)}
    binding_path = OUTPUT / "binding.json"
    if binding_path.exists():
        if read_json(binding_path) != binding:
            raise ValueError("collected continuation inputs changed")
    else:
        write_json(binding_path, binding)
    bundle = Path(read_json(ROOT / "artifacts/construction/experiment.json")["complete_bundle"])
    rows = []
    for case in CASES:
        folder = OUTPUT / case
        if (folder / "comparison.json").exists():
            result = read_json(folder / "comparison.json")
        else:
            print({"starting_interaction_case": case}, flush=True)
            result = evaluate_tiled(PARENTS / case / "detector.pt", bundle, folder,
                                    baseline_evaluation=PARENTS / case / "evaluation.json", device="cpu")
        rows.append({"case": case, **result})
        write_json(OUTPUT / "results.json", {"binding_sha256": file_digest(binding_path), "runs": rows})
        print({"case": case, "full_AP": result["full_frame_metrics"]["AP"]*100,
               "full_NMS_AP": result["full_frame_nms_metrics"]["AP"]*100,
               "tiled_AP": result["tiled_metrics"]["AP"]*100,
               "delta_points": result["tiled_minus_fresh_full_AP_points"]}, flush=True)


if __name__ == "__main__":
    main()
