"""Resume four fixed construction continuations; collect weights before local scoring."""
import fcntl
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import torch

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_object_crop_run
from coveragecv.training.evaluate import evaluate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/object-crops/construction/20260917"
CASES = {"aware_standard": ("aware", False), "aware_object_crops": ("aware", True),
         "naive_object_crops": ("naive", True), "complete_reference_object_crops": ("complete_reference", True)}


def collect(case, protocol):
    arm, crops = CASES[case]
    folder = OUTPUT / case
    folder.mkdir(parents=True, exist_ok=True)
    request = {"case": case, "protocol_sha256": file_digest(ROOT / "artifacts/object-crops/protocol.json")}
    if not (folder / "receipt.json").exists():
        if (folder / "call.json").exists():
            saved = read_json(folder / "call.json")
            if saved["request"] != request:
                raise ValueError("saved crop call belongs to another protocol")
            call = modal.FunctionCall.from_id(saved["call_id"])
        else:
            reservation = reserve_object_crop_run()
            call = modal.Function.from_name("coveragecv-object-crops", "experiment").spawn(case=case)
            write_json(folder / "call.json", {"call_id": call.object_id, "request": request,
                                              "reservation_id": reservation})
        returned = call.get()
        with zipfile.ZipFile(io.BytesIO(returned)) as archive:
            for name in archive.namelist():
                safe_child(folder, name).write_bytes(archive.read(name))
    run = read_json(folder / "run.json")
    sha = file_digest(folder / "detector.pt")
    expected_view = protocol["complete_view_digest" if arm == "complete_reference" else "partial_view_digest"]
    if (run["status"] != "completed" or run["arm"] != arm or run["method"] != case
            or run["seed"] != 20260917 or run["steps"] != 2000 or run["total_training_steps"] != 6000
            or run["detector_sha256"] != sha or run["view_digest"] != expected_view
            or run["parent_checkpoint_sha256"] != protocol["parents"][arm]["checkpoint_sha256"]
            or run["warm_start_sha256"] != protocol["parents"][arm]["checkpoint_sha256"]
            or run["experiment_protocol_sha256"] != request["protocol_sha256"]
            or run["object_crop_plan_digest"] != (protocol["crop_plan_digest"] if crops else None)):
        raise ValueError("returned detector does not satisfy the fixed training contract")
    if (folder / "receipt.json").exists():
        if read_json(folder / "receipt.json")["checkpoint_sha256"] != sha:
            raise ValueError("persisted crop checkpoint receipt mismatch")
    else:
        write_json(folder / "receipt.json", {"checkpoint_sha256": sha, "status": "trained",
                                            "evaluation_status": "pending"})
    print({"case": case, "status": "durable_checkpoint_collected", "sha256": sha}, flush=True)
    return case


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        protocol = read_json(ROOT / "artifacts/object-crops/protocol.json")
        ex = read_json(ROOT / "artifacts/construction/experiment.json")
        bundle = Path(ex["complete_bundle"])
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(collect, case, protocol) for case in CASES]
            for future in as_completed(futures):
                case = future.result()
                folder = OUTPUT / case
                checkpoint = folder / "detector.pt"
                target = folder / "evaluation.json"
                if target.exists():
                    result = read_json(target)
                else:
                    result = evaluate_checkpoint(checkpoint, bundle, target, device="cpu")
                if (result["checkpoint_sha256"] != file_digest(checkpoint) or result["split"] != "valid"
                        or result["bundle_digest"] != protocol["reference_bundle_digest"]):
                    raise ValueError("crop evaluation is bound to different weights or reference")
                write_json(folder / "receipt.json", {"checkpoint_sha256": file_digest(checkpoint),
                    "status": "completed", "evaluation_status": "completed"})
                print({"case": case, "AP": result["metrics"]["AP"], "AP50": result["metrics"]["AP50"]}, flush=True)
        rows = {}
        for case, (arm, _) in CASES.items():
            result = read_json(OUTPUT / case / "evaluation.json")
            parent = read_json(ROOT / "artifacts/gpu/construction/20260917" / arm / "evaluation.json")
            rows[case] = {"metrics": result["metrics"], "parent_metrics": parent["metrics"],
                          "checkpoint_sha256": result["checkpoint_sha256"],
                          "parent_checkpoint_sha256": parent["checkpoint_sha256"]}
        delta = 100*(rows["aware_object_crops"]["metrics"]["AP"]-rows["aware_standard"]["metrics"]["AP"])
        write_json(OUTPUT / "comparison.json", {"status": "completed", "results": rows,
            "primary_AP_delta_points": delta, "seed": 20260917, "crop_plan_digest": protocol["crop_plan_digest"],
            "protocol_sha256": file_digest(ROOT / "artifacts/object-crops/protocol.json"),
            "interpretation": "One-seed exploratory comparison; validation guided design; test labels unevaluated"})
        print({"primary_AP_delta_points": delta}, flush=True)


if __name__ == "__main__":
    main()
