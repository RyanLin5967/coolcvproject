"""Reconnect fixed acquisition runs; persist every checkpoint before local evaluation."""
import fcntl
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal
import torch

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_acquisition_run
from coveragecv.training.evaluate import evaluate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/acquisition/construction/20260917"
PROTOCOL = ROOT / "artifacts/acquisition/protocol.json"
CASES = {"guided": "aware", "random": "aware", "complete_standard": "complete_reference"}


def collect(case, protocol):
    arm = CASES[case]
    folder = OUTPUT / case
    folder.mkdir(parents=True, exist_ok=True)
    request = {"case": case, "protocol_sha256": file_digest(PROTOCOL)}
    if not (folder / "receipt.json").exists():
        call_path = folder / "call.json"
        if call_path.exists():
            saved = read_json(call_path)
            if saved["request"] != request:
                raise ValueError("saved acquisition call belongs to another protocol")
            call = modal.FunctionCall.from_id(saved["call_id"])
        else:
            if (folder / "reservation.json").exists():
                raise RuntimeError("reserved call has no saved ID; reconcile provider state before resubmitting")
            # A reservation is never released automatically, including on network errors.
            reservation = reserve_acquisition_run()
            write_json(folder / "reservation.json", {"id": reservation, "request": request})
            call = modal.Function.from_name("coveragecv-acquisition", "experiment").spawn(case=case)
            write_json(call_path, {"call_id": call.object_id, "request": request,
                                   "reservation_id": reservation})
        returned = call.get()
        with zipfile.ZipFile(io.BytesIO(returned)) as archive:
            if set(archive.namelist()) != {"run.json", "detector.pt", "progress.json", "compute.json"}:
                raise ValueError("unexpected acquisition result archive")
            for name in archive.namelist():
                target = safe_child(folder, name)
                temporary = target.with_suffix(target.suffix + ".partial")
                temporary.write_bytes(archive.read(name))
                temporary.replace(target)
    run = read_json(folder / "run.json")
    sha = file_digest(folder / "detector.pt")
    parent = protocol["parents"][arm]["checkpoint_sha256"]
    if (run["status"] != "completed" or run["arm"] != arm or run["method"] != case
            or run["seed"] != protocol["seed"] or run["steps"] != protocol["steps"]
            or run["total_training_steps"] != protocol["total_training_steps"]
            or run["detector_sha256"] != sha or run["view_digest"] != protocol["views"][case]["digest"]
            or run["parent_checkpoint_sha256"] != parent or run["warm_start_sha256"] != parent
            or run["experiment_protocol_sha256"] != request["protocol_sha256"]
            or run["recipe"] != protocol["recipe"] or run["batch"] != protocol["batch"]
            or run["model_config"]["resolution"] != protocol["resolution"]
            or run["validation_during_training"] is not False
            or run["annotation_acquisition"] != protocol["views"][case]["acquisition"]):
        raise ValueError("returned checkpoint does not satisfy the fixed acquisition contract")
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        if read_json(receipt_path)["checkpoint_sha256"] != sha:
            raise ValueError("persisted acquisition checkpoint receipt mismatch")
    else:
        write_json(receipt_path, {"checkpoint_sha256": sha, "status": "trained", "evaluation_status": "pending"})
    print({"case": case, "status": "durable_checkpoint_collected", "sha256": sha}, flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(PROTOCOL)
        torch.set_num_threads(4)
        # Finish checkpoint collection for all cases even if one future fails.
        failures = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(collect, case, protocol): case for case in CASES}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - persist all independent checkpoints before failing
                    failures.append({"case": futures[future], "error": repr(error)})
        if failures:
            write_json(OUTPUT / "collection_failures.json", failures)
            raise RuntimeError(f"{len(failures)} acquisition calls failed; no automatic resubmission")
        bundle = Path(read_json(ROOT / "artifacts/construction/experiment.json")["complete_bundle"])
        folders = {case: OUTPUT / case for case in CASES}
        folders["zero_review"] = ROOT / protocol["zero_review_control"]["folder"]
        rows = {}
        for case, folder in folders.items():
            checkpoint = folder / "detector.pt"
            target = folder / "evaluation.json"
            result = read_json(target) if target.exists() else evaluate_checkpoint(checkpoint, bundle, target, device="cpu")
            sha = file_digest(checkpoint)
            if (result["checkpoint_sha256"] != sha or result["split"] != "valid"
                    or result["bundle_digest"] != protocol["reference_bundle_digest"]
                    or result["device"] != "cpu" or result["resolution"] != protocol["resolution"]
                    or result["score_threshold"] != 0.25
                    or result["postprocess"] != "stock RF-DETR; reserved output omitted from semantic COCO mapping"):
                raise ValueError("acquisition evaluation is bound to different weights or reference")
            if case == "zero_review" and sha != protocol["zero_review_control"]["checkpoint_sha256"]:
                raise ValueError("zero-review control checkpoint changed")
            if case in CASES:
                write_json(folder / "receipt.json", {"checkpoint_sha256": sha, "status": "completed",
                                                     "evaluation_status": "completed"})
            rows[case] = {"metrics": result["metrics"], "checkpoint_sha256": sha,
                          "evaluation_sha256": file_digest(target),
                          "acquisition": protocol["views"][case]["acquisition"] if case in CASES else None}
            print({"case": case, "AP": result["metrics"]["AP"], "AP50": result["metrics"]["AP50"]}, flush=True)
        ap = {case: row["metrics"]["AP"] * 100 for case, row in rows.items()}
        comparison = {"status": "completed", "results": rows, "seed": protocol["seed"],
            "protocol_sha256": file_digest(PROTOCOL), "review_units_per_acquisition_arm": 150,
            "primary_AP_delta_points": ap["guided"] - ap["random"],
            "guided_vs_zero_review_AP_delta_points": ap["guided"] - ap["zero_review"],
            "random_vs_zero_review_AP_delta_points": ap["random"] - ap["zero_review"],
            "limitations": protocol["limitations"]}
        write_json(OUTPUT / "comparison.json", comparison)
        print({"primary_guided_minus_random_AP_points": comparison["primary_AP_delta_points"]}, flush=True)


if __name__ == "__main__":
    main()
