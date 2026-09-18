"""Resume smoke → train → collect → GPU score; never execute a local model."""
import fcntl
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_continuous_scale

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/scale"


def invoke(folder, stage, request, arguments):
    """Reserve before submission; reconnect saved IDs; fail closed on ambiguous submission."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "call.json").exists():
        saved = read_json(folder / "call.json")
        if saved["request"] != request:
            raise ValueError("Saved cloud call belongs to a different request")
        call = modal.FunctionCall.from_id(saved["call_id"])
    else:
        if (folder / "reservation.json").exists():
            raise RuntimeError("Orphan reservation: reconcile provider submission before proceeding")
        reservation = reserve_continuous_scale(stage)
        write_json(folder / "reservation.json", {"reservation_id": reservation, "request": request})
        function = "train_case" if stage == "train" else stage
        call = modal.Function.from_name("coveragecv-continuous-scale", function).spawn(**arguments)
        write_json(folder / "call.json", {"reservation_id": reservation, "request": request, "call_id": call.object_id})
        print({"stage": stage, "case": request.get("case"), "call_id": call.object_id}, flush=True)
    return call.get()


def training(case, spec, protocol, sha):
    folder = OUT / "runs" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = {"case": case, "protocol_sha256": sha}
        receipt_path = folder / "receipt.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["request"] != request or receipt["spec"] != spec:
                raise ValueError("Saved training receipt differs from protocol")
        else:
            returned = invoke(folder, "train", request, request)
            with zipfile.ZipFile(io.BytesIO(returned)) as archive:
                if set(archive.namelist()) != {"run.json", "detector.pt", "progress.json"}:
                    raise ValueError("Unexpected checkpoint inventory")
                for name in archive.namelist():
                    safe_child(folder, name).write_bytes(archive.read(name))
        run = read_json(folder / "run.json")
        parent = protocol["parents"][spec["arm"]]
        checkpoint_sha = file_digest(folder / "detector.pt")
        if (run["status"] != "completed" or run["detector_sha256"] != checkpoint_sha
                or run["view_digest"] != parent["view_digest"]
                or run["warm_start_sha256"] != parent["checkpoint_sha256"]
                or run["exposure_plan_digest"] != protocol["exposure_plans"][spec["sampling"]]["digest"]
                or any(run[k] != spec[k] for k in ("steps", "arm", "alpha", "exclusive_groups", "resolution", "seed"))):
            raise ValueError("Collected checkpoint violates frozen training contract")
        receipt = {"request": request, "spec": spec, "checkpoint_sha256": checkpoint_sha,
                   "training_seconds": run["elapsed_seconds"]}
        write_json(receipt_path, receipt)
        print({"case": case, "status": "checkpoint_collected", "seconds": run["elapsed_seconds"]}, flush=True)
        return receipt


def scoring(case, protocol, sha):
    folder = OUT / "runs" / case
    receipt = read_json(folder / "receipt.json")
    checkpoint = folder / "detector.pt"
    checkpoint_sha = file_digest(checkpoint)
    if checkpoint_sha != receipt["checkpoint_sha256"]:
        raise ValueError("Collected checkpoint changed before scoring")
    request = {"case": case, "checkpoint_sha256": checkpoint_sha, "protocol_sha256": sha}
    if (folder / "evaluation.json").exists():
        result = read_json(folder / "evaluation.json")
    else:
        result = invoke(folder / "scoring", "evaluate", request,
                        {"checkpoint": checkpoint.read_bytes(), "checkpoint_sha256": checkpoint_sha,
                         "protocol_sha256": sha})
    if (result["experiment_protocol_sha256"] != sha or result["checkpoint_sha256"] != checkpoint_sha
            or result["bundle_digest"] != protocol["reference_bundle_digest"] or result["device"] != "cuda"
            or result["resolution"] != protocol["cases"][case]["resolution"]):
        raise ValueError("Evaluation does not match checkpoint, device and reference")
    write_json(folder / "evaluation.json", result)
    print({"case": case, "AP50_95": result["metrics"]["AP"] * 100}, flush=True)
    return result["metrics"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".loop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read_json(OUT / "protocol.json")
        sha = file_digest(OUT / "protocol.json")
        state = {"status": "gpu_smoke", "protocol_sha256": sha, "receipts": {}, "metrics": {}, "failures": {}}
        write_json(OUT / "state.json", state)
        smoke_path = OUT / "smoke/result.json"
        if smoke_path.exists():
            result = read_json(smoke_path)
        else:
            request = {"protocol_sha256": sha}
            result = invoke(OUT / "smoke", "smoke", request, request)
            write_json(smoke_path, result)
        if result["status"] != "passed" or result["protocol_sha256"] != sha:
            raise ValueError("GPU smoke has not passed for this exact protocol")
        state["status"] = "training"
        write_json(OUT / "state.json", state)
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs = {pool.submit(training, case, spec, protocol, sha): case for case, spec in protocol["cases"].items()}
            for future in as_completed(jobs):
                case = jobs[future]
                try:
                    state["receipts"][case] = future.result()
                except Exception as error:  # noqa: BLE001 -- retain all independent paid checkpoints
                    state["failures"][case] = {"stage": "training", "error_type": type(error).__name__, "error": str(error)}
                write_json(OUT / "state.json", state)
        state["status"] = "gpu_evaluation"
        write_json(OUT / "state.json", state)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(scoring, case, protocol, sha): case for case in state["receipts"]}
            for future in as_completed(jobs):
                case = jobs[future]
                try:
                    state["metrics"][case] = future.result()
                except Exception as error:  # noqa: BLE001 -- collect other independent evaluations
                    state["failures"][case] = {"stage": "evaluation", "error_type": type(error).__name__, "error": str(error)}
                write_json(OUT / "state.json", state)
        state["status"] = "needs_reconciliation" if state["failures"] else "choose_next_hypothesis"
        state["goal_complete"] = False
        write_json(OUT / "state.json", state)
        print({"status": state["status"], "evaluated": len(state["metrics"]), "failures": len(state["failures"])}, flush=True)


if __name__ == "__main__":
    main()
