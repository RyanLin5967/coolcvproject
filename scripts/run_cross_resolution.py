"""Collect the fixed GPU inference matrix, preserving paid calls across restarts."""
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, write_json
from coveragecv.training.cloud_budget import reserve_cross_resolution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/cross_resolution"


def collect(case, spec, protocol, sha):
    folder = OUT / "cases" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = {"case": case, "protocol_sha256": sha}
        if (folder / "evaluation.json").exists():
            result = read_json(folder / "evaluation.json")
        else:
            if (folder / "call.json").exists():
                saved = read_json(folder / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved call differs from request")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (folder / "reservation.json").exists():
                    raise RuntimeError("Orphan reservation: reconcile submission before proceeding")
                reservation = reserve_cross_resolution()
                write_json(folder / "reservation.json", {"request": request, "reservation_id": reservation})
                call = modal.Function.from_name("coveragecv-cross-resolution", "evaluate").spawn(**request)
                write_json(folder / "call.json", {"request": request, "reservation_id": reservation,
                                                  "call_id": call.object_id})
                print({"case": case, "call_id": call.object_id}, flush=True)
            result = call.get()
        if (result["experiment_protocol_sha256"] != sha or result["case"] != case
                or result["checkpoint_sha256"] != protocol["checkpoints"][spec["checkpoint"]]["checkpoint_sha256"]
                or result["bundle_digest"] != protocol["references"][spec["task"]]["bundle_digest"]
                or result["resolution"] != spec["resolution"] or result["device"] != "cuda"):
            raise ValueError("Evaluation result violates frozen case contract")
        write_json(folder / "evaluation.json", result)
        print({"case": case, "AP50_95": result["metrics"]["AP"]*100}, flush=True)
        return result["metrics"]


def main():
    protocol = read_json(OUT / "protocol.json")
    sha = file_digest(OUT / "protocol.json")
    state = {"protocol_sha256": sha, "metrics": {}, "failures": {}, "status": "scoring"}
    with (OUT / ".loop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(collect, case, spec, protocol, sha): case for case, spec in protocol["cases"].items()}
            for future in as_completed(jobs):
                case = jobs[future]
                try:
                    state["metrics"][case] = future.result()
                except Exception as error:  # noqa: BLE001 -- retain all independently paid evaluations
                    state["failures"][case] = {"error_type": type(error).__name__, "error": str(error)}
                write_json(OUT / "state.json", state)
        state["status"] = "needs_reconciliation" if state["failures"] else "choose_next_hypothesis"
        write_json(OUT / "state.json", state)
        print({"status": state["status"], "completed": len(state["metrics"])}, flush=True)


if __name__ == "__main__":
    main()
