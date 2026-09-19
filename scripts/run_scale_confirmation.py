"""Resume the frozen test/tile comparison; never retrain or alter test-selected recipes."""
import fcntl
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_scale_confirmation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/continuous/confirmation"


def collect(case, protocol, sha):
    folder = OUT / "cases" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = {"case": case, "protocol_sha256": sha}
        if not (folder / "receipt.json").exists():
            if (folder / "call.json").exists():
                saved = read_json(folder / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved call differs from frozen request")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (folder / "reservation.json").exists():
                    raise RuntimeError("Orphan reservation: reconcile before any new submission")
                reservation = reserve_scale_confirmation()
                write_json(folder / "reservation.json", {"request": request, "reservation_id": reservation})
                call = modal.Function.from_name("coveragecv-scale-confirmation", "evaluate").spawn(**request)
                write_json(folder / "call.json", {"request": request, "reservation_id": reservation, "call_id": call.object_id})
                print({"case": case, "call_id": call.object_id}, flush=True)
            with zipfile.ZipFile(io.BytesIO(call.get())) as archive:
                if not {"receipt.json", "test.json"}.issubset(archive.namelist()):
                    raise ValueError("Incomplete confirmation evidence")
                for name in sorted(archive.namelist(), key=lambda n: n == "receipt.json"):
                    if not name.endswith(".json"):
                        raise ValueError("Unexpected confirmation artifact")
                    target = safe_child(folder, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    write_json(target, json.loads(archive.read(name)))
        receipt = read_json(folder / "receipt.json")
        if (receipt["protocol_sha256"] != sha or receipt["case"] != case
                or receipt["checkpoint_sha256"] != protocol["cases"][case]["checkpoint_sha256"]
                or receipt["bundle_digest"] != protocol["reference"]["bundle_digest"]):
            raise ValueError("Confirmation receipt does not bind this protocol")
        if receipt["status"] != "completed":
            raise RuntimeError(f"Primary test evidence saved; secondary stage failed: {receipt.get('failure')}")
        print({"case": case, "test_AP": receipt["test_metrics"]["AP"]*100,
               "validation_tiled_AP": receipt["validation_tiled_metrics"]["AP"]*100}, flush=True)
        return receipt


def main():
    protocol = read_json(OUT / "protocol.json")
    sha = file_digest(OUT / "protocol.json")
    state = {"protocol_sha256": sha, "receipts": {}, "failures": {}, "status": "scoring"}
    with (OUT / ".loop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(collect, case, protocol, sha): case for case in protocol["cases"]}
            for future in as_completed(jobs):
                case = jobs[future]
                try:
                    state["receipts"][case] = future.result()
                except Exception as error:  # noqa: BLE001 -- preserve independent paid outcomes
                    state["failures"][case] = {"error_type": type(error).__name__, "error": str(error)}
                write_json(OUT / "state.json", state)
        state["status"] = "needs_reconciliation" if state["failures"] else "choose_next_hypothesis"
        write_json(OUT / "state.json", state)


if __name__ == "__main__":
    main()
