"""Reserve, collect, and verify the frozen GPU matrix; scoring runs separately on GPUs."""
import argparse
import fcntl
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_research_v2_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_v2"


def collect(case, spec, protocol):
    folder = OUT / "runs" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = {"case": case, "protocol_sha256": file_digest(OUT / "protocol.json")}
        parent = protocol["datasets"][spec["task"]]["parents"][spec["arm"]]
        if (folder / "receipt.json").exists():
            saved_receipt = read_json(folder / "receipt.json")
            if (saved_receipt["protocol_sha256"] != request["protocol_sha256"]
                    or saved_receipt["spec"] != spec or saved_receipt["case"] != case):
                raise ValueError("Saved receipt belongs to a different protocol or case")
        if not (folder / "receipt.json").exists():
            if (folder / "call.json").exists():
                saved = read_json(folder / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved request differs from this frozen protocol")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (folder / "reservation.json").exists():
                    raise RuntimeError("Orphan reservation: resolve existing submission before any retry")
                reservation = reserve_research_v2_run()
                write_json(folder / "reservation.json", {"reservation_id": reservation, "request": request})
                remote = modal.Function.from_name("coveragecv-research-v2", "experiment")
                call = remote.spawn(**request)
                write_json(folder / "call.json", {"reservation_id": reservation, "request": request,
                                                  "call_id": call.object_id})
                print({"case": case, "call_id": call.object_id, "status": "submitted"}, flush=True)
            returned = call.get()
            with zipfile.ZipFile(io.BytesIO(returned)) as archive:
                if set(archive.namelist()) != {"run.json", "detector.pt", "progress.json"}:
                    raise ValueError("Unexpected returned artifact inventory")
                for name in archive.namelist():
                    safe_child(folder, name).write_bytes(archive.read(name))
        run = read_json(folder / "run.json")
        sha = file_digest(folder / "detector.pt")
        if (run["status"] != "completed" or run["detector_sha256"] != sha
                or run["warm_start_sha256"] != parent["checkpoint_sha256"]
                or run["view_digest"] != parent["view_digest"]
                or any(run[k] != spec[k] for k in ("arm", "alpha", "exclusive_groups", "steps", "seed"))):
            raise ValueError("Returned checkpoint violates the experiment contract")
        receipt = {"case": case, "status": "trained", "checkpoint_sha256": sha,
                   "protocol_sha256": request["protocol_sha256"], "spec": spec}
        write_json(folder / "receipt.json", receipt)
        print({"case": case, "status": "checkpoint_collected", "training_seconds": run["elapsed_seconds"]}, flush=True)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", help="Optional, preselected subset of the frozen matrix")
    parser.add_argument("--collect-only", action="store_true", help="Compatibility flag; collection never scores locally")
    args = parser.parse_args()
    protocol = read_json(OUT / "protocol.json")
    selected = args.cases or list(protocol["cases"])
    if len(set(selected)) != len(selected) or any(case not in protocol["cases"] for case in selected):
        raise ValueError("Select unique cases from the frozen protocol")
    plan = {"protocol_sha256": file_digest(OUT / "protocol.json"), "cases": sorted(selected)}
    plan_path = OUT / "execution_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Execution subset differs from the saved plan; resolve it before submitting calls")
    write_json(plan_path, plan)
    receipts, failures = {}, {}
    # Collection is independent of scoring: failed evaluation cannot lose paid weights.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(collect, case, protocol["cases"][case], protocol): case for case in selected}
        for future in as_completed(futures):
            case = futures[future]
            try:
                receipts[case] = future.result()
            except Exception as error:  # noqa: BLE001 -- collect every independent paid call before reporting failures
                failures[case] = {"type": type(error).__name__, "message": str(error)}
                print({"case": case, "status": "failed", "error_type": type(error).__name__}, flush=True)
            write_json(OUT / "collection.json", {"receipts": receipts, "failures": failures})
    print({"status": "collection_finished", "trained": len(receipts), "failures": len(failures),
           "next": "Run scripts/run_research_evaluation.py for cloud GPU scoring"}, flush=True)


if __name__ == "__main__":
    main()
