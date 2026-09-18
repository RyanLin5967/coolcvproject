"""Collect hash-verified GPU scores; never run inference on the user's computer."""
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, write_json
from coveragecv.training.cloud_budget import reserve_research_evaluation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/research_v2"


def collect(case, task, checkpoint, destination):
    evaluation = read_json(OUT / "evaluation_protocol.json")
    request = {"task": task, "checkpoint_sha256": file_digest(checkpoint),
               "protocol_sha256": file_digest(OUT / "evaluation_protocol.json")}
    folder = OUT / "evaluation_calls" / case
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            result = read_json(destination)
        else:
            if (folder / "call.json").exists():
                saved = read_json(folder / "call.json")
                if saved["request"] != request:
                    raise ValueError("Saved evaluation call belongs to a different checkpoint or reference")
                call = modal.FunctionCall.from_id(saved["call_id"])
            else:
                if (folder / "reservation.json").exists():
                    raise ValueError("Orphan evaluation reservation; never automatically retry a paid call")
                reservation = reserve_research_evaluation()
                write_json(folder / "reservation.json", {"request": request, "reservation_id": reservation})
                remote = modal.Function.from_name("coveragecv-research-evaluation", "evaluate")
                call = remote.spawn(**request, checkpoint=checkpoint.read_bytes())
                write_json(folder / "call.json", {"request": request, "call_id": call.object_id})
                print({"case": case, "status": "evaluation_submitted", "call_id": call.object_id}, flush=True)
            result = call.get()
        if (result["checkpoint_sha256"] != request["checkpoint_sha256"]
                or result["evaluation_protocol_sha256"] != request["protocol_sha256"]
                or result["bundle_digest"] != evaluation["references"][task]["bundle_digest"]
                or result["split"] != "valid" or result["device"] != "cuda"):
            raise ValueError("Evaluation violated its checkpoint, reference, split, or device contract")
        write_json(destination, result)
        print({"case": case, "AP": result["metrics"]["AP"], "status": "scored"}, flush=True)
        return {"metrics": result["metrics"], "checkpoint_sha256": result["checkpoint_sha256"],
                "evaluation_sha256": file_digest(destination), "task": task}


def main():
    protocol = read_json(OUT / "protocol.json")
    jobs = []
    for task, data in protocol["datasets"].items():
        for arm, parent in data["parents"].items():
            jobs.append((f"{task}-parent-{arm}", task, Path(parent["local_parent"]) / "detector.pt",
                         OUT / "parent_evaluations" / f"{task}-{arm}.json"))
    for case, spec in protocol["cases"].items():
        folder = OUT / "runs" / case
        if (folder / "receipt.json").exists():
            receipt = read_json(folder / "receipt.json")
            if (receipt["protocol_sha256"] != file_digest(OUT / "protocol.json")
                    or receipt["checkpoint_sha256"] != file_digest(folder / "detector.pt")
                    or receipt["spec"] != spec):
                raise ValueError("Saved training receipt changed before scoring")
            jobs.append((case, spec["task"], folder / "detector.pt", folder / "evaluation.json"))
    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(collect, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case] = future.result()
            except Exception as error:  # noqa: BLE001 -- preserve every independent scoring receipt
                failures[case] = {"type": type(error).__name__, "message": str(error)}
                print({"case": case, "status": "failed", "type": type(error).__name__}, flush=True)
            write_json(OUT / "gpu_results.json", {"status": "scoring", "results": results, "failures": failures})
    expected = len(protocol["cases"])+2*len(protocol["datasets"])
    status = "completed" if len(results) == expected and not failures else "partial"
    write_json(OUT / "gpu_results.json", {"status": status, "results": results, "failures": failures})


if __name__ == "__main__":
    main()
