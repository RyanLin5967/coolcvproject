"""Three matched Large/704/EMA controls in the full 13-class chess task."""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[1]
DATASET = "fb5c4fa70b4db2ec71e2bf321c0249f6bff60ad50efd4a782a117e4e0b9cd96b"
SEED = 20260917
remote = modal.Function.from_name("coveragecv-capacity", "experiment")


def execute(arm):
    method = f"{arm}_large_704_ema"
    folder = ROOT / "artifacts/capacity/all-pieces" / str(SEED) / method
    folder.mkdir(parents=True, exist_ok=True)
    request = {"dataset_id": DATASET, "arm": arm, "seed": SEED, "steps": 2000}
    if (folder / "receipt.json").exists():
        result = read_json(folder / "receipt.json")
        if result["checkpoint_sha256"] != file_digest(folder / "detector.pt"):
            raise ValueError("checkpoint receipt mismatch")
        return result
    if (folder / "call.json").exists():
        submitted = read_json(folder / "call.json")
        if submitted["request"] != request:
            raise ValueError("existing call request differs")
        call = modal.FunctionCall.from_id(submitted["call_id"])
    else:
        from coveragecv.training.cloud_budget import reserve_capacity_run
        reservation = reserve_capacity_run()
        call = remote.spawn(**request)
        write_json(folder / "call.json", {"call_id": call.object_id, "request": request,
                                          "reservation_id": reservation})
    returned = call.get()
    with zipfile.ZipFile(io.BytesIO(returned)) as archive:
        for name in archive.namelist():
            safe_child(folder, name).write_bytes(archive.read(name))
    run, evaluation = read_json(folder / "run.json"), read_json(folder / "evaluation.json")
    sha = file_digest(folder / "detector.pt")
    if (run["status"] != "completed" or run["steps"] != 2000 or run["seed"] != SEED
            or run["arm"] != arm or run["recipe"] != "large_fresh" or run["export_weights"] != "ema"
            or run["detector_sha256"] != sha or evaluation["checkpoint_sha256"] != sha):
        raise ValueError("returned experiment binding failed")
    result = {"seed": SEED, "method": method, "status": "completed", "checkpoint_sha256": sha,
              "AP": evaluation["metrics"]["AP"]}
    write_json(folder / "receipt.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(execute, ("naive", "aware", "complete_reference")))
