"""Nine matched, resumable experiments in an independent industrial domain."""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json

ROOT = Path(__file__).resolve().parents[1]
dataset = read_json(ROOT / "artifacts/construction/cloud_input.json")["dataset_id"]
remote = modal.Function.from_name("coveragecv-ablation", "fresh_experiment")


def execute(seed, arm):
    folder = ROOT / "artifacts/gpu/construction" / str(seed) / arm
    folder.mkdir(parents=True, exist_ok=True)
    request = {"dataset_id": dataset, "arm": arm, "seed": seed, "steps": 4000}
    if (folder / "receipt.json").exists():
        receipt = read_json(folder / "receipt.json")
        run = read_json(folder / "run.json")
        if (receipt["checkpoint_sha256"] != file_digest(folder / "detector.pt")
                or run["arm"] != arm or run["seed"] != seed or run["steps"] != 4000):
            raise ValueError("saved construction result does not match its request")
        return {"seed": seed, "arm": arm, "status": "reused"}
    if (folder / "call.json").exists():
        submitted = read_json(folder / "call.json")
        if submitted["request"] != request:
            raise ValueError("existing call uses a different construction protocol")
        call = modal.FunctionCall.from_id(submitted["call_id"])
    else:
        from coveragecv.training.cloud_budget import require_cloud_execution
        require_cloud_execution("coveragecv-ablation")
        call = remote.spawn(**request)
        write_json(folder / "call.json", {"call_id": call.object_id, "request": request})
    result = call.get()
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        for name in archive.namelist():
            safe_child(folder, name).write_bytes(archive.read(name))
    run, evaluation = read_json(folder / "run.json"), read_json(folder / "evaluation.json")
    sha = file_digest(folder / "detector.pt")
    if (run["status"] != "completed" or run["steps"] != 4000 or run["seed"] != seed or run["arm"] != arm
            or run["recipe"] != "augmented_fresh" or run["detector_sha256"] != sha
            or evaluation["checkpoint_sha256"] != sha):
        raise ValueError("returned construction result is not bound to its request")
    receipt = {"seed": seed, "arm": arm, "status": "completed", "checkpoint_sha256": sha,
               "AP": evaluation["metrics"]["AP"]}
    write_json(folder / "receipt.json", receipt)
    return receipt


with ThreadPoolExecutor(max_workers=6) as pool:
    jobs = [pool.submit(execute, seed, arm) for seed in (20260917, 20260918, 20260919)
            for arm in ("naive", "aware", "complete_reference")]
    for job in as_completed(jobs):
        print(job.result(), flush=True)
