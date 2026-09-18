"""Predeclared final test evaluation, after all nine validation runs are collected.

The complete set of recipes/seeds is fixed: no choosing a winner using these labels.
Runs locally without provisioning any cloud resources.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coveragecv.artifacts import file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/gpu/construction"
jobs = [(seed, arm) for seed in (20260917, 20260918, 20260919)
        for arm in ("naive", "aware", "complete_reference")]
protocol = {"recipes": ["augmented_fresh"], "steps": 4000, "seeds": [20260917, 20260918, 20260919],
            "arms": ["naive", "aware", "complete_reference"], "threshold": .25,
            "split": "test", "purpose": "Evaluate every predeclared arm/seed once; no test-guided recipe selection."}
write_json(ROOT / "artifacts/construction/holdout_protocol.json", protocol)
deadline = time.monotonic()+7200
while not all((BASE / str(seed) / arm / "receipt.json").exists() for seed, arm in jobs):
    if time.monotonic() > deadline:
        raise SystemExit("Waiting for completed validation runs; held-out labels have not been evaluated.")
    time.sleep(15)


def evaluate(job):
    # Separate subprocesses keep Torch/Lightning global state independent.
    import subprocess
    import sys
    seed, arm = job
    folder = BASE / str(seed) / arm
    checkpoint = folder / "detector.pt"
    run = read_json(folder / "run.json")
    if run["recipe"] != "augmented_fresh" or run["steps"] != 4000 or run["status"] != "completed":
        raise ValueError("unapproved holdout recipe")
    target = folder / "test_evaluation.json"
    bundle = Path(read_json(ROOT / "artifacts/construction/experiment.json")["complete_bundle"])
    if not target.exists():
        with (folder / "test_evaluation.log").open("wb") as log:
            subprocess.run([sys.executable, "-m", "coveragecv.cli", "evaluate", str(checkpoint), str(bundle),
                            str(target), "--split", "test"], cwd=ROOT, stdout=log, stderr=log, check=True)
    result = read_json(target)
    if result["checkpoint_sha256"] != file_digest(checkpoint) or result["split"] != "test":
        raise ValueError("holdout result is not bound to its checkpoint/split")
    print({"seed": seed, "arm": arm, "test_AP": result["metrics"]["AP"]}, flush=True)
    return {"seed": seed, "arm": arm, "metrics": result["metrics"],
            "checkpoint_sha256": result["checkpoint_sha256"], "bundle_digest": result["bundle_digest"]}


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(evaluate, jobs))
write_json(ROOT / "artifacts/construction/holdout_results.json", {"protocol": protocol, "runs": results})
