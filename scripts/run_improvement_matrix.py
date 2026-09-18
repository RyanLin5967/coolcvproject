"""Attach or submit the declared matrix, waiting for outstanding stage-one results."""
import argparse
import time
from pathlib import Path

from coveragecv.training.modal_improve import run_improvement_batch

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=("pawns", "all-pieces"))
parser.add_argument("--seeds", nargs="+", type=int, default=[20260917, 20260918, 20260919])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / "artifacts/improved" / args.task
deadline = time.monotonic()+7200
while time.monotonic() < deadline:
    run_improvement_batch(args.task, seeds=tuple(args.seeds))
    count = sum(len(list((root / str(seed)).glob("*/receipt.json"))) for seed in args.seeds)
    if count == 5*len(args.seeds):
        print(f"Completed all {count} selected stage-two {args.task} runs", flush=True)
        break
    print("Waiting for missing stage-one checkpoints; existing calls will be reattached, not resubmitted.", flush=True)
    time.sleep(30)
else:
    raise SystemExit("Matrix collection deadline reached. Call IDs persist for reattachment.")
