"""Train/collect first, then evaluate locally; an evaluation failure never loses the model."""
import io
import os
import zipfile
from pathlib import Path

import modal

from coveragecv.artifacts import file_digest, read_json, safe_child, write_json
from coveragecv.training.cloud_budget import reserve_refinement_run

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "artifacts/refinement/all-pieces/20260917/partial-human"
folder.mkdir(parents=True, exist_ok=True)
parents = []
for method in ("naive_augmented_512", "aware_augmented_512", "complete_augmented_512"):
    path = ROOT / "artifacts/improved/all-pieces/20260917" / method
    parents.append({"method": method, "call_id": read_json(path / "call.json")["call_id"],
                    "checkpoint_sha256": read_json(path / "receipt.json")["checkpoint_sha256"]})
request = {"dataset_id": "fb5c4fa70b4db2ec71e2bf321c0249f6bff60ad50efd4a782a117e4e0b9cd96b", "parents": parents}
if (folder / "receipt.json").exists():
    receipt = read_json(folder / "receipt.json")
    if receipt["checkpoint_sha256"] != file_digest(folder / "refiner.pt"):
        raise ValueError("saved refiner receipt mismatch")
    print("Verified existing trained refiner; no new cloud call", flush=True)
else:
    if (folder / "call.json").exists():
        saved = read_json(folder / "call.json")
        if saved["request"] != request:
            raise ValueError("existing refinement request differs")
        call = modal.FunctionCall.from_id(saved["call_id"])
    else:
        reservation = reserve_refinement_run()
        remote = modal.Function.from_name("coveragecv-refinement", "experiment")
        call = remote.spawn(**request)
        write_json(folder / "call.json", {"call_id": call.object_id, "request": request,
                                          "reservation_id": reservation})
    returned = call.get()
    with zipfile.ZipFile(io.BytesIO(returned)) as archive:
        for name in archive.namelist():
            safe_child(folder, name).write_bytes(archive.read(name))
    run = read_json(folder / "refiner_run.json")
    sha = file_digest(folder / "refiner.pt")
    expected_view = read_json(Path(read_json(ROOT / "artifacts/full_chess/experiment.json")["partial_view"])
                              / "manifest.json")["digest"]
    if (run["checkpoint_sha256"] != sha or run["status"] != "completed" or run["steps"] != 1000
            or run["seed"] != 20260917 or run["training_view_digest"] != expected_view):
        raise ValueError("returned refiner did not complete its fixed training contract")
    write_json(folder / "receipt.json", {"checkpoint_sha256": sha, "status": "trained",
                                        "evaluation_status": "pending"})
    print("Trained checkpoint collected and verified; starting local evaluation", flush=True)

import torch

from coveragecv.training.refiner import evaluate_refiner

torch.set_num_threads(4)
# ResNet's 7x7 -> 4x4 adaptive spatial pooling is unsupported on MPS.
# CPU preserves the exact trained architecture without a device-specific rewrite.
device = os.environ.get("COVERAGECV_REFINER_DEVICE", "cpu")
bundle = Path(read_json(ROOT / "artifacts/full_chess/experiment.json")["complete_bundle"])
results = {}
for parent in parents:
    method = parent["method"]
    baseline = ROOT / "artifacts/improved/all-pieces/20260917" / method
    target = folder / f"{method}.json"
    if target.exists():
        result = read_json(target)
        if (result["refiner_checkpoint_sha256"] != file_digest(folder / "refiner.pt")
                or result["checkpoint_sha256"] != parent["checkpoint_sha256"]
                or result["baseline_evaluation_sha256"] != file_digest(baseline / "evaluation.json")):
            raise ValueError("saved refinement evaluation binding mismatch")
    else:
        result = evaluate_refiner(folder / "refiner.pt", baseline / "evaluation.json", baseline / "detector.pt",
                                   bundle, target, device=device)
    results[method] = {k: v for k, v in result.items() if k != "predictions"}
    print({"method": method, "baseline_AP": result["baseline_metrics"]["AP"],
           "refined_AP": result["metrics"]["AP"]}, flush=True)
write_json(folder / "comparison.json", {"status": "completed", "results": results,
    "same_refiner_for_all_arms": True, "training_labels": "partial observed human boxes",
    "refiner_sha256": file_digest(folder / "refiner.pt"), "evaluation_device": device})
write_json(folder / "receipt.json", {"checkpoint_sha256": file_digest(folder / "refiner.pt"),
                                    "status": "completed", "evaluation_status": "completed"})
