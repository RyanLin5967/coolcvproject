"""Collect predeclared three-seed fusion controls; no new training or cloud compute."""
from pathlib import Path

from coveragecv.artifacts import read_json, write_json
from coveragecv.training.ensemble import evaluate_ensemble

ROOT = Path(__file__).resolve().parents[1]
protocol = read_json(ROOT / "artifacts/ensemble_protocol.json")
results = []
for task, spec in (("pawns", "chess_experiment.json"), ("all-pieces", "full_chess/experiment.json")):
    bundle = Path(read_json(ROOT / "artifacts" / spec)["complete_bundle"])
    for method in protocol["families"]:
        folders = [ROOT / "artifacts/improved" / task / str(seed) / method for seed in protocol["seeds"]]
        if not all((folder / "receipt.json").exists() for folder in folders):
            continue
        output = ROOT / "artifacts/ensembles" / task / method / "evaluation.json"
        result = evaluate_ensemble(folders, bundle, output)
        results.append({"task": task, "method": method, **{k: v for k, v in result.items() if k != "predictions"}})
        print({"task": task, "method": method, "AP": result["metrics"]["AP"]}, flush=True)
write_json(ROOT / "artifacts/ensemble_results.json", {"protocol": protocol, "runs": results})
