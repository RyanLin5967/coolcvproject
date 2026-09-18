"""Refresh verified benchmark summaries and adopt completed stage-one comparisons."""
import time
from pathlib import Path

from coveragecv.artifacts import read_json
from coveragecv.benchmarks import build_summary
from coveragecv.workbench.service import import_saved_experiment
from coveragecv.workbench.store import Store

workspace = Path(__file__).resolve().parents[1]
store = Store(workspace / "artifacts/workbench")
deadline = time.monotonic()+7200
while time.monotonic() < deadline:
    summary = build_summary(workspace)
    for task, experiment in (("pawns", "chess_experiment.json"), ("all-pieces", "full_chess/experiment.json"),
                             ("construction", "construction/experiment.json")):
        ex = read_json(workspace / "artifacts" / experiment)
        project = next((p for p in store.projects() if p["reference_bundle"] == ex["complete_bundle"]), None)
        if project is None:
            continue
        digest = read_json(Path(ex["partial_view"]) / "manifest.json")["digest"]
        revision = next((r for r in store.revisions(project["id"]) if r["status"] == "ready"
                         and read_json(Path(r["view"]) / "manifest.json")["digest"] == digest), None)
        if revision is None:
            continue
        saved = {job["payload"].get("source") for job in store.jobs(project["id"])}
        for seed in (20260917, 20260918, 20260919):
            folder = workspace / "artifacts/gpu" / task / str(seed)
            if str(folder) in saved or not all((folder / arm / "receipt.json").exists()
                                               for arm in ("naive", "aware", "complete_reference")):
                continue
            job = import_saved_experiment(store, project["id"], revision["id"], folder, Path(ex["complete_bundle"]))
            print(f"Adopted {task} seed {seed}: {job['id']}", flush=True)
    print(f"Verified {summary['completed_runs']}/{summary['planned_runs']} planned runs; "
          f"{summary['submitted_runs']} submitted; {len(summary['issues'])} issues", flush=True)
    if summary["completed_runs"] == summary["planned_runs"]:
        break
    time.sleep(15)
