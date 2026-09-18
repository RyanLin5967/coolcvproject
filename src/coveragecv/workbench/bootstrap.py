"""Adopt the existing verified pilot; never silently launch a training experiment."""
from pathlib import Path

from coveragecv.artifacts import read_json

from .service import import_saved_experiment, import_spec
from .worker import execute


def bootstrap_demo(store):
    workspace = Path(__file__).resolve().parents[3]
    path = workspace / "artifacts/chess_experiment.json"
    if not path.exists():
        return
    experiment = read_json(path)
    reference = Path(experiment["complete_bundle"])
    project = import_spec(store, workspace / "data/chess/grouped/partial-spec.json",
                           "Chess · partial annotations", reference=reference)
    revision = store.get("revisions", project["active_revision"])
    job = store.jobs(project["id"])[0]
    output = store.root / "jobs" / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    result = execute(job, store.root, output)
    store.finish(job["id"], {"status": "completed", "result": result})
    for folder in (workspace / "artifacts/mvp", workspace / "artifacts/extended/500-seed20260917"):
        if all((folder / arm / "evaluation.json").exists() for arm in ("naive", "aware", "complete_reference")):
            import_saved_experiment(store, project["id"], revision["id"], folder, reference)
