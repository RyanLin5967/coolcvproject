"""Export a small, verified, read-only workbench snapshot for static hosting.

Run with the local workbench running: uv run --no-sync python scripts/export_public_demo.py
The exporter never starts training or calls a cloud provider. Metrics retain the complete
validation cohort; only the interactive image gallery is sampled. No model weights ship.
"""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image

from coveragecv.artifacts import file_digest, read_json, verify

ARMS = ("naive", "aware", "complete_reference")
DROP_KEYS = {
    "folder", "path", "base_dir", "source", "view", "bundle", "reference", "reference_bundle",
    "checkpoint_path", "evaluation_path", "source_path", "output_dir", "output_root", "pid",
    "cloud_status", "cloud_spend", "cloud_incident", "interruption_counts", "billing", "cost",
    "api_key", "token", "token_secret", "secret", "credentials", "authorization", "hardware",
}
PRIVATE = re.compile(r"(?:/Users/|/home/|/root/|/tmp/|/private/|~\/|[A-Za-z]:\\)")


def clean(value):
    """Defense in depth behind explicit route and object field allowlists."""
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()
                if key.lower() not in DROP_KEYS and not key.lower().endswith(("_path", "_folder"))}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str) and PRIVATE.search(value):
        raise ValueError("A private filesystem path reached the public snapshot")
    return value


def take(value, keys):
    return clean({key: value[key] for key in keys if key in value})


def public_revision(revision):
    result = take(revision, ("id", "project_id", "number", "created", "status", "classes", "diagnostics"))
    result["spec"] = {"classes": revision["classes"]}
    return result


def public_job(job):
    result = take(job, ("id", "project_id", "revision_id", "kind", "status", "created", "started", "finished"))
    result["payload"] = take(job["payload"], ("steps", "seed", "batch", "arms", "recipe", "device"))
    result["live"] = {}
    result["result"] = {"arms": {}}
    for name in ARMS:
        arm = job["result"]["arms"][name]
        result["result"]["arms"][name] = {
            "metrics": clean(arm["metrics"]),
            "run": take(arm["run"], ("arm", "batch", "classes", "detector_sha256", "device",
                                     "elapsed_seconds", "epochs", "initial_parameter_digest",
                                     "initialization_sha256", "recipe", "resolution", "seed", "steps",
                                     "total_training_steps", "model_variant", "view_digest", "status")),
        }
        model_resolution = arm["run"].get("model_config", {}).get("resolution")
        if model_resolution is not None:
            result["result"]["arms"][name]["run"].setdefault("resolution", model_resolution)
    return result


def verify_job(job, examples):
    """Bind published metrics/predictions to actual local checkpoints and reference."""
    reference = Path(job["result"].get("reference_bundle") or job["payload"]["reference"])
    manifest = verify(reference)
    if manifest["digest"] != examples["bundle_digest"]:
        raise ValueError("Gallery reference digest does not match verified bundle")
    initializations, budgets, seeds = set(), set(), set()
    evaluations = {}
    for name in ARMS:
        arm = job["result"]["arms"][name]
        evaluation = read_json(Path(arm["evaluation_path"]))
        sha = file_digest(Path(arm["checkpoint_path"]))
        if sha != arm["run"]["detector_sha256"] or sha != evaluation["checkpoint_sha256"]:
            raise ValueError("Checkpoint contract failed")
        if evaluation["bundle_digest"] != manifest["digest"] or evaluation["split"] != "valid":
            raise ValueError("Evaluation reference contract failed")
        if evaluation["metrics"] != arm["metrics"]:
            raise ValueError("Displayed metrics do not match saved evaluation")
        if evaluation["metrics"]["images"] != len(examples["images"]):
            raise ValueError("Validation cohort count does not match gallery source")
        initializations.add(arm["run"]["initial_parameter_digest"])
        budgets.add((arm["run"]["steps"], arm["run"].get("total_training_steps", arm["run"]["steps"])))
        seeds.add(arm["run"]["seed"])
        evaluations[name] = evaluation
    if len(initializations) != 1 or len(budgets) != 1 or len(seeds) != 1:
        raise ValueError("Selected arms do not form a matched comparison")
    return evaluations


def attribution_text():
    return """# Public demo image attribution

All included dataset images are distributed under the source-declared
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).
No endorsement by the original creators, Roboflow, or the dataset hosts is implied.
The exact image inventory and SHA-256 values are in `data/snapshot.json`.

## Chess Pieces

**Joseph Nelson and Brad Dwyer**, Chess Pieces, RF100 / Roboflow version 1.
[Dataset source](https://universe.roboflow.com/roboflow-100/chess-pieces-mjzgj/dataset/1).
[LibreYOLO mirror](https://huggingface.co/datasets/LibreYOLO/chess-pieces-mjzgj),
revision `17e0d3e7c76bea701ad623b0f7b13bec8859ff80`.
Source license and attribution recorded in the downloaded dataset metadata and
`research/roboflow/build_plan/DATA_PROTOCOL.md`.

CoverageCV converts source annotations to COCO and creates a two-pawn projection
and a separate all-13-class task. Training-source class withholding is deliberate;
it does not imply missing source labels. Exact duplicate boxes were removed in
the documented preparation. The shared validation images remain byte-identical.

## Construction Safety

**Anonymous**, original [Worker Safety project](https://universe.roboflow.com/computer-vision/worker-safety),
republished as [Roboflow100 Construction Safety, version 1](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1).
[LibreYOLO mirror](https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb),
revision `342e545489a6b5f76d6c8225f1ef2629c5a4770a`.
The original creator credit and CC BY 4.0 declaration are preserved in the downloaded
`README.dataset.txt`; `data.yaml` also declares CC BY 4.0.

The upstream export auto-oriented and resized images to 640 × 640. CoverageCV
converts annotations, repairs grouped splits, and deliberately withholds class
families in training. Published validation labels have documented defects and
remain unchanged in the reported metrics. Included image bytes are unchanged.

## Display and selection

Each gallery contains six evenly spaced entries in the existing validation-image
order, including both endpoints. Selection does not use predictions or accuracy.
Chess galleries reuse the same six images. Browser overlays are model predictions
or reference boxes, not edits to image files. Metrics use the entire validation
split; gallery sampling does not change them. No test images are exported.
"""


def export(base_url, output):
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only the local read-only workbench may supply an export")

    def get(path):
        with urlopen(base_url.rstrip("/") + path, timeout=120) as response:
            return response.read()

    def api(path):
        return json.loads(get("/api" + path))

    state = api("/state")
    research = api("/research")
    if len(state["projects"]) != 3 or set(research["tasks"]) != {"pawns", "all-pieces", "construction"}:
        raise ValueError("Expected the three verified demo tasks")
    routes = {"/research": take(research, ("tasks", "metric", "uncertainty", "limitations", "completed_runs",
                                         "submitted_runs", "planned_runs", "acquisition"))}
    if routes["/research"].get("acquisition", {}).get("status") != "completed":
        raise ValueError("Acquisition evidence is incomplete")
    projects, jobs, inventory, image_payloads = [], [], [], {}
    for project in state["projects"]:
        full_project = api("/projects/" + project["id"])
        candidates = [job for job in full_project["jobs"]
                      if job["status"] == "completed"
                      and set((job.get("result") or {}).get("arms", {})) == set(ARMS)
                      and min(arm["run"]["steps"] for arm in job["result"]["arms"].values()) >= 2000]
        if not candidates:
            raise ValueError("No complete, substantial three-arm comparison exists")
        # max() preserves the first candidate on equal steps. No score-based selection.
        chosen = max(candidates, key=lambda job: min(arm["run"].get("total_training_steps", arm["run"]["steps"])
                                                   for arm in job["result"]["arms"].values()))
        revision = public_revision(api("/revisions/" + project["active_revision"]))
        public = take(project, ("id", "name", "created", "active_revision", "revision_count"))
        public["reference_bundle"] = bool(project["reference_bundle"])
        public["latest"] = revision
        job = public_job(chosen)
        projects.append(public)
        jobs.append(job)
        routes["/projects/" + project["id"]] = {**public, "revisions": [revision], "jobs": [job]}
        routes["/revisions/" + revision["id"]] = revision
        routes["/jobs/" + job["id"]] = job
        examples = api("/jobs/" + job["id"] + "/examples")
        evaluations = verify_job(chosen, examples)
        count = len(examples["images"])
        indices = [round(index * (count - 1) / 5) for index in range(6)]
        selected = []
        for index in indices:
            original = examples["images"][index]
            data = get(original["image_url"])
            sha = hashlib.sha256(data).hexdigest()
            if sha != Path(original["file_name"]).stem:
                raise ValueError("API image bytes do not match their content-addressed identity")
            with Image.open(io.BytesIO(data)) as bitmap:
                if bitmap.size != (original["width"], original["height"]) or bitmap.format != "JPEG":
                    raise ValueError("Image format or dimensions violate gallery contract")
            image_name = sha + ".jpg"
            image_payloads[image_name] = data
            selected.append({**take(original, ("id", "width", "height", "annotations")),
                             "file_name": image_name, "image_url": "/images/" + image_name})
            predictions = api(f"/jobs/{job['id']}/predictions/{original['id']}")
            if set(predictions) != set(ARMS):
                raise ValueError("Prediction arm set differs from the selected comparison")
            for name, rows in predictions.items():
                expected = [p for p in evaluations[name]["predictions"] if p["image_id"] == original["id"]]
                if rows != expected:
                    raise ValueError("API predictions differ from checkpoint-bound saved evaluation")
            routes[f"/jobs/{job['id']}/predictions/{original['id']}"] = {
                name: [take(row, ("image_id", "category_id", "bbox", "score")) for row in rows]
                for name, rows in predictions.items()}
            inventory.append({"project_id": project["id"], "job_id": job["id"], "image_id": original["id"],
                              "validation_index": index, "sha256": sha, "bytes": len(data)})
        routes["/jobs/" + job["id"] + "/examples"] = {
            "bundle_digest": examples["bundle_digest"], "classes": examples["classes"], "images": selected,
            "total_validation_images": count, "selection": "Six evenly spaced validation indices; no score selection."}
    routes["/state"] = {"projects": projects, "jobs": jobs, "execution": "public_snapshot", "queue_error": None}
    snapshot = {"schema_version": 1, "read_only": True, "routes": routes, "gallery_inventory": inventory,
                "selection": {"jobs": "Highest total update count complete three-arm comparison; stable first on ties.",
                              "images": "Six evenly spaced indices from each selected validation gallery.",
                              "metrics": "Full validation split; unchanged by gallery selection."},
                "attribution_url": "/ATTRIBUTION.md"}
    encoded = (json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if PRIVATE.search(encoded.decode()):
        raise ValueError("Private filesystem path found in final encoded snapshot")
    total = len(encoded) + sum(map(len, image_payloads.values()))
    if total > 5_000_000:
        raise ValueError(f"Public data payload exceeds 5 MB: {total}")
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)
    for name, data in image_payloads.items():
        (output / "images" / name).write_bytes(data)
    (output / "data/snapshot.json").write_bytes(encoded)
    (output / "ATTRIBUTION.md").write_text(attribution_text())
    print(json.dumps({"projects": len(projects), "jobs": len(jobs), "gallery_entries": len(inventory),
                      "unique_images": len(image_payloads), "snapshot_bytes": len(encoded),
                      "total_bytes": total, "snapshot_sha256": hashlib.sha256(encoded).hexdigest()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("public-demo"))
    args = parser.parse_args()
    export(args.base_url, args.output)
