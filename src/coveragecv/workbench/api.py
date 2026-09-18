"""Loopback-only application API. Every mutation requires the workbench client header."""
import asyncio
import contextlib
import json
import time
import zipfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from coveragecv.artifacts import digest, file_digest, read_json, safe_child, verify
from coveragecv.compiler import _jsonl
from coveragecv.schema import CoverageState, DiagnosticError

from . import service
from .scheduler import Scheduler
from .store import Store, identifier
from .worker import local_hardware, validate_local_device


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImportRequest(Input):
    name: str = Field(min_length=1, max_length=100)
    spec_path: str


class PolicyCell(Input):
    source: str
    class_name: str
    state: CoverageState


class PolicyRequest(Input):
    base_revision: str
    changes: list[PolicyCell] = Field(min_length=1, max_length=10000)


class TrainRequest(Input):
    steps: int = Field(default=100, ge=1, le=5000)
    seed: int = Field(default=20260917, ge=0, le=2**31-1)
    batch: int = Field(default=2, ge=1, le=8)
    timeout_seconds: int = Field(default=1800, ge=30, le=7200)
    arms: list[Literal["naive", "aware", "complete_reference"]] = Field(default=["naive", "aware"], min_length=1, max_length=3)
    recipe: Literal["pilot", "augmented_fresh", "large_fresh"] = "pilot"
    device: Literal["cpu", "mps"] = "cpu"


def refinement_evidence(artifacts: Path):
    """Publish the separate refiner pilot only after its matched evaluations finish."""
    folder = artifacts / "refinement/all-pieces/20260917/partial-human"
    methods = ("naive_augmented_512", "aware_augmented_512", "complete_augmented_512")
    evidence = {"task": "all-pieces", "seed": 20260917, "n": 1, "status": "not_started",
                "results": {}, "prior_failed_attempts": len(list(folder.glob("attempts/*/failure.json")))}
    if (folder / "call.json").exists():
        evidence["status"] = "submitted"
    receipt = read_json(folder / "receipt.json") if (folder / "receipt.json").exists() else {}
    if receipt.get("status") in ("trained", "completed"):
        evidence.update(status="evaluating", checkpoint_collected=True,
                        refiner_sha256=receipt.get("checkpoint_sha256"))
    comparison = folder / "comparison.json"
    if not comparison.exists():
        return evidence
    data = read_json(comparison)
    rows = data.get("results", {})
    sha = receipt.get("checkpoint_sha256")
    if (receipt.get("status") != "completed" or receipt.get("evaluation_status") != "completed"
            or data.get("status") != "completed"
            or not sha or data.get("refiner_sha256") != sha or set(rows) != set(methods)
            or data.get("same_refiner_for_all_arms") is not True
            or any(row.get("refiner_checkpoint_sha256") != sha
                   or row.get("classification_and_scores_changed") is not False
                   or row.get("train_evaluation_exact_image_overlap") is not False
                   for row in rows.values())):
        evidence.update(status="evidence_incomplete", issue="Refiner completion or provenance checks are incomplete.")
        return evidence
    evidence.update(status="completed", same_refiner_for_all_arms=True,
                    training_labels=data.get("training_labels"), evaluation_device=data.get("evaluation_device"),
                    results={method: {key: value for key, value in rows[method].items() if key != "predictions"}
                             for method in methods})
    return evidence


def object_crop_evidence(artifacts: Path):
    """Read the fixed construction study without mixing it into detector seed means."""
    study = artifacts / "object-crops"
    protocol_path = study / "protocol.json"
    if not protocol_path.exists():
        return None
    evidence = {"task": "construction", "status": "in_progress", "n": 1, "cases": {}}
    cases = {"aware_standard": ("aware", False), "aware_object_crops": ("aware", True),
             "naive_object_crops": ("naive", True), "complete_reference_object_crops": ("complete_reference", True)}
    try:
        protocol, plan, audit = (read_json(study / name) for name in ("protocol.json", "plan.json", "plan_audit.json"))
        protocol_sha = file_digest(protocol_path)
        plan_sha = digest({key: value for key, value in plan.items() if key != "digest"})
        if (set(protocol["cases"]) != set(cases) or plan["digest"] != plan_sha
                or protocol["crop_plan_digest"] != plan_sha or audit["plan_digest"] != plan_sha
                or plan["source_view_digest"] != protocol["partial_view_digest"]
                or plan["train_only"] is not True or plan["validation_or_test_labels_used"] is not False):
            raise ValueError("study plan identity mismatch")
        no_helmet = str(plan["classes"].index("no-helmet") + 1)
        evidence.update(seed=protocol["seed"], steps=protocol["steps"], total_steps=protocol["total_steps"],
                        resolution=protocol["resolution"], crop_plan_digest=plan_sha, protocol_sha256=protocol_sha,
                        samples=plan["samples"], full_frame_samples=audit["full_frame_samples"],
                        no_helmet_anchor_samples=audit["anchor_samples_by_class"][no_helmet],
                        no_helmet_original_boxes=plan["anchor_counts"][no_helmet],
                        no_helmet_median_max_side=audit["median_anchor_max_side_pixels_by_class"][no_helmet],
                        validation_during_training=protocol["validation_during_training"],
                        test_evaluated=protocol["test_evaluated"])
    except (OSError, ValueError, KeyError, TypeError):
        return {**evidence, "status": "evidence_incomplete", "issue": "Study protocol or crop plan verification failed."}
    output = study / "construction" / str(protocol["seed"])
    for case, (arm, crops) in cases.items():
        folder = output / case
        row = {"arm": arm, "uses_object_crops": crops, "status": "not_started"}
        evidence["cases"][case] = row
        try:
            if not (folder / "call.json").exists():
                continue
            request = {"case": case, "protocol_sha256": protocol_sha}
            if read_json(folder / "call.json")["request"] != request:
                raise ValueError("cloud request identity mismatch")
            row["status"] = "submitted"
            if not (folder / "receipt.json").exists():
                continue
            receipt, run = read_json(folder / "receipt.json"), read_json(folder / "run.json")
            sha = file_digest(folder / "detector.pt")
            expected_view = protocol["complete_view_digest" if arm == "complete_reference" else "partial_view_digest"]
            parent_sha = protocol["parents"][arm]["checkpoint_sha256"]
            if (receipt["checkpoint_sha256"] != sha or run["detector_sha256"] != sha
                    or run["status"] != "completed" or run["arm"] != arm or run["method"] != case
                    or run["seed"] != protocol["seed"] or run["steps"] != protocol["steps"]
                    or run["batch"] != protocol["batch"] or run["recipe"] != protocol["recipe"]
                    or run["model_config"]["resolution"] != protocol["resolution"]
                    or run["total_training_steps"] != protocol["total_steps"] or run["view_digest"] != expected_view
                    or run["parent_checkpoint_sha256"] != parent_sha or run["warm_start_sha256"] != parent_sha
                    or run["experiment_protocol_sha256"] != protocol_sha
                    or run["object_crop_plan_digest"] != (plan_sha if crops else None)
                    or run["validation_during_training"] is not False):
                raise ValueError("trained checkpoint contract mismatch")
            row.update(status="evaluating", checkpoint_sha256=sha, parent_checkpoint_sha256=parent_sha)
            if receipt.get("status") != "completed" or receipt.get("evaluation_status") != "completed":
                continue
            evaluation = read_json(folder / "evaluation.json")
            parent = read_json(artifacts / "gpu/construction" / str(protocol["seed"]) / arm / "evaluation.json")
            if (evaluation["checkpoint_sha256"] != sha or parent["checkpoint_sha256"] != parent_sha
                    or any(item["split"] != "valid" or item["bundle_digest"] != protocol["reference_bundle_digest"]
                           for item in (evaluation, parent))):
                raise ValueError("evaluation reference binding mismatch")
            row.update(status="completed", metrics=evaluation["metrics"], parent_metrics=parent["metrics"])
        except (OSError, ValueError, KeyError, TypeError):
            row.update(status="evidence_incomplete", issue="Request, checkpoint, plan or evaluation verification failed.")
            row.pop("metrics", None)
            row.pop("parent_metrics", None)
    comparison_path = output / "comparison.json"
    if all(row["status"] == "completed" for row in evidence["cases"].values()) and comparison_path.exists():
        try:
            comparison = read_json(comparison_path)
            rows = comparison["results"]
            if (comparison["status"] != "completed" or set(rows) != set(cases)
                    or comparison["protocol_sha256"] != protocol_sha or comparison["crop_plan_digest"] != plan_sha
                    or comparison["seed"] != protocol["seed"]
                    or any(rows[case] != {key: row[key] for key in ("metrics", "parent_metrics", "checkpoint_sha256",
                                                                   "parent_checkpoint_sha256")}
                           for case, row in evidence["cases"].items())):
                raise ValueError("comparison binding mismatch")
            delta = 100 * (rows["aware_object_crops"]["metrics"]["AP"] - rows["aware_standard"]["metrics"]["AP"])
            if abs(delta - comparison["primary_AP_delta_points"]) > 1e-9:
                raise ValueError("comparison arithmetic mismatch")
            evidence.update(status="completed", primary_AP_delta_points=delta)
        except (OSError, ValueError, KeyError, TypeError):
            evidence.update(status="evidence_incomplete", issue="The final matched comparison could not be verified.")
    return evidence


def acquisition_evidence(artifacts: Path):
    """Expose a separate annotation-reveal simulation, with its actual review budget."""
    study = artifacts / "acquisition"
    protocol_path = study / "protocol.json"
    if not protocol_path.exists():
        return None
    evidence = {"task": "construction", "status": "in_progress", "n": 1, "simulation": True, "cases": {}}
    cases = {"guided": "aware", "random": "aware", "zero_review": "aware", "complete_standard": "complete_reference"}
    try:
        protocol, frozen = read_json(protocol_path), read_json(study / "selection_frozen.json")
        protocol_sha = file_digest(protocol_path)
        declaration = frozen["declaration"]
        if (protocol["kind"] != "fixed_budget_coverage_acquisition_simulation"
                or file_digest(study / "selection_frozen.json") != protocol["selection_frozen_file_sha256"]
                or digest(declaration) != frozen["declaration_digest"]
                or declaration["reference_labels_read_for_selection"] is not False
                or declaration["test_labels_used"] is not False or protocol["validation_during_training"] is not False):
            raise ValueError("selection freeze binding mismatch")
        yields = {}
        for case in ("guided", "random"):
            acquisition = protocol["views"][case]["acquisition"]
            per_class = acquisition["per_class"]
            if (acquisition["simulation"] is not True or acquisition["actual_new_human_annotation"] is not False
                    or acquisition["same_label_budget_as_original_experiment"] is not False
                    or acquisition["validation_or_test_labels_used"] is not False
                    or acquisition["unrequested_reference_labels_used"] is not False
                    or acquisition["plan_digest"] != declaration["plans"][case]
                    or acquisition["partial_view_digest"] != declaration["partial_view_digest"]
                    or acquisition["review_units"] != protocol["review_units_per_acquisition_arm"]
                    or sum(row["review_units"] for row in per_class) != acquisition["review_units"]
                    or sum(row["added_boxes"] for row in per_class) != acquisition["added_boxes"]
                    or acquisition["boxes_after"] - acquisition["boxes_before"] != acquisition["added_boxes"]):
                raise ValueError("acquisition counts or simulation provenance mismatch")
            yields[case] = {key: acquisition[key] for key in
                           ("review_units", "added_boxes", "boxes_before", "boxes_after", "per_class")}
        evidence.update(seed=protocol["seed"], steps=protocol["steps"], total_steps=protocol["total_training_steps"],
                        review_units_per_arm=protocol["review_units_per_acquisition_arm"],
                        resolution=protocol["resolution"], protocol_sha256=protocol_sha,
                        selection_frozen_at=frozen["frozen_at"], limitations=protocol["limitations"])
    except (OSError, ValueError, KeyError, TypeError):
        return {**evidence, "status": "evidence_incomplete", "issue": "Acquisition protocol, selection freeze or review counts failed verification."}
    output = study / "construction" / str(protocol["seed"])
    verified = {}
    for case, arm in cases.items():
        row = {"status": "not_started", "acquisition": yields.get(case)}
        evidence["cases"][case] = row
        try:
            zero_review = case == "zero_review"
            folder = (safe_child(artifacts.parent, protocol["zero_review_control"]["folder"])
                      if zero_review else output / case)
            acquisition = None if zero_review else protocol["views"][case]["acquisition"]
            if not zero_review:
                if not (folder / "call.json").exists():
                    continue
                if read_json(folder / "call.json")["request"] != {"case": case, "protocol_sha256": protocol_sha}:
                    raise ValueError("request binding mismatch")
                row["status"] = "submitted"
            if not (folder / "receipt.json").exists():
                continue
            receipt, run = read_json(folder / "receipt.json"), read_json(folder / "run.json")
            sha = file_digest(folder / "detector.pt")
            parent = protocol["parents"][arm]["checkpoint_sha256"]
            expected_view = declaration["partial_view_digest"] if zero_review else protocol["views"][case]["digest"]
            if (receipt["checkpoint_sha256"] != sha or run["detector_sha256"] != sha
                    or run["status"] != "completed" or run["arm"] != arm
                    or run["method"] != ("aware_standard" if zero_review else case)
                    or run["seed"] != protocol["seed"] or run["steps"] != protocol["steps"]
                    or run["total_training_steps"] != protocol["total_training_steps"]
                    or run["batch"] != protocol["batch"] or run["recipe"] != protocol["recipe"]
                    or run["model_config"]["resolution"] != protocol["resolution"]
                    or run["view_digest"] != expected_view or run["parent_checkpoint_sha256"] != parent
                    or run["warm_start_sha256"] != parent or run["validation_during_training"] is not False):
                raise ValueError("checkpoint training contract mismatch")
            if zero_review:
                if sha != protocol["zero_review_control"]["checkpoint_sha256"] or run.get("object_crop_plan_digest") is not None:
                    raise ValueError("zero-review control differs")
            elif run["experiment_protocol_sha256"] != protocol_sha or run["annotation_acquisition"] != acquisition:
                raise ValueError("acquired training view differs")
            row.update(status="evaluating", checkpoint_sha256=sha)
            if receipt.get("status") != "completed" or receipt.get("evaluation_status") != "completed":
                continue
            evaluation_path = folder / "evaluation.json"
            evaluation = read_json(evaluation_path)
            if (evaluation["checkpoint_sha256"] != sha or evaluation["split"] != "valid"
                    or evaluation["bundle_digest"] != protocol["reference_bundle_digest"]
                    or evaluation["device"] != "cpu" or evaluation["resolution"] != protocol["resolution"]
                    or evaluation["score_threshold"] != 0.25
                    or evaluation["postprocess"] != "stock RF-DETR; reserved output omitted from semantic COCO mapping"):
                raise ValueError("evaluation binding mismatch")
            row.update(status="completed", metrics=evaluation["metrics"])
            verified[case] = {"metrics": evaluation["metrics"], "checkpoint_sha256": sha,
                              "evaluation_sha256": file_digest(evaluation_path), "acquisition": acquisition}
        except (OSError, ValueError, KeyError, TypeError):
            row.update(status="evidence_incomplete", issue="Acquisition request, checkpoint or reference verification failed.")
            row.pop("metrics", None)
    comparison_path = output / "comparison.json"
    if len(verified) == len(cases) and comparison_path.exists():
        try:
            comparison = read_json(comparison_path)
            if (comparison["status"] != "completed" or comparison["seed"] != protocol["seed"]
                    or comparison["protocol_sha256"] != protocol_sha or comparison["results"] != verified
                    or comparison["review_units_per_acquisition_arm"] != protocol["review_units_per_acquisition_arm"]):
                raise ValueError("comparison identity mismatch")
            scores = {case: row["metrics"]["AP"] * 100 for case, row in verified.items()}
            deltas = {"primary_AP_delta_points": scores["guided"] - scores["random"],
                      "guided_vs_zero_review_AP_delta_points": scores["guided"] - scores["zero_review"],
                      "random_vs_zero_review_AP_delta_points": scores["random"] - scores["zero_review"]}
            if any(abs(value - comparison[key]) > 1e-9 for key, value in deltas.items()):
                raise ValueError("comparison arithmetic mismatch")
            evidence.update(status="completed", **deltas)
        except (OSError, ValueError, KeyError, TypeError):
            evidence.update(status="evidence_incomplete", issue="The final acquisition comparison failed verification.")
    return evidence


def create_app(root: Path = Path("artifacts/workbench"), *, run_scheduler=True):
    store = Store(root)
    scheduler = Scheduler(store)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if run_scheduler:
            scheduler.start()
        yield
        if run_scheduler:
            scheduler.close()

    app = FastAPI(title="CoverageCV Workbench", version="0.2.0", lifespan=lifespan)
    app.state.store, app.state.scheduler = store, scheduler

    @app.middleware("http")
    async def local_client(request: Request, call_next):
        host = request.url.hostname
        if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
            return JSONResponse({"detail": "The workbench is available on localhost only"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        if request.method in ("POST", "PATCH", "DELETE", "PUT") and request.headers.get("x-coveragecv") != "workbench":
            return JSONResponse({"detail": "Missing workbench client header"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(DiagnosticError)
    async def diagnostic_error(request, exc):
        return JSONResponse(status_code=422, content=exc.as_dict())

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def required(table, identity):
        result = store.get(table, identity)
        if result is None:
            raise HTTPException(404, "Not found")
        return result

    def ready_revision(identity):
        revision = required("revisions", identity)
        if revision["status"] != "ready":
            raise HTTPException(409, "This revision has not compiled successfully")
        return revision

    def live_job(job):
        output = store.root / "jobs" / job["id"]
        live = read_json(output / "live.json") if (output / "live.json").exists() else {}
        if live.get("arm"):
            path = output / live["arm"] / "live.json"
            if path.exists():
                live.update(read_json(path))
        result = job.get("result")
        if result is None and (output / "partial_result.json").exists():
            result = read_json(output / "partial_result.json")
        return {**job, "live": live, "result": result}

    @app.get("/api/state")
    def state():
        projects = store.projects()
        for project in projects:
            revisions = store.revisions(project["id"])
            project["revision_count"] = len(revisions)
            project["latest"] = revisions[0] if revisions else None
        error_path = store.root / "scheduler_error.json"
        return {"projects": projects, "jobs": [live_job(j) for j in store.jobs()[:30]],
                "execution": "local", "hardware": local_hardware(),
                "cloud_spend": "No cloud workloads launched by this workbench",
                "queue_error": read_json(error_path) if error_path.exists() else None}

    @app.get("/api/research")
    def research():
        path = Path(__file__).resolve().parents[3] / "artifacts/research_summary.json"
        if not path.exists():
            return {"tasks": {}, "completed_runs": 0, "submitted_runs": 0, "planned_runs": 0,
                    "limitations": ["No verified cloud benchmark has been collected in this workspace."]}
        result = read_json(path)
        diagnosis = path.parent / "error_analysis.json"
        result["diagnostics"] = read_json(diagnosis).get("runs", []) if diagnosis.exists() else []
        holdout = path.parent / "construction/holdout_results.json"
        result["construction_holdout"] = read_json(holdout) if holdout.exists() else None
        ensembles = path.parent / "ensemble_results.json"
        result["ensembles"] = read_json(ensembles).get("runs", []) if ensembles.exists() else []
        incident = path.parent / "cloud_incident.json"
        result["cloud_incident"] = read_json(incident) if incident.exists() else None
        result["refinement"] = refinement_evidence(path.parent)
        result["object_crops"] = object_crop_evidence(path.parent)
        result["acquisition"] = acquisition_evidence(path.parent)
        if result["cloud_incident"]:
            project_root = path.parent.parent
            completed = {str(Path(run["folder"]).relative_to(project_root))
                         for task in result["tasks"].values() for run in task["runs"]
                         if Path(run["folder"]).is_relative_to(project_root)}
            cancelled = result["cloud_incident"].get("cancelled", [])
            restarted = sum(row["path"] in completed for row in cancelled)
            result["interruption_counts"] = {"historical": len(cancelled), "subsequently_completed": restarted,
                                              "remaining": len(cancelled) - restarted}
        return result

    @app.get("/api/projects/{project_id}")
    def project(project_id: str):
        return {**required("projects", project_id), "revisions": store.revisions(project_id),
                "jobs": [live_job(j) for j in store.jobs(project_id)]}

    @app.post("/api/projects/import")
    def import_local(body: ImportRequest):
        return service.import_spec(store, Path(body.spec_path), body.name)

    @app.post("/api/projects/upload")
    async def upload_dataset(file: UploadFile, name: str = "Imported dataset"):
        if not name.strip() or len(name) > 100:
            raise HTTPException(422, "Choose a project name of 1–100 characters")
        imports = store.root / "imports"
        imports.mkdir(exist_ok=True)
        identity = identifier()
        archive, target = imports / f"{identity}.zip", imports / identity
        size = 0
        try:
            with archive.open("wb") as stream:
                while chunk := await file.read(1024*1024):
                    size += len(chunk)
                    if size > 100*1024*1024:
                        raise HTTPException(413, "The compressed ZIP limit is 100 MB")
                    stream.write(chunk)
            spec = service.extract_dataset(archive, target)
            return service.import_spec(store, spec, name, import_root=target)
        finally:
            archive.unlink(missing_ok=True)
            await file.close()

    @app.post("/api/projects/demo")
    def import_demo():
        workspace = Path(__file__).resolve().parents[3]
        experiment_file = workspace / "artifacts/chess_experiment.json"
        if not experiment_file.exists():
            raise HTTPException(409, "Run coveragecv demo first to prepare the public dataset")
        experiment = read_json(experiment_file)
        return service.import_spec(store, workspace / "data/chess/grouped/partial-spec.json", "Chess · partial annotations",
                                    reference=Path(experiment["complete_bundle"]))

    @app.post("/api/projects/{project_id}/policy")
    def policy(project_id: str, body: PolicyRequest):
        try:
            return service.change_policy(store, project_id, body.base_revision,
                                          [change.model_dump() for change in body.changes])
        except ValueError as exc:
            raise HTTPException(409 if "another tab" in str(exc) else 422, str(exc)) from None

    @app.get("/api/revisions/{revision_id}")
    def revision(revision_id: str):
        return service.revision_summary(required("revisions", revision_id))

    @app.get("/api/revisions/{revision_id}/diff/{previous_id}")
    def diff(revision_id: str, previous_id: str):
        before, after = ready_revision(previous_id), ready_revision(revision_id)
        if before["project_id"] != after["project_id"]:
            raise HTTPException(422, "Compare revisions from the same project")
        return service.revision_diff(before, after)

    @app.get("/api/revisions/{revision_id}/samples/{image_id}")
    def sample(revision_id: str, image_id: int):
        revision = ready_revision(revision_id)
        bundle = Path(revision["bundle"])
        verify(bundle)
        row = next((r for r in _jsonl(bundle / "coverage.jsonl") if r["image_id"] == image_id), None)
        if row is None:
            raise HTTPException(404, "Image not found")
        data = read_json(bundle / f"splits/{row['split']}.coco.json")
        image = next(im for im in data["images"] if im["id"] == image_id)
        annotations = [a for a in data["annotations"] if a["image_id"] == image_id]
        project = required("projects", revision["project_id"])
        reference = None
        if project["reference_bundle"]:
            reference_root = Path(project["reference_bundle"])
            original = read_json(reference_root / f"splits/{row['split']}.coco.json")
            if any(im["id"] == image_id and im["file_name"] == image["file_name"] for im in original["images"]):
                reference = [a for a in original["annotations"] if a["image_id"] == image_id]
        provenance = next(r for r in _jsonl(bundle / "provenance.jsonl") if r["image_id"] == image_id)
        return {"image": image, "image_url": f"/api/revisions/{revision_id}/images/{image_id}",
                "coverage": row, "annotations": annotations, "reference": reference, "provenance": provenance}

    @app.get("/api/revisions/{revision_id}/images/{image_id}")
    def image_file(revision_id: str, image_id: int):
        revision = ready_revision(revision_id)
        bundle = Path(revision["bundle"])
        verify(bundle)
        for split in ("train", "valid", "test"):
            for im in read_json(bundle / f"splits/{split}.coco.json")["images"]:
                if im["id"] == image_id:
                    return FileResponse(safe_child(bundle, im["file_name"]))
        raise HTTPException(404, "Image not found")

    @app.get("/api/revisions/{revision_id}/download")
    def download(revision_id: str):
        revision = ready_revision(revision_id)
        view = Path(revision["view"])
        manifest = verify(view)
        target = store.root / "exports" / f"{manifest['digest']}.zip"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix("."+identifier()+".tmp")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ["manifest.json", *manifest["files"]]:
                    archive.write(safe_child(view, name), name)
            temporary.replace(target)
        return FileResponse(target, filename=f"coveragecv-r{revision['number']}-{manifest['digest'][:10]}.zip")

    @app.post("/api/revisions/{revision_id}/train")
    def train(revision_id: str, body: TrainRequest):
        if len(set(body.arms)) != len(body.arms):
            raise HTTPException(422, "Choose each comparison arm once")
        validate_local_device(body.device)
        return service.queue_training(store, ready_revision(revision_id), body.model_dump())

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return live_job(required("jobs", job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        required("jobs", job_id)
        return store.request_cancel(job_id)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str):
        required("jobs", job_id)
        path = store.root / "jobs" / job_id / "worker.log"
        if not path.exists():
            return {"text": ""}
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size-16000))
            return {"text": stream.read().decode(errors="replace")}

    @app.get("/api/jobs/{job_id}/predictions/{image_id}")
    def predictions(job_id: str, image_id: int):
        job = live_job(required("jobs", job_id))
        result = job.get("result") or {}
        arms = {}
        for name, arm in result.get("arms", {}).items():
            evaluation = read_json(Path(arm["evaluation_path"]))
            arms[name] = [p for p in evaluation["predictions"] if p["image_id"] == image_id]
        return arms

    @app.get("/api/jobs/{job_id}/examples")
    def job_examples(job_id: str):
        job = live_job(required("jobs", job_id))
        result = job.get("result") or {}
        reference = result.get("reference_bundle") or job["payload"].get("reference")
        if not reference:
            raise HTTPException(409, "This job has no evaluation reference yet")
        bundle = Path(reference)
        manifest = verify(bundle)
        data = read_json(bundle / "splits/valid.coco.json")
        return {"bundle_digest": manifest["digest"], "classes": read_json(bundle / "ontology.json")["classes"],
                "images": [{**im, "image_url": f"/api/jobs/{job_id}/images/{im['id']}",
                            "annotations": [a for a in data["annotations"] if a["image_id"] == im["id"]]}
                           for im in data["images"]]}

    @app.get("/api/jobs/{job_id}/images/{image_id}")
    def job_image(job_id: str, image_id: int):
        job = live_job(required("jobs", job_id))
        reference = (job.get("result") or {}).get("reference_bundle") or job["payload"].get("reference")
        if not reference:
            raise HTTPException(409, "No evaluation reference")
        bundle = Path(reference)
        verify(bundle)
        for im in read_json(bundle / "splits/valid.coco.json")["images"]:
            if im["id"] == image_id:
                return FileResponse(safe_child(bundle, im["file_name"]))
        raise HTTPException(404, "Validation image not found")

    @app.post("/api/projects/{project_id}/adopt-pilot")
    def adopt_pilot(project_id: str):
        project = required("projects", project_id)
        revision = ready_revision(project["active_revision"])
        workspace = Path(__file__).resolve().parents[3]
        experiment = read_json(workspace / "artifacts/chess_experiment.json")
        if project["reference_bundle"] != experiment["complete_bundle"]:
            raise HTTPException(422, "The saved pilot belongs to the prepared chess project")
        return service.import_saved_experiment(store, project_id, revision["id"], workspace / "artifacts/mvp",
                                                Path(experiment["complete_bundle"]))

    @app.get("/api/platform")
    def platform():
        workspace = Path(__file__).resolve().parents[3]
        provider = workspace / "artifacts/provider"
        names = {"dataset": "roboflow_upload.json", "verification": "roundtrip_verification.json",
                 "deployment": "model/server_status.json", "hosted": "hosted/training.json",
                 "hosted_inference": "model/hosted_prediction.json",
                 "corrected_deployment": "model/semantic-corrected/deployment.json",
                 "semantic_parity": "model/semantic-corrected/semantic_parity.json",
                 "modal_billing": "modal_billing.json"}
        result = {}
        for name, filename in names.items():
            path = provider / filename
            if path.exists():
                data = read_json(path)
                result[name] = {k: v for k, v in data.items() if k != "images"}
        return result

    @app.get("/api/events")
    async def events(request: Request):
        try:
            after = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            after = 0

        async def stream():
            cursor, last_heartbeat = after, time.monotonic()
            while not await request.is_disconnected():
                for event in store.events(cursor):
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                if time.monotonic()-last_heartbeat > 10:
                    yield ": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                await asyncio.sleep(0.5)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
