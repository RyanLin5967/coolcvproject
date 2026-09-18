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

from coveragecv.artifacts import read_json, safe_child, verify
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
