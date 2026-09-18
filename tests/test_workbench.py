import asyncio
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from coveragecv.artifacts import digest, file_digest, read_json, verify, write_json
from coveragecv.workbench.api import (
    acquisition_evidence,
    create_app,
    object_crop_evidence,
    refinement_evidence,
)
from coveragecv.workbench.scheduler import Scheduler
from coveragecv.workbench.service import extract_dataset, revision_diff
from coveragecv.workbench.store import Store
from coveragecv.workbench.worker import execute

HEADERS = {"X-CoverageCV": "workbench"}


@pytest.mark.parametrize("cursor", [None, "", "invalid", "-1", "99999"])
def test_new_event_stream_syncs_once_without_replaying_history(tmp_path, cursor):
    app = create_app(tmp_path / "state", run_scheduler=False)
    store = app.state.store
    with store.connection() as db:
        for _ in range(205):  # More than two historical event batches.
            store._event(db, None, {"type": "completed"})
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/events")
    request = SimpleNamespace(headers={} if cursor is None else {"last-event-id": cursor},
                              is_disconnected=AsyncMock(return_value=False))

    async def inspect():
        response = await endpoint(request)
        stream = response.body_iterator
        try:
            frame = await anext(stream)
            assert frame.startswith("id: 205\n")
            assert json.loads(frame.split("data: ")[1]) == {"id": 205, "data": {"type": "sync"}}
            # A mutation arriving during the initial state fetch still follows
            # the sync; no older event may be replayed in its place.
            with store.connection() as db:
                store._event(db, None, {"type": "queued", "project_id": "new-project"})
            frame = await anext(stream)
            assert frame.startswith("id: 206\n")
            assert json.loads(frame.split("data: ")[1])["data"]["project_id"] == "new-project"
        finally:
            await stream.aclose()

    asyncio.run(inspect())


@pytest.mark.parametrize("cursor", [0, 199])
def test_reconnected_event_stream_replays_only_missed_events(tmp_path, cursor):
    app = create_app(tmp_path / "state", run_scheduler=False)
    store = app.state.store
    with store.connection() as db:
        for index in range(205):
            store._event(db, None, {"type": "completed", "sequence": index + 1})
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/events")
    request = SimpleNamespace(headers={"last-event-id": str(cursor)},
                              is_disconnected=AsyncMock(return_value=False))

    async def inspect():
        response = await endpoint(request)
        stream = response.body_iterator
        try:
            for expected in range(cursor + 1, 206):
                frame = await anext(stream)
                event = json.loads(frame.split("data: ")[1])
                assert event["id"] == expected
                assert event["data"] == {"type": "completed", "sequence": expected}
        finally:
            await stream.aclose()

    asyncio.run(inspect())


def test_idle_event_stream_heartbeat_is_not_a_state_change(tmp_path, monkeypatch):
    from coveragecv.workbench import api
    app = create_app(tmp_path / "state", run_scheduler=False)
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/api/events")
    request = SimpleNamespace(headers={}, is_disconnected=AsyncMock(return_value=False))
    clock = iter([0, 11])
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    async def inspect():
        response = await endpoint(request)
        stream = response.body_iterator
        try:
            assert "sync" in await anext(stream)
            assert await anext(stream) == ": heartbeat\n\n"
        finally:
            await stream.aclose()

    asyncio.run(inspect())


def test_refiner_evidence_requires_all_arms_and_durable_completion(tmp_path):
    folder = tmp_path / "refinement/all-pieces/20260917/partial-human"
    folder.mkdir(parents=True)
    assert refinement_evidence(tmp_path)["status"] == "not_started"
    write_json(folder / "call.json", {"call_id": "test-call"})
    assert refinement_evidence(tmp_path)["status"] == "submitted"
    receipt = {"status": "trained", "checkpoint_sha256": "refiner-hash", "evaluation_status": "pending"}
    write_json(folder / "receipt.json", receipt)
    assert refinement_evidence(tmp_path)["status"] == "evaluating"
    comparison = {"status": "completed", "refiner_sha256": "refiner-hash", "same_refiner_for_all_arms": True,
                  "results": {method: {"refiner_checkpoint_sha256": "refiner-hash",
                                       "classification_and_scores_changed": False,
                                       "train_evaluation_exact_image_overlap": False,
                                       "baseline_metrics": {"AP": .73}, "metrics": {"AP": .68},
                                       "predictions": [{"image_id": 1}]}
                              for method in ("naive_augmented_512", "aware_augmented_512", "complete_augmented_512")}}
    write_json(folder / "comparison.json", comparison)
    assert refinement_evidence(tmp_path)["results"] == {}
    receipt.update(status="completed", evaluation_status="completed")
    write_json(folder / "receipt.json", receipt)
    evidence = refinement_evidence(tmp_path)
    assert evidence["status"] == "completed" and evidence["n"] == 1
    assert len(evidence["results"]) == 3
    assert evidence["results"]["aware_augmented_512"]["metrics"]["AP"] == .68  # Preserve regressions.
    assert "predictions" not in evidence["results"]["aware_augmented_512"]


@pytest.mark.parametrize("invalid", ["checkpoint", "changed_scores", "overlap", "missing_arm"])
def test_refiner_rejects_inconsistent_comparison_provenance(tmp_path, invalid):
    folder = tmp_path / "refinement/all-pieces/20260917/partial-human"
    folder.mkdir(parents=True)
    write_json(folder / "receipt.json", {"status": "completed", "evaluation_status": "completed",
                                         "checkpoint_sha256": "refiner-hash"})
    row = {"refiner_checkpoint_sha256": "refiner-hash", "classification_and_scores_changed": False,
           "train_evaluation_exact_image_overlap": False}
    if invalid == "checkpoint":
        row["refiner_checkpoint_sha256"] = "different-model"
    elif invalid == "changed_scores":
        row["classification_and_scores_changed"] = True
    elif invalid == "overlap":
        row["train_evaluation_exact_image_overlap"] = True
    methods = ["naive_augmented_512", "aware_augmented_512", "complete_augmented_512"]
    if invalid == "missing_arm":
        methods.pop()
    write_json(folder / "comparison.json", {"status": "completed", "refiner_sha256": "refiner-hash",
               "same_refiner_for_all_arms": True,
               "results": {method: row for method in methods}})
    evidence = refinement_evidence(tmp_path)
    assert evidence["status"] == "evidence_incomplete" and evidence["results"] == {}


@pytest.fixture
def crop_study(tmp_path):
    study = tmp_path / "object-crops"
    output = study / "construction/20260917"
    cases = {"aware_standard": ("aware", False), "aware_object_crops": ("aware", True),
             "naive_object_crops": ("naive", True), "complete_reference_object_crops": ("complete_reference", True)}
    plan = {"source_view_digest": "partial-view", "train_only": True, "validation_or_test_labels_used": False,
            "classes": ["no-helmet"], "samples": 2, "anchor_counts": {"1": 33}}
    plan["digest"] = digest(plan)
    protocol = {"cases": list(cases), "crop_plan_digest": plan["digest"], "partial_view_digest": "partial-view",
                "complete_view_digest": "complete-view", "reference_bundle_digest": "reference",
                "seed": 20260917, "steps": 2000, "total_steps": 6000, "batch": 4,
                "resolution": 512, "recipe": "augmented", "validation_during_training": False,
                "test_evaluated": False, "parents": {arm: {"checkpoint_sha256": f"parent-{arm}"}
                                                    for arm, _ in cases.values()}}
    write_json(study / "plan.json", plan)
    write_json(study / "protocol.json", protocol)
    write_json(study / "plan_audit.json", {"plan_digest": plan["digest"], "full_frame_samples": 1,
               "anchor_samples_by_class": {"1": 1}, "median_anchor_max_side_pixels_by_class": {"1": 96}})
    protocol_sha = file_digest(study / "protocol.json")
    comparison = {"status": "completed", "seed": 20260917, "protocol_sha256": protocol_sha,
                  "crop_plan_digest": plan["digest"], "primary_AP_delta_points": 5, "results": {}}
    for case, (arm, crops) in cases.items():
        folder = output / case
        folder.mkdir(parents=True)
        (folder / "detector.pt").write_bytes(case.encode())
        sha = file_digest(folder / "detector.pt")
        parent_sha = protocol["parents"][arm]["checkpoint_sha256"]
        metrics = {"AP": .5 if crops else .45, "AP50": .8, "AP75": .4, "per_class_AP": {"no-helmet": .25}}
        parent_metrics = {"AP": .4, "AP50": .7, "AP75": .3, "per_class_AP": {"no-helmet": .15}}
        write_json(folder / "call.json", {"request": {"case": case, "protocol_sha256": protocol_sha}})
        write_json(folder / "receipt.json", {"status": "completed", "evaluation_status": "completed",
                                             "checkpoint_sha256": sha})
        write_json(folder / "run.json", {"status": "completed", "detector_sha256": sha, "arm": arm,
                   "method": case, "seed": 20260917, "steps": 2000, "total_training_steps": 6000,
                   "batch": 4, "recipe": "augmented", "model_config": {"resolution": 512},
                   "view_digest": "complete-view" if arm == "complete_reference" else "partial-view",
                   "parent_checkpoint_sha256": parent_sha, "warm_start_sha256": parent_sha,
                   "experiment_protocol_sha256": protocol_sha, "object_crop_plan_digest": plan["digest"] if crops else None,
                   "validation_during_training": False})
        write_json(folder / "evaluation.json", {"checkpoint_sha256": sha, "split": "valid",
                                                "bundle_digest": "reference", "metrics": metrics})
        write_json(tmp_path / "gpu/construction/20260917" / arm / "evaluation.json",
                   {"checkpoint_sha256": parent_sha, "split": "valid", "bundle_digest": "reference",
                    "metrics": parent_metrics})
        comparison["results"][case] = {"metrics": metrics, "parent_metrics": parent_metrics,
                                         "checkpoint_sha256": sha, "parent_checkpoint_sha256": parent_sha}
    write_json(output / "comparison.json", comparison)
    return tmp_path


def test_crop_study_keeps_continuations_separate_and_waits_for_evaluation(crop_study):
    evidence = object_crop_evidence(crop_study)
    assert evidence["status"] == "completed" and evidence["n"] == 1
    assert evidence["primary_AP_delta_points"] == pytest.approx(5)
    assert len(evidence["cases"]) == 4
    assert evidence["cases"]["aware_object_crops"]["parent_metrics"]["AP"] == .4
    receipt_path = crop_study / "object-crops/construction/20260917/aware_standard/receipt.json"
    receipt = read_json(receipt_path)
    receipt.update(status="trained", evaluation_status="pending")
    write_json(receipt_path, receipt)
    evidence = object_crop_evidence(crop_study)
    assert evidence["status"] == "in_progress" and "primary_AP_delta_points" not in evidence
    assert evidence["cases"]["aware_standard"]["status"] == "evaluating"
    assert "metrics" not in evidence["cases"]["aware_standard"]


@pytest.mark.parametrize("invalid", ["checkpoint", "reference", "request", "plan", "warm_start", "comparison"])
def test_crop_study_refuses_broken_artifact_bindings(crop_study, invalid):
    study = crop_study / "object-crops"
    case = study / "construction/20260917/aware_object_crops"
    if invalid == "checkpoint":
        (case / "detector.pt").write_bytes(b"different trained model")
    else:
        path, key, value = {
            "reference": (case / "evaluation.json", "bundle_digest", "different-reference"),
            "request": (case / "call.json", "request", {"case": "different-study"}),
            "plan": (study / "plan.json", "source_view_digest", "unobserved-source"),
            "warm_start": (case / "run.json", "warm_start_sha256", "different-parent"),
            "comparison": (study / "construction/20260917/comparison.json", "primary_AP_delta_points", 99),
        }[invalid]
        data = read_json(path)
        data[key] = value
        write_json(path, data)
    evidence = object_crop_evidence(crop_study)
    assert "primary_AP_delta_points" not in evidence
    if invalid in ("comparison", "plan"):
        assert evidence["status"] == "evidence_incomplete"
    else:
        assert evidence["cases"]["aware_object_crops"]["status"] == "evidence_incomplete"
        assert "metrics" not in evidence["cases"]["aware_object_crops"]


@pytest.fixture
def acquisition_study(crop_study):
    study = crop_study / "acquisition"
    zero = crop_study / "object-crops/construction/20260917/aware_standard"
    old_protocol = read_json(crop_study / "object-crops/protocol.json")
    declaration = {"reference_labels_read_for_selection": False, "test_labels_used": False,
                   "partial_view_digest": "partial-view", "plans": {"guided": "guided-plan", "random": "random-plan"}}
    write_json(study / "selection_frozen.json", {"declaration": declaration, "declaration_digest": digest(declaration),
                                                "frozen_at": "2026-09-18T00:00:00Z"})
    protocol = {key: old_protocol[key] for key in ("seed", "steps", "batch", "recipe", "resolution", "parents",
                                                  "reference_bundle_digest", "validation_during_training")}
    protocol.update(kind="fixed_budget_coverage_acquisition_simulation", total_training_steps=6000,
                    selection_frozen_file_sha256=file_digest(study / "selection_frozen.json"),
                    review_units_per_acquisition_arm=150, views={}, limitations=["Simulation only"],
                    zero_review_control={"folder": str(zero.relative_to(crop_study.parent)),
                                         "checkpoint_sha256": file_digest(zero / "detector.pt")})
    cases = {"guided": "aware", "random": "aware", "complete_standard": "complete_reference"}
    for case in cases:
        acquisition = None
        if case != "complete_standard":
            per_class = [{"class_name": name, "review_units": 30, "boxes_before": 33,
                          "added_boxes": 2 if case == "guided" else 1,
                          "boxes_after": 35 if case == "guided" else 34}
                         for name in ("helmet", "no-helmet", "no-vest", "person", "vest")]
            acquisition = {"simulation": True, "actual_new_human_annotation": False,
                           "same_label_budget_as_original_experiment": False, "validation_or_test_labels_used": False,
                           "unrequested_reference_labels_used": False, "plan_digest": declaration["plans"][case],
                           "partial_view_digest": "partial-view", "review_units": 150, "per_class": per_class,
                           "added_boxes": 10 if case == "guided" else 5, "boxes_before": 165,
                           "boxes_after": 175 if case == "guided" else 170}
        protocol["views"][case] = {"digest": f"{case}-view", "acquisition": acquisition}
    write_json(study / "protocol.json", protocol)
    protocol_sha = file_digest(study / "protocol.json")
    output = study / "construction/20260917"
    results = {}
    for case, arm in cases.items():
        folder = output / case
        folder.mkdir(parents=True)
        (folder / "detector.pt").write_bytes(f"acquisition-{case}".encode())
        sha = file_digest(folder / "detector.pt")
        parent = protocol["parents"][arm]["checkpoint_sha256"]
        run = {key: protocol[key] for key in ("seed", "steps", "batch", "recipe", "total_training_steps",
                                             "validation_during_training")}
        run.update(status="completed", arm=arm, method=case, detector_sha256=sha,
                   view_digest=protocol["views"][case]["digest"], parent_checkpoint_sha256=parent,
                   warm_start_sha256=parent, experiment_protocol_sha256=protocol_sha,
                   model_config={"resolution": 512}, annotation_acquisition=protocol["views"][case]["acquisition"])
        write_json(folder / "run.json", run)
        write_json(folder / "call.json", {"request": {"case": case, "protocol_sha256": protocol_sha}})
        write_json(folder / "receipt.json", {"status": "completed", "evaluation_status": "completed",
                                             "checkpoint_sha256": sha})
        metrics = {"AP": {"guided": .42, "random": .46, "complete_standard": .6}[case], "AP50": .8}
        write_json(folder / "evaluation.json", {"checkpoint_sha256": sha, "split": "valid",
                   "bundle_digest": "reference", "metrics": metrics, "device": "cpu", "resolution": 512,
                   "score_threshold": 0.25, "postprocess": "stock RF-DETR; reserved output omitted from semantic COCO mapping"})
        results[case] = {"metrics": metrics, "checkpoint_sha256": sha,
                         "evaluation_sha256": file_digest(folder / "evaluation.json"),
                         "acquisition": protocol["views"][case]["acquisition"]}
    zero_eval = read_json(zero / "evaluation.json")
    zero_eval.update(device="cpu", resolution=512, score_threshold=0.25,
                     postprocess="stock RF-DETR; reserved output omitted from semantic COCO mapping")
    write_json(zero / "evaluation.json", zero_eval)
    results["zero_review"] = {"metrics": zero_eval["metrics"], "checkpoint_sha256": zero_eval["checkpoint_sha256"],
                              "evaluation_sha256": file_digest(zero / "evaluation.json"), "acquisition": None}
    write_json(output / "comparison.json", {"status": "completed", "seed": 20260917, "results": results,
               "protocol_sha256": protocol_sha, "review_units_per_acquisition_arm": 150,
               "primary_AP_delta_points": -4, "guided_vs_zero_review_AP_delta_points": -3,
               "random_vs_zero_review_AP_delta_points": 1})
    return crop_study


def test_acquisition_reports_real_yield_and_regression_without_claiming_new_human_labels(acquisition_study):
    evidence = acquisition_evidence(acquisition_study)
    assert evidence["status"] == "completed" and evidence["simulation"] is True and evidence["n"] == 1
    assert evidence["primary_AP_delta_points"] == pytest.approx(-4)
    assert evidence["guided_vs_zero_review_AP_delta_points"] == pytest.approx(-3)
    assert evidence["cases"]["guided"]["acquisition"]["added_boxes"] == 10
    assert evidence["cases"]["random"]["acquisition"]["added_boxes"] == 5
    assert "queries" not in evidence["cases"]["guided"]["acquisition"]
    receipt_path = acquisition_study / "acquisition/construction/20260917/guided/receipt.json"
    receipt = read_json(receipt_path)
    receipt.update(status="trained", evaluation_status="pending")
    write_json(receipt_path, receipt)
    pending = acquisition_evidence(acquisition_study)
    assert pending["status"] == "in_progress" and "primary_AP_delta_points" not in pending
    assert pending["cases"]["guided"]["status"] == "evaluating"
    assert "metrics" not in pending["cases"]["guided"]
    assert pending["cases"]["guided"]["acquisition"]["added_boxes"] == 10


@pytest.mark.parametrize("invalid", ["frozen_selection", "checkpoint", "acquisition", "reference", "comparison",
                                    "device", "resolution", "threshold", "postprocess"])
def test_acquisition_refuses_mismatched_selection_and_result_bindings(acquisition_study, invalid):
    study = acquisition_study / "acquisition"
    folder = study / "construction/20260917/guided"
    if invalid == "checkpoint":
        (folder / "detector.pt").write_bytes(b"unrelated model")
    else:
        path, key, value = {
            "frozen_selection": (study / "selection_frozen.json", "frozen_at", "changed"),
            "acquisition": (folder / "run.json", "annotation_acquisition", None),
            "reference": (folder / "evaluation.json", "bundle_digest", "changed"),
            "device": (folder / "evaluation.json", "device", "mps"),
            "resolution": (folder / "evaluation.json", "resolution", 704),
            "threshold": (folder / "evaluation.json", "score_threshold", .5),
            "postprocess": (folder / "evaluation.json", "postprocess", "different decoder"),
            "comparison": (study / "construction/20260917/comparison.json", "primary_AP_delta_points", 95),
        }[invalid]
        data = read_json(path)
        data[key] = value
        write_json(path, data)
    evidence = acquisition_evidence(acquisition_study)
    assert "primary_AP_delta_points" not in evidence
    if invalid in ("frozen_selection", "comparison"):
        assert evidence["status"] == "evidence_incomplete"
    else:
        assert evidence["cases"]["guided"]["status"] == "evidence_incomplete"
        assert "metrics" not in evidence["cases"]["guided"]


@pytest.fixture
def dataset_spec(tmp_path):
    classes = ["forklift", "person", "pallet"]
    categories = [{"id": i+1, "name": c} for i, c in enumerate(classes)]
    images = tmp_path / "images"
    images.mkdir()
    sources = []
    for split_index, split in enumerate(("train", "valid", "test")):
        data = {"images": [], "annotations": [], "categories": categories}
        for i in range(2):
            name = f"{split}-{i}.png"
            Image.new("RGB", (64, 48), (20+split_index*70, 40+i*70, 80)).save(images / name)
            data["images"].append({"id": i, "file_name": name, "width": 64, "height": 48})
            data["annotations"].append({"id": i+1, "image_id": i, "category_id": i+1,
                                         "bbox": [4, 5, 20, 25]})
        write_json(tmp_path / f"{split}.json", data)
        sources.append({"id": split, "revision": "fixture-1", "annotations": f"{split}.json", "images": "images",
            "split": split, "class_map": dict(zip(classes, classes)),
            "coverage": {c: "exhaustive" if split != "train" or i == 0 else "unknown" for i, c in enumerate(classes)},
            "evidence": "Synthetic test policy", "attribution": "Generated test images"})
    path = tmp_path / "coverage.json"
    write_json(path, {"classes": classes, "sources": sources})
    return path


def compile_queued(store):
    job = store.claim()
    folder = store.root / "jobs" / job["id"]
    folder.mkdir(parents=True)
    try:
        result = execute(job, store.root, folder)
        store.finish(job["id"], {"status": "completed", "result": result})
    except ValueError as exc:
        store.finish(job["id"], {"status": "failed", "error": {"message": str(exc)}})
    return job


def test_import_edit_diff_and_immutable_export(dataset_spec, tmp_path):
    app = create_app(tmp_path / "state", run_scheduler=False)
    with TestClient(app) as client:
        response = client.post("/api/projects/import", headers=HEADERS,
                               json={"name": "Warehouse", "spec_path": str(dataset_spec)})
        assert response.status_code == 200
        project = response.json()
        compile_queued(app.state.store)
        first = client.get(f"/api/revisions/{project['active_revision']}").json()
        assert first["classes"] == ["forklift", "person", "pallet"]
        assert first["class_stats"]["person"]["unknown_images"] == 2
        export = client.get(f"/api/revisions/{first['id']}/download")
        with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
            assert "coverage.jsonl" in archive.namelist()
            assert not any("test/" in name for name in archive.namelist())
        policy = {"base_revision": first["id"], "changes": [
            {"source": "train", "class_name": "person", "state": "positive_only"}]}
        edited = client.post(f"/api/projects/{project['id']}/policy", headers=HEADERS, json=policy)
        assert edited.status_code == 200
        assert client.post(f"/api/projects/{project['id']}/policy", headers=HEADERS, json=policy).status_code == 409
        compile_queued(app.state.store)
        second = app.state.store.get("revisions", edited.json()["id"])
        diff = revision_diff(app.state.store.get("revisions", first["id"]), second)
        assert diff["changed_images"] == 2
        assert diff["negative_training_cells_before"] == diff["negative_training_cells_after"]
        assert verify(Path(first["bundle"]))["digest"] == first["manifest"]["digest"]
        assert first["bundle"] != second["bundle"]


def test_compile_failure_preserves_previous_revision(dataset_spec, tmp_path):
    from coveragecv.workbench.service import change_policy, import_spec
    store = Store(tmp_path / "state")
    project = import_spec(store, dataset_spec, "Warehouse")
    compile_queued(store)
    first = store.get("revisions", project["active_revision"])
    second = change_policy(store, project["id"], first["id"], [
        {"source": "train", "class_name": "forklift", "state": "verified_absent"}])
    compile_queued(store)
    assert store.get("revisions", second["id"])["status"] == "failed"
    assert store.get("revisions", first["id"])["status"] == "ready"


def test_queue_claim_is_exclusive_and_restart_detects_dead_worker(dataset_spec, tmp_path):
    from coveragecv.workbench.service import import_spec
    store = Store(tmp_path / "state")
    import_spec(store, dataset_spec, "Warehouse")
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: store.claim(), range(4)))
    assert sum(row is not None for row in claimed) == 1
    job = next(row for row in claimed if row)
    with store.connection() as db:
        db.execute("UPDATE jobs SET started=?,pid=? WHERE id=?", (time.time()-30, 99999999, job["id"]))
    recovered = Store(tmp_path / "state")
    Scheduler(recovered).tick()
    assert recovered.get("jobs", job["id"])["status"] == "interrupted"


def test_mutations_reject_cross_origin_and_cancel_queued(dataset_spec, tmp_path):
    app = create_app(tmp_path / "state", run_scheduler=False)
    with TestClient(app) as client:
        payload = {"name": "Warehouse", "spec_path": str(dataset_spec)}
        assert client.post("/api/projects/import", json=payload).status_code == 403
        assert client.post("/api/projects/import", json=payload,
                           headers={**HEADERS, "Origin": "https://unrelated.example"}).status_code == 403
        client.post("/api/projects/import", json=payload, headers=HEADERS)
        job = app.state.store.jobs()[0]
        assert client.post(f"/api/jobs/{job['id']}/cancel", headers=HEADERS, json={}).json()["status"] == "cancelled"
        assert app.state.store.claim() is None


@pytest.mark.parametrize("unsafe", ["../escape.txt", "/absolute.txt", "folder/../../escape"])
def test_archive_paths_cannot_escape(tmp_path, unsafe):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(unsafe, "bad")
    with pytest.raises(ValueError):
        extract_dataset(archive, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_unknown_validation_blocks_experiment(dataset_spec, tmp_path):
    from coveragecv.workbench.service import import_spec, queue_training
    spec = read_json(dataset_spec)
    spec["sources"][1]["coverage"]["person"] = "unknown"
    write_json(dataset_spec, spec)
    store = Store(tmp_path / "state")
    project = import_spec(store, dataset_spec, "Warehouse")
    compile_queued(store)
    revision = store.get("revisions", project["active_revision"])
    with pytest.raises(ValueError, match="Validation"):
        queue_training(store, revision, {"batch": 1, "steps": 2, "seed": 42, "timeout_seconds": 30,
                                         "arms": ["aware"]})


def test_local_device_availability_and_large_recipe(dataset_spec, tmp_path, monkeypatch):
    from coveragecv.workbench import api, worker
    hardware = {"devices": {"cpu": True, "mps": False}, "mps_unavailable_reason": "MPS unavailable"}
    monkeypatch.setattr(api, "local_hardware", lambda: hardware)
    monkeypatch.setattr(worker, "local_hardware", lambda: hardware)
    app = create_app(tmp_path / "state", run_scheduler=False)
    with TestClient(app) as client:
        project = client.post("/api/projects/import", headers=HEADERS,
                              json={"name": "Warehouse", "spec_path": str(dataset_spec)}).json()
        compile_queued(app.state.store)
        endpoint = f"/api/revisions/{project['active_revision']}/train"
        assert client.get("/api/state").json()["hardware"] == hardware
        baseline = client.post(endpoint, headers=HEADERS, json={"arms": ["aware"]})
        assert baseline.status_code == 200
        assert baseline.json()["payload"]["device"] == "cpu"
        count = len(app.state.store.jobs())
        rejected = client.post(endpoint, headers=HEADERS,
                               json={"device": "mps", "recipe": "large_fresh", "arms": ["aware"]})
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == "MPS unavailable"
        assert client.post(endpoint, headers=HEADERS, json={"device": "cuda"}).status_code == 422
        assert len(app.state.store.jobs()) == count
        # Persisted jobs also fail before initialization if acceleration disappears after queuing.
        with pytest.raises(ValueError, match="MPS unavailable"):
            execute({"kind": "train", "payload": {"device": "mps"}}, app.state.store.root,
                    tmp_path / "unused")
        assert not (tmp_path / "unused").exists()
        hardware["devices"]["mps"] = True
        hardware["mps_unavailable_reason"] = None
        accepted = client.post(endpoint, headers=HEADERS,
                               json={"device": "mps", "recipe": "large_fresh", "arms": ["aware"]})
        assert accepted.status_code == 200
        assert accepted.json()["payload"]["recipe"] == "large_fresh"
        assert accepted.json()["payload"]["device"] == "mps"


@pytest.mark.parametrize("recipe,variant,device", [("pilot", "nano", "cpu"), ("large_fresh", "large", "mps")])
def test_worker_keeps_model_variants_separate_and_uses_selected_device(
        dataset_spec, tmp_path, monkeypatch, recipe, variant, device):
    pytest.importorskip("rfdetr")
    from coveragecv.training import evaluate, runner
    from coveragecv.workbench import worker
    from coveragecv.workbench.service import import_spec, queue_training
    store = Store(tmp_path / "state")
    project = import_spec(store, dataset_spec, "Warehouse")
    compile_queued(store)
    revision = store.get("revisions", project["active_revision"])
    job = queue_training(store, revision, {"batch": 1, "steps": 2, "seed": 42, "timeout_seconds": 30,
                                          "arms": ["naive", "aware"], "recipe": recipe, "device": device})
    calls = {"training": []}

    def initialize(view, output, **options):
        calls["initialization"] = (output, options)
        return output / "initialization.pt"

    def train(view, initialization, output, **options):
        calls["training"].append((initialization, options))
        return {"status": "completed"}

    monkeypatch.setattr(worker, "local_hardware", lambda: {"devices": {"cpu": True, "mps": True}})
    monkeypatch.setattr(runner, "create_initialization", initialize)
    monkeypatch.setattr(runner, "run_training", train)
    monkeypatch.setattr(evaluate, "evaluate_checkpoint", lambda *args: {"metrics": {}})
    result = execute(job, store.root, tmp_path / "output")
    path, initialization_options = calls["initialization"]
    assert path.name == f"{verify(Path(revision['view']))['ontology_digest']}-{variant}-42"
    assert initialization_options == {"seed": 42, "variant": variant}
    assert len(calls["training"]) == 2
    assert all(init == path / "initialization.pt" and opts["device"] == device and opts["recipe"] == recipe
               for init, opts in calls["training"])
    assert set(result["arms"]) == {"naive", "aware"}


@pytest.mark.training
def test_real_training_and_reload_with_three_non_chess_classes(dataset_spec, tmp_path):
    pytest.importorskip("rfdetr")
    if not (Path.home() / ".cache/coveragecv/rf-detr-nano.pth").exists():
        pytest.skip("real-model integration uses the locally cached official checkpoint; no test downloads")
    from coveragecv.training.evaluate import evaluate_checkpoint
    from coveragecv.training.runner import create_initialization, run_training
    from coveragecv.workbench.service import import_spec
    store = Store(tmp_path / "state")
    project = import_spec(store, dataset_spec, "Warehouse")
    compile_queued(store)
    revision = store.get("revisions", project["active_revision"])
    view = Path(revision["view"])
    init = create_initialization(view, tmp_path / "init")
    output = tmp_path / "trained"
    run = run_training(view, init, output, arm="aware", epochs=1, max_steps=1, batch=1, timeout_seconds=60)
    assert run["status"] == "completed"
    assert run["classes"] == ["forklift", "person", "pallet"]
    assert run["initial_parameter_digest"] != run["final_parameter_digest"]
    evaluation = evaluate_checkpoint(output / "detector.pt", Path(revision["bundle"]), output / "evaluation.json")
    assert evaluation["checkpoint_sha256"] == run["detector_sha256"]
    assert set(evaluation["metrics"]["per_class_AP"]) == set(run["classes"])
