"""No model imports: exposure planning and durable cloud submission contracts."""
import importlib.util
from pathlib import Path

import pytest

from coveragecv.artifacts import digest, write_json
from coveragecv.training import exposure


@pytest.fixture
def observed_view(tmp_path, monkeypatch):
    images = [{"id": i, "file_name": f"{i}.jpg"} for i in range(10)]
    annotations = [{"image_id": i, "category_id": 1} for i in range(8)]
    annotations += [{"image_id": 9, "category_id": 2}]
    write_json(tmp_path / "train/_annotations.coco.json", {"images": images, "annotations": annotations})
    monkeypatch.setattr(exposure, "verify", lambda _: {
        "kind": "training_view", "digest": "partial", "files": {f"train/{i}.jpg": f"hash{i}" for i in range(10)}})
    return tmp_path


def test_exposure_is_deterministic_and_boosts_observed_rare_images(observed_view):
    kwargs = {"samples": 8000, "seed": 19, "repeat": True}
    plan = exposure.build_exposure_plan(observed_view, **kwargs)
    assert plan == exposure.build_exposure_plan(observed_view, **kwargs)
    assert plan["class_repeat_factors"]["2"] == pytest.approx(2 ** .5)
    assert plan["class_repeat_factors"]["1"] == 1
    uniform = exposure.build_exposure_plan(observed_view, **{**kwargs, "repeat": False})
    assert plan["sample_ids"].count(9) > uniform["sample_ids"].count(9)
    assert len(plan["sample_ids"]) == 8000
    assert set(plan["sample_ids"]) == set(range(10))  # retain background images too
    assert plan["digest"] == digest({k: v for k, v in plan.items() if k != "digest"})


def test_pseudo_annotations_cannot_define_observed_exposure(observed_view):
    write_json(observed_view / "train/_annotations.coco.json", {
        "images": [{"id": 1, "file_name": "1.jpg"}],
        "annotations": [{"image_id": 1, "category_id": 1, "is_pseudo": True}]})
    with pytest.raises(ValueError, match="observed"):
        exposure.build_exposure_plan(observed_view, samples=8, seed=19, repeat=True)


@pytest.fixture
def collector(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_continuous_scale.py"
    spec = importlib.util.spec_from_file_location("continuous_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected new paid submission")

    monkeypatch.setattr(module, "reserve_continuous_scale", forbidden)
    monkeypatch.setattr(module.modal.Function, "from_name", forbidden)
    return module


def test_saved_call_resumes_without_reserving_or_submitting(collector, tmp_path, monkeypatch):
    request = {"case": "aware-scale", "protocol_sha256": "frozen"}
    write_json(tmp_path / "call.json", {"request": request, "call_id": "paid"})

    class Paid:
        def get(self):
            return b"existing checkpoint"

    monkeypatch.setattr(collector.modal.FunctionCall, "from_id", lambda call_id: Paid() if call_id == "paid" else None)
    assert collector.invoke(tmp_path, "train", request, request) == b"existing checkpoint"


def test_orphan_and_changed_request_fail_without_spending(collector, tmp_path):
    request = {"protocol_sha256": "new"}
    write_json(tmp_path / "reservation.json", {"reservation_id": "paid"})
    with pytest.raises(RuntimeError, match="Orphan"):
        collector.invoke(tmp_path, "train", request, request)
    write_json(tmp_path / "call.json", {"request": {"protocol_sha256": "old"}, "call_id": "paid"})
    with pytest.raises(ValueError, match="different request"):
        collector.invoke(tmp_path, "train", request, request)
