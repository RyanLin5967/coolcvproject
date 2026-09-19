"""Synthetic geometry and TRAIN-gate contracts; never import or execute a model."""
import copy

import pytest

from coveragecv.training.context_zoom import (
    assert_prediction_contract,
    crop_plan,
    crop_window,
    gate_predictions,
    holdout_ids,
    map_crop_box,
    refine_anchor,
)


def pred(x=150., y=150., w=40., h=60., *, score=.9, label=1, image=1):
    return {"image_id": image, "category_id": label, "score": score, "bbox": [x, y, w, h]}


def test_crop_retains_square_context_and_exact_integer_frame_shift():
    assert crop_window([600., 601., 30., 30.], (640, 640)) == (320, 320, 640, 640)
    assert crop_window([100., 100., 107., 40.], (640, 640)) == (0, 0, 321, 321)
    assert crop_window([0., 0., 171., 10.], (640, 640)) is None
    assert crop_window([0., 0., 0., 30.], (640, 640)) is None


def test_candidate_mapping_uses_actual_crop_and_rejects_only_internal_edges():
    assert map_crop_box([10., 15., 40., 60.], (320, 320, 640, 640), (640, 640)) == [330., 335., 40., 60.]
    assert map_crop_box([2., 15., 40., 60.], (320, 320, 640, 640), (640, 640)) is None
    assert map_crop_box([0., 15., 40., 60.], (0, 0, 320, 320), (640, 640)) == [0., 15., 40., 60.]
    assert map_crop_box([280., 15., 40., 60.], (320, 320, 640, 640), (640, 640)) == [600., 335., 40., 60.]


def test_plan_preserves_zero_area_and_low_score_outputs_and_bounds_cloud_work():
    rows = [pred(score=.95, w=0), pred(score=.01)] + [pred(score=.9-i*.01) for i in range(40)]
    before = copy.deepcopy(rows)
    plan = crop_plan(rows, (640, 640))
    assert len(plan) == 32
    assert [index for index, _ in plan] == list(range(2, 34))
    assert rows == before


def test_association_uses_iou_then_confidence_and_preserves_detection_decisions():
    anchor = pred()
    # Crop origin is (10, 20); both candidates have equal overlap with the anchor.
    candidates = [pred(142, 130, score=.7), pred(138, 130, score=.8), pred(140, 130, score=.99, label=2)]
    adjusted, matched = refine_anchor(anchor, candidates, (10, 20, 330, 340), (640, 640))
    assert matched and adjusted["bbox"] == [149., 150., 40., 60.]
    assert_prediction_contract([anchor], [adjusted])
    assert anchor["bbox"] == [150., 150., 40., 60.]
    candidates[0]["score"] = .8
    adjusted, _ = refine_anchor(anchor, candidates, (10, 20, 330, 340), (640, 640))
    assert adjusted["bbox"][0] == 151.  # deterministic first-index tie


def test_correction_is_bounded_and_unmatched_zero_area_is_identity():
    anchor = pred(w=100, h=100)
    adjusted, matched = refine_anchor(anchor, [pred(167, 150, 100, 100)], (0, 0, 512, 512), (640, 640))
    assert matched and adjusted["bbox"] == [157.5, 150., 100., 100.]
    empty = pred(w=0)
    assert refine_anchor(empty, [anchor], (0, 0, 512, 512), (640, 640)) == (empty, False)
    assert refine_anchor(anchor, [pred(350, 350, 100, 100)], (0, 0, 512, 512), (640, 640)) == (anchor, False)


def test_holdout_groups_duplicate_image_bytes():
    images = [{"id": i, "file_name": f"{i}.jpg"} for i in range(1000)]
    manifest = {"files": {f"train/{i}.jpg": str(i//2) for i in range(1000)}}
    selected = holdout_ids(images, manifest)
    assert 100 < len(selected) < 400
    assert all((i in selected) == (i+1 in selected) for i in range(0, 1000, 2))


def gate_fixture(count=100):
    original = [pred(1., 0., 10., 10., image=i) for i in range(count)]
    zoom = [pred(0., 0., 10., 10., image=i) for i in range(count)]
    ground = [{"id": i, "image_id": i, "category_id": 1, "bbox": [0., 0., 10., 10.]} for i in range(count)]
    return original, zoom, ground, set(range(count))


def test_gate_requires_enough_matches_and_real_mean_geometry_gain():
    original, zoom, ground, ids = gate_fixture()
    result = gate_predictions(original, zoom, ground, ids)
    assert result["accepted"] and result["heldout_matches"] == 100
    assert result["after_strict_matches"] == 100
    result = gate_predictions(original, original, ground, ids)
    assert not result["accepted"] and "insufficient_mean_iou_gain" in result["rejection_reasons"]
    result = gate_predictions(*gate_fixture(99))
    assert not result["accepted"] and "insufficient_observed_matches" in result["rejection_reasons"]


def test_gate_rejects_strict_regression_even_if_mean_iou_improves():
    original, zoom, ground, ids = gate_fixture()
    for i in range(100):
        original[i]["bbox"][0] = .5 if i < 50 else 3.
        zoom[i]["bbox"][0] = .6
    result = gate_predictions(original, zoom, ground, ids)
    assert result["after_mean_iou"]-result["before_mean_iou"] >= .003
    assert "strict_iou_matches_decreased" in result["rejection_reasons"]


def test_gate_rejects_any_new_half_iou_loss_and_changed_confidences():
    original, zoom, ground, ids = gate_fixture()
    zoom[0]["bbox"][0] = 5.
    result = gate_predictions(original, zoom, ground, ids)
    assert not result["accepted"] and result["new_iou_below_half_losses"] == 1
    zoom[0]["score"] = .5
    with pytest.raises(ValueError, match="non-geometric"):
        gate_predictions(original, zoom, ground, ids)


def test_gate_does_not_count_duplicate_predictions_as_new_observed_objects():
    original, zoom, ground, ids = gate_fixture(99)
    original.append(copy.deepcopy(original[0]))
    zoom.append(copy.deepcopy(zoom[0]))
    result = gate_predictions(original, zoom, ground, ids)
    assert result["heldout_matches"] == 99 and not result["accepted"]
    ground[0]["is_pseudo"] = True
    with pytest.raises(ValueError, match="human"):
        gate_predictions(original, zoom, ground, ids)


def test_failed_train_gate_never_opens_validation_reference(tmp_path, monkeypatch):
    import json

    from coveragecv.training import context_zoom

    original, _, ground, ids = gate_fixture()
    data = {"images": [{"id": i} for i in ids], "annotations": ground}

    def verify_train_only(path):
        assert path == tmp_path / "view", "Validation reference was accessed before the gate passed"
        return {"kind": "training_view", "digest": "partial"}

    def read_train_only(path):
        if path == tmp_path / "view/train/_annotations.coco.json":
            return data
        assert path == tmp_path / "view/ontology.json"
        return {"classes": ["object"]}

    calls = []

    def synthetic_inference(checkpoint, classes, images, image_root):
        calls.append(checkpoint)
        return original, copy.deepcopy(original), {"device": "cuda", "images": len(images)}

    monkeypatch.setattr(context_zoom, "verify", verify_train_only)
    monkeypatch.setattr(context_zoom, "read_json", read_train_only)
    monkeypatch.setattr(context_zoom, "holdout_ids", lambda *_: ids)
    monkeypatch.setattr(context_zoom, "_checkpoint", lambda root, protocol, name: name)
    monkeypatch.setattr(context_zoom, "infer_zoom", synthetic_inference)
    result = context_zoom.run(tmp_path, {"policy": context_zoom.POLICY, "view_digest": "partial"}, tmp_path / "output")
    assert result["status"] == "rejected_before_validation"
    assert calls == ["aware"] and not result["evaluations"]
    saved = json.loads((tmp_path / "output/result.json").read_text())
    assert saved == result
