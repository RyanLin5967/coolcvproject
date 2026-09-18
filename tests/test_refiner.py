import json

import pytest

pytest.importorskip("torchvision")
import torch
from PIL import Image

from coveragecv.artifacts import publish, stage, write_json
from coveragecv.training.refiner import (
    MODEL_CONFIG,
    CropBoxRefiner,
    CropGeometry,
    HumanBoxDataset,
    _state_digest,
    apply_deltas,
    bounded_deltas,
    flip_box,
    prepare_crop,
    refine_predictions,
)


def learner(tmp_path, annotations=None, *, kind="training_view", filename="sample.png"):
    root = tmp_path / "views"
    staging = stage(root)
    (staging / "train").mkdir()
    Image.new("RGB", (20, 10), (255, 80, 20)).save(staging / "train/sample.png")
    write_json(staging / "train/_annotations.coco.json", {
        "images": [{"id": 7, "width": 20, "height": 10, "file_name": filename}],
        "annotations": annotations if annotations is not None else [
            {"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 2, 6, 4]}],
        "categories": [{"id": 1, "name": "object"}],
    })
    # Invalid JSON deliberately proves dataset initialization/sampling never
    # parses validation labels; verification merely hashes the inventory.
    (staging / "valid").mkdir()
    (staging / "valid/_annotations.coco.json").write_text("not parsed as labels")
    return publish(staging, root, {"schema_version": 1, "kind": kind})


@pytest.mark.parametrize("box", [(0.1, 0.2, 6.3, 3.7), (-2., 4., 8., 18.),
                                   (17.8, 8.6, 22.1, 11.5), (2., 1., 19., 5.)])
def test_geometry_roundtrip_and_zero_delta_identity(box):
    geometry = CropGeometry.around(box)
    assert all(isinstance(x, int) for x in (geometry.left, geometry.top, geometry.right, geometry.bottom))
    assert geometry.denormalize(geometry.normalize(box)) == pytest.approx(box, abs=1e-13)
    corners = torch.tensor([box], dtype=torch.float64)
    assert torch.equal(apply_deltas(corners, torch.zeros_like(corners)), corners)
    normalized = torch.tensor([geometry.normalize(box)], dtype=torch.float64)
    assert torch.equal(apply_deltas(normalized, torch.zeros_like(normalized)), normalized)
    delta = torch.tensor([[.1, -.07, .12, -.04]], dtype=torch.float64)
    pixel_decoded = apply_deltas(corners, delta)[0].tolist()
    normalized_decoded = apply_deltas(normalized, delta)[0].tolist()
    assert geometry.denormalize(normalized_decoded) == pytest.approx(pixel_decoded, abs=1e-13)


def test_padding_and_flip_align_image_proposal_and_target():
    image = Image.new("RGB", (20, 10), "white")
    image.putpixel((0, 0), (255, 0, 0))
    proposal = (0., 0., 8., 4.)
    target = (1., .5, 7., 3.)
    geometry = CropGeometry.around(proposal)
    crop = geometry.extract(image)
    assert crop.size == (12, 6)
    assert crop.getpixel((0, 0)) == (0, 0, 0)
    assert crop.getpixel((-geometry.left, -geometry.top)) == (255, 0, 0)
    ordinary, p, t, _ = prepare_crop(image, proposal, target=target)
    mirrored, fp, ft, _ = prepare_crop(image, proposal, target=target, flipped=True)
    assert torch.allclose(mirrored, ordinary.flip(-1), atol=1e-7)
    assert fp.tolist() == pytest.approx(flip_box(p.tolist()))
    assert ft.tolist() == pytest.approx(flip_box(t.tolist()))
    assert flip_box(flip_box(target)) == pytest.approx(target)


def test_only_human_train_boxes_and_deterministic_online_samples(tmp_path):
    view = learner(tmp_path)
    data = HumanBoxDataset(view, seed=47)
    again = HumanBoxDataset(view, seed=47)
    assert len(data.instances) == 1
    assert data.pseudo_boxes_filtered == 0
    for index in range(3):
        first, second = data[index], again[index]
        assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert not torch.equal(data[0][1], data[1][1])


def test_pseudo_rejected_or_explicitly_filtered_and_empty_rejected(tmp_path):
    anns = [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 2, 6, 4]},
            {"id": 2, "image_id": 7, "category_id": 1, "bbox": [2, 3, 6, 4], "is_pseudo": True}]
    view = learner(tmp_path, anns)
    with pytest.raises(ValueError, match="pseudo annotations"):
        HumanBoxDataset(view)
    data = HumanBoxDataset(view, pseudo_policy="filter")
    assert len(data.instances) == 1 and data.pseudo_boxes_filtered == 1
    with pytest.raises(ValueError, match="at least one human"):
        HumanBoxDataset(learner(tmp_path / "only", anns[1:]), pseudo_policy="filter")
    with pytest.raises(ValueError, match="at least one human"):
        HumanBoxDataset(learner(tmp_path / "empty", []))
    with pytest.raises(ValueError, match="training_view"):
        HumanBoxDataset(learner(tmp_path / "wrong", kind="bundle"))


@pytest.mark.parametrize("box", [[1, 1, 0, 2], [1, 1, -1, 2]])
def test_bad_human_geometry_rejected(tmp_path, box):
    view = learner(tmp_path, [{"id": 1, "image_id": 7, "category_id": 1, "bbox": box}])
    with pytest.raises(ValueError, match="positive xyxy"):
        HumanBoxDataset(view)


def test_escaping_image_path_rejected_before_any_image_load(tmp_path):
    view = learner(tmp_path, filename="../sample.png")
    with pytest.raises(ValueError, match="relative path"):
        HumanBoxDataset(view)


def test_identity_head_spatial_features_frozen_batch_norm_and_finite_gradients():
    torch.set_num_threads(2)
    model = CropBoxRefiner(pretrained=False).train()
    assert model.encoder.avgpool.output_size == (4, 4)
    assert all(not m.training for m in model.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d))
    proposals = torch.tensor([[.2, .1, .7, .8], [.3, .25, .65, .9]])
    crops = torch.randn(2, 3, 224, 224)
    deltas = model(crops, proposals)
    assert torch.equal(deltas, torch.zeros_like(deltas))
    decoded = apply_deltas(proposals, deltas)
    assert torch.equal(decoded, proposals)
    target = proposals+.01
    loss = torch.nn.functional.smooth_l1_loss(decoded, target)
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()
    assert torch.count_nonzero(model.head[-1].weight.grad)
    extreme = apply_deltas(proposals, bounded_deltas(torch.tensor([[1e20]*4, [-1e20]*4])))
    assert torch.isfinite(extreme).all()
    assert torch.all(extreme[:, 2:] > extreme[:, :2])


def save_model(tmp_path, *, bias=0.):
    model = CropBoxRefiner(pretrained=False).eval()
    with torch.no_grad():
        model.head[-1].bias.fill_(bias)
    state = model.state_dict()
    checkpoint = tmp_path / "refiner.pt"
    torch.save({"model": state, "metadata": {"model_config": MODEL_CONFIG,
               "final_parameters_sha256": _state_digest(state)}}, checkpoint)
    return checkpoint


def test_inference_zero_identity_non_square_edge_and_class_score_preservation(tmp_path):
    Image.new("RGB", (20, 10), (70, 100, 30)).save(tmp_path / "image.png")
    images = [{"id": 1, "file_name": "image.png", "width": 20, "height": 10}]
    predictions = [{"image_id": 1, "category_id": 2, "score": .91, "bbox": [-1.25, 7.9, 8.7, 3.1]},
                   {"image_id": 1, "category_id": 9, "score": .01, "bbox": [1, 2, 3, 4]}]
    before = json.dumps(predictions)
    result = refine_predictions(save_model(tmp_path), images, tmp_path, predictions)
    assert json.dumps(result) == before
    assert json.dumps(predictions) == before


def test_top_k_threshold_order_and_untouched_geometry(tmp_path):
    Image.new("RGB", (20, 10), (70, 100, 30)).save(tmp_path / "image.png")
    images = [{"id": 1, "file_name": "image.png", "width": 20, "height": 10}]
    predictions = [{"image_id": 1, "category_id": i % 3+1, "score": score, "bbox": [1., 2., 3., 4.]}
                   for i, score in enumerate([.01, .8, .9, .8, .7])]
    before = json.dumps(predictions)
    result = refine_predictions(save_model(tmp_path, bias=.1), images, tmp_path, predictions,
                                confidence=.05, top_k=2, batch_size=1)
    assert [p["bbox"] != original["bbox"] for p, original in zip(result, predictions)] == [False, True,
                                                                                           True, False, False]
    for p, original in zip(result, predictions):
        assert {k: v for k, v in p.items() if k != "bbox"} == {k: v for k, v in original.items() if k != "bbox"}
    assert json.dumps(predictions) == before


@pytest.mark.parametrize("confidence,top_k", [(.049, 100), (.05, 101), (float("nan"), 100)])
def test_inference_rejects_policy_outside_predeclared_envelope(tmp_path, confidence, top_k):
    with pytest.raises(ValueError, match="selection policy"):
        refine_predictions(tmp_path / "not_loaded.pt", [], tmp_path, [], confidence=confidence, top_k=top_k)


def test_nonfinite_box_and_parameter_tampering_rejected(tmp_path):
    from coveragecv.training.refiner import xywh_to_xyxy
    with pytest.raises(ValueError, match="finite"):
        xywh_to_xyxy([0, 0, float("nan"), 1])
    checkpoint = save_model(tmp_path)
    state = torch.load(checkpoint, weights_only=True)
    state["model"]["head.2.bias"][0] = 1
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="digest mismatch"):
        refine_predictions(checkpoint, [], tmp_path, [])


def test_zero_area_predictions_preserved_at_any_confidence_and_do_not_take_top_k_slot(tmp_path):
    Image.new("RGB", (20, 10), (70, 100, 30)).save(tmp_path / "image.png")
    images = [{"id": 1, "file_name": "image.png", "width": 20, "height": 10}]
    predictions = [{"image_id": 1, "category_id": i+1, "score": score, "bbox": box}
                   for i, (score, box) in enumerate([
                       (.01, [1., 2., 0., 4.]), (.02, [1, 2, 3, 0]),
                       (.99, [1., 2., 0., 4.]), (.98, [1, 2, 3, 0]),
                       (.97, [1., 2., 0., 0.]), (.8, [1., 2., 3., 4.])])]
    before = json.dumps(predictions)
    result = refine_predictions(save_model(tmp_path, bias=.1), images, tmp_path, predictions, top_k=1)
    assert json.dumps(result[:5]) == json.dumps(predictions[:5])
    assert result[5]["bbox"] != predictions[5]["bbox"]
    for actual, original in zip(result, predictions):
        assert {k: v for k, v in actual.items() if k != "bbox"} == {
            k: v for k, v in original.items() if k != "bbox"}
    assert json.dumps(predictions) == before


@pytest.mark.parametrize("box", [[1, 2, -1, 4], [1, 2, 3, -1], [float("nan"), 2, 0, 4],
                                   [1, float("inf"), 3, 0], [1, 2, 3]])
def test_inference_rejects_invalid_geometry_even_below_confidence(tmp_path, box):
    images = [{"id": 1, "file_name": "unread.png", "width": 20, "height": 10}]
    predictions = [{"image_id": 1, "category_id": 1, "score": .001, "bbox": box}]
    with pytest.raises(ValueError, match="four finite coordinates and nonnegative"):
        refine_predictions(save_model(tmp_path), images, tmp_path, predictions)
