import copy

import numpy as np
import pytest
import torch
from rfdetr.utilities.box_ops import elementwise_generalized_box_iou
from test_criterion import fixtures, leaves, stock

from coveragecv.training.criterion import CoverageSetCriterion
from coveragecv.training.localization_cascade import ExactSpatialPool, apply_corrections, training_holdout
from coveragecv.training.power_localization import power_giou_loss, with_power_localization


def test_exact_pool_preserves_old_encoder_on_supported_and_mps_geometry():
    for size in (7, 8):
        x = torch.randn(2, 5, size, size, requires_grad=True)
        expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        actual = ExactSpatialPool()(x)
        torch.testing.assert_close(actual, expected)
        ga, = torch.autograd.grad(actual.square().sum(), (x,), retain_graph=True)
        ge, = torch.autograd.grad(expected.square().sum(), (x,))
        torch.testing.assert_close(ga, ge)


def test_corner_bounds_preserve_geometry_identity_classes_and_scores():
    rows = [{"image_id": 1, "category_id": 2, "score": .75, "bbox": [-2., 4., 10., 20.]}]
    assert apply_corrections(rows, np.ones((1, 4)), 0) == rows
    adjusted = apply_corrections(rows, np.asarray([[1e4, 1e4, -1e4, -1e4]]), 1)[0]
    assert adjusted["category_id"] == 2 and adjusted["score"] == .75
    assert adjusted["bbox"][2:] == pytest.approx([7., 14.])
    assert rows[0]["bbox"] == [-2., 4., 10., 20.]
    with pytest.raises(ValueError):
        apply_corrections(rows, np.full((1, 4), np.nan), 1)


def test_holdout_keeps_duplicate_bytes_and_multiple_views_together():
    images = {i: {"file_name": f"{i}.jpg"} for i in range(500)}
    manifest = {"files": {f"train/{i}.jpg": str(i//2) for i in images}}
    rows = [{"image_id": i} for i in images for _ in range(2)]
    held = training_holdout(rows, images, manifest)
    assert np.all(held[::4] == held[1::4])
    assert np.all(held[::4] == held[2::4])
    assert np.all(held[::4] == held[3::4])


def test_alpha_one_giou_values_and_gradients_match_upstream():
    first = torch.tensor([[.1, .1, .4, .4], [.6, .6, .9, .9]], requires_grad=True)
    second = torch.tensor([[.15, .1, .5, .45], [.1, .1, .3, .3]])
    expected = 1-elementwise_generalized_box_iou(first, second)
    actual = power_giou_loss(first, second, 1)
    torch.testing.assert_close(actual, expected)
    ga, = torch.autograd.grad(actual.sum(), (first,), retain_graph=True)
    ge, = torch.autograd.grad(expected.sum(), (first,))
    torch.testing.assert_close(ga, ge)
    assert torch.equal(power_giou_loss(second, second, 3), torch.zeros(2))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("aware", [False, True])
def test_alpha_one_preserves_all_heads_losses_and_gradients(empty, aware):
    original = stock(2)
    if aware:
        original = CoverageSetCriterion.from_stock(original, torch.tensor([[True, False], [False, False]]))
    power = with_power_localization(original, 1)
    a, targets = fixtures(2, empty)
    b = copy.deepcopy(a)
    first, second = original(a, targets), power(b, targets)
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    sum(v for k, v in first.items() if k.startswith("loss_")).backward()
    sum(v for k, v in second.items() if k.startswith("loss_")).backward()
    for x, y in zip(leaves(a), leaves(b)):
        torch.testing.assert_close(x.grad, y.grad)


def test_power_changes_boxes_but_preserves_coverage_classification():
    original = CoverageSetCriterion.from_stock(stock(), torch.tensor([[True, False], [False, False]]))
    power = with_power_localization(original, 3)
    a, targets = fixtures()
    b = copy.deepcopy(a)
    first, second = original(a, targets), power(b, targets)
    for key in first:
        if "giou" not in key:
            torch.testing.assert_close(first[key], second[key])
    assert not torch.equal(first["loss_giou"], second["loss_giou"])
    sum(v for k, v in second.items() if k.startswith("loss_")).backward()
    assert torch.count_nonzero(b["pred_logits"].grad[0, :, 1]) == 0
    assert all(torch.isfinite(t.grad).all() for t in leaves(b))


@pytest.mark.parametrize("alpha", [1, 3])
def test_exclusivity_only_supervises_matched_queries_and_preserves_positives(alpha):
    base = CoverageSetCriterion.from_stock(stock(2), torch.zeros(2, 2, dtype=torch.bool))
    extended = CoverageSetCriterion.from_stock(stock(2), torch.zeros(2, 2, dtype=torch.bool),
                                                exclusive_groups=((0, 1),))
    extended = with_power_localization(extended, alpha)
    a, targets = fixtures(2)
    b = copy.deepcopy(a)
    first, second = base(a, targets), extended(b, targets)
    sum(v for k, v in first.items() if k.startswith("loss_ce")).backward()
    sum(v for k, v in second.items() if k.startswith("loss_ce")).backward()
    for before, after in zip([a]+a["aux_outputs"]+[a["enc_outputs"]],
                             [b]+b["aux_outputs"]+[b["enc_outputs"]]):
        indices = base.matcher(before, targets, group_detr=2)
        for image_id, (queries, _) in enumerate(indices):
            unmatched = torch.ones(before["pred_logits"].shape[1], dtype=torch.bool)
            unmatched[queries] = False
            gradient = after["pred_logits"].grad
            assert torch.count_nonzero(gradient[image_id, unmatched, :2]) == 0
            assert (gradient[image_id, queries, 1-image_id] > 0).all()
            torch.testing.assert_close(gradient[image_id, queries, image_id],
                                        before["pred_logits"].grad[image_id, queries, image_id])
        torch.testing.assert_close(after["pred_logits"].grad[..., 2], before["pred_logits"].grad[..., 2])


@pytest.mark.parametrize("groups", [((0,),), ((0, 0),), ((0, 2),), ((0, 1), (0, 1)), ((True, 1),)])
def test_invalid_exclusivity_declarations_fail_closed(groups):
    with pytest.raises(ValueError, match="Exclusive groups"):
        CoverageSetCriterion.from_stock(stock(), torch.zeros(2, 2, dtype=torch.bool), exclusive_groups=groups)


def test_power_wrapper_does_not_silently_drop_pseudo_box_weight():
    original = CoverageSetCriterion.from_stock(stock(), torch.zeros(2, 2, dtype=torch.bool), pseudo_box_weight=.1)
    with pytest.raises(ValueError, match="human-only"):
        with_power_localization(original, 3)
