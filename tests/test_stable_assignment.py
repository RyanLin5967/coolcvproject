"""Value/gradient and grouped-assignment checks, to execute in cloud smoke only."""
import copy
import math

import pytest

# This module is cloud-smoke-only, as the docstring says: it reaches into RF-DETR
# internals that the local environment does not provide. A missing dependency must
# skip this file, not abort collection for every other test in the suite.
pytest.importorskip("rfdetr.utilities.misc",
                    reason="RF-DETR training internals are only present in the cloud image")

import torch
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher

from coveragecv.training.criterion import CoverageSetCriterion
from coveragecv.training.power_localization import with_power_localization
from coveragecv.training.stable_assignment import PositionModulatedMatcher, with_stable_assignment


def stock(groups=1):
    return SetCriterion(num_classes=3, matcher=HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2),
                        weight_dict={"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}, focal_alpha=.25,
                        losses=["labels", "boxes", "cardinality"], group_detr=groups, ia_bce_loss=True)


def fixture(groups=1, empty=False):
    torch.manual_seed(33)

    def head():
        return {"pred_logits": torch.randn(2, 6*groups, 3, requires_grad=True),
                "pred_boxes": (torch.rand(2, 6*groups, 4)*.5+.2).requires_grad_()}

    outputs = {**head(), "aux_outputs": [head(), head()], "enc_outputs": head()}
    targets = [{"image_id": torch.tensor([i]), "labels": torch.tensor([i], dtype=torch.int64),
                "boxes": torch.tensor([[.5, .5, .25, .3]])} for i in range(2)]
    if empty:
        for target in targets:
            target.update(labels=torch.empty(0, dtype=torch.int64), boxes=torch.empty((0, 4)))
    return outputs, targets


def heads(outputs):
    return [outputs, *outputs["aux_outputs"], outputs["enc_outputs"]]


def backward(losses):
    sum(value for key, value in losses.items() if key.startswith("loss_")).backward()


@pytest.mark.parametrize("groups,empty", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("training", [True, False])
def test_full_coverage_position_targets_match_exact_stock_values_and_gradients(groups, empty, training):
    original = stock(groups).train(training)
    aware = CoverageSetCriterion.from_stock(original, torch.ones(2, 2, dtype=torch.bool))
    changed = with_stable_assignment(aware, position_targets=True, position_matching=False)
    original.ia_bce_loss, original.use_position_supervised_loss = False, True
    a, targets = fixture(groups, empty)
    b = copy.deepcopy(a)
    first, second = original(a, targets, num_boxes=7.), changed(b, targets, num_boxes=7.)
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=1e-6, atol=1e-7)
    backward(first)
    backward(second)
    for before, after in zip(heads(a), heads(b), strict=True):
        for name in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(before[name].grad, after[name].grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("matching", [False, True])
def test_unknown_cells_zero_positive_reserved_and_exclusive_gradients_are_preserved(matching):
    original = stock(2)
    allowed = torch.zeros(2, 2, dtype=torch.bool)
    aware = CoverageSetCriterion.from_stock(original, allowed, exclusive_groups=((0, 1),))
    changed = with_stable_assignment(aware, position_targets=True, position_matching=matching)
    full = with_stable_assignment(original, position_targets=True, position_matching=matching)
    a, targets = fixture(2)
    b = copy.deepcopy(a)
    first, second = full(a, targets), changed(b, targets)
    sum(v for k, v in first.items() if k.startswith("loss_ce")).backward()
    sum(v for k, v in second.items() if k.startswith("loss_ce")).backward()
    for before, after in zip(heads(a), heads(b), strict=True):
        indices = changed.matcher(after, targets, group_detr=2)
        for image_id, (queries, _) in enumerate(indices):
            unmatched = torch.ones(after["pred_logits"].shape[1], dtype=torch.bool)
            unmatched[queries] = False
            assert torch.count_nonzero(after["pred_logits"].grad[image_id, unmatched, :2]) == 0
            torch.testing.assert_close(before["pred_logits"].grad[image_id, queries],
                                        after["pred_logits"].grad[image_id, queries])
            assert (after["pred_logits"].grad[image_id, queries, 1-image_id] > 0).all()
        torch.testing.assert_close(before["pred_logits"].grad[..., 2], after["pred_logits"].grad[..., 2])
        assert after["pred_boxes"].grad is None  # Quality targets and matching do not backprop into boxes.
    assert all("_coverage_allowed" not in target for target in targets)


def test_noop_preserves_power_one_box_losses_and_does_not_mutate_input():
    original = with_power_localization(CoverageSetCriterion.from_stock(stock(), torch.ones(2, 2, dtype=torch.bool)), 1)
    changed = with_stable_assignment(original, position_targets=False, position_matching=False)
    a, targets = fixture()
    b = copy.deepcopy(a)
    first, second = original(a, targets), changed(b, targets)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    backward(first)
    backward(second)
    for before, after in zip(heads(a), heads(b), strict=True):
        for name in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(before[name].grad, after[name].grad, rtol=0, atol=0)
    assert original.ia_bce_loss and not original.use_position_supervised_loss
    assert changed.matcher is not original.matcher


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quality_matching_can_prefer_precise_lower_confidence_query_in_each_group(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU solver is exercised in the Modal smoke")
    original = HungarianMatcher(cost_class=2, cost_bbox=0, cost_giou=0)
    changed = PositionModulatedMatcher.from_stock(original)
    scores = [.6, .7, .6, .7]
    logits = torch.tensor([[[math.log(p/(1-p)), -3., -3.] for p in scores]], device=device)
    boxes = torch.tensor([[[.5, .5, .2, .2], [.5, .5, .6, .6]]*2], device=device)
    targets = [{"labels": torch.tensor([0], device=device), "boxes": torch.tensor([[.5, .5, .2, .2]], device=device)}]
    output = {"pred_logits": logits, "pred_boxes": boxes}
    assert original(output, targets, group_detr=2)[0][0].tolist() == [1, 3]
    assert changed(output, targets, group_detr=2)[0][0].tolist() == [0, 2]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repeated_class_targets_remain_distinct_and_empty_images_stay_empty(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU solver is exercised in the Modal smoke")
    changed = PositionModulatedMatcher(cost_class=2, cost_bbox=0, cost_giou=0)
    boxes = torch.tensor([[[.2, .2, .1, .1], [.8, .8, .1, .1]]]*2, device=device)
    output = {"pred_boxes": boxes, "pred_logits": torch.zeros(2, 2, 3, device=device)}
    targets = [{"labels": torch.tensor([0, 0], device=device), "boxes": boxes[0].flip(0)},
               {"labels": torch.empty(0, dtype=torch.int64, device=device), "boxes": torch.empty(0, 4, device=device)}]
    indices = changed(output, targets)
    assert indices[0][0].tolist() == [0, 1] and indices[0][1].tolist() == [1, 0]
    assert indices[1][0].numel() == indices[1][1].numel() == 0


def test_both_paths_support_all_unknown_empty_images_without_semantic_gradients():
    aware = CoverageSetCriterion.from_stock(stock(2), torch.zeros(2, 2, dtype=torch.bool))
    changed = with_stable_assignment(aware, position_targets=True, position_matching=True)
    output, targets = fixture(2, True)
    losses = changed(output, targets)
    backward(losses)
    for head in heads(output):
        assert torch.count_nonzero(head["pred_logits"].grad[..., :2]) == 0
        assert torch.count_nonzero(head["pred_logits"].grad[..., 2]) > 0
        assert torch.isfinite(head["pred_logits"].grad).all()


@pytest.mark.parametrize("kind", ["pseudo_weight", "varifocal", "alpha3", "padded", "pseudo_target"])
def test_unsupported_supervision_fails_closed(kind):
    original = CoverageSetCriterion.from_stock(stock(), torch.ones(2, 2, dtype=torch.bool))
    if kind == "pseudo_weight":
        original.pseudo_box_weight = .1
    elif kind == "varifocal":
        original.use_varifocal_loss = True
    elif kind == "alpha3":
        original = with_power_localization(original, 3)
    if kind in ("pseudo_weight", "varifocal", "alpha3"):
        with pytest.raises(ValueError, match="human-only"):
            with_stable_assignment(original, position_targets=True, position_matching=True)
    else:
        changed = with_stable_assignment(original, position_targets=True, position_matching=True)
        output, targets = fixture()
        if kind == "padded":
            targets[0]["valid"] = torch.tensor([True])
        else:
            targets[0]["is_pseudo"] = torch.tensor([True])
        with pytest.raises(ValueError):
            changed(output, targets)


def test_nonfinite_matching_and_reserved_targets_fail_closed():
    changed = PositionModulatedMatcher()
    output, targets = fixture()
    targets[0]["labels"] = torch.tensor([2])
    with pytest.raises(ValueError, match="reserved"):
        changed(output, targets)
    targets[0]["labels"] = torch.tensor([0])
    with torch.no_grad():
        output["pred_boxes"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        changed(output, targets)
