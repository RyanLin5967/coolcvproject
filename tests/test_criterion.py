import copy

import pytest
import torch

pytest.importorskip("rfdetr")
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher

from coveragecv.training.criterion import CoverageSetCriterion


def stock(groups=1):
    return SetCriterion(num_classes=3, matcher=HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2),
                        weight_dict={"loss_ce": 2, "loss_bbox": 5, "loss_giou": 2}, focal_alpha=0.25,
                        losses=["labels", "boxes", "cardinality"], group_detr=groups, ia_bce_loss=True)


def fixtures(groups=1, empty=False):
    torch.manual_seed(33)
    def head():
        return {"pred_logits": torch.randn(2, 6*groups, 3, requires_grad=True),
                "pred_boxes": (torch.rand(2, 6*groups, 4)*0.5+0.2).requires_grad_()}
    outputs = {**head(), "aux_outputs": [head(), head()], "enc_outputs": head()}
    targets = [{"image_id": torch.tensor([i]), "labels": torch.tensor([i], dtype=torch.int64),
                "boxes": torch.tensor([[.5, .5, .25, .3]])} for i in range(2)]
    if empty:
        for t in targets:
            t["labels"] = torch.empty(0, dtype=torch.int64)
            t["boxes"] = torch.empty((0, 4))
    return outputs, targets


def leaves(outputs):
    for key in ("pred_logits", "pred_boxes"):
        yield outputs[key]
    for out in outputs["aux_outputs"]+[outputs["enc_outputs"]]:
        yield from (out["pred_logits"], out["pred_boxes"])


@pytest.mark.parametrize("groups,empty", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("override", [None, 7.0])
def test_exhaustive_matches_stock_all_branches_and_gradients(groups, empty, training, override):
    original = stock(groups).train(training)
    aware = CoverageSetCriterion.from_stock(original, torch.ones(2, 2, dtype=torch.bool))
    out1, targets = fixtures(groups, empty)
    out2 = copy.deepcopy(out1)
    normalizer = None if override is None else torch.tensor(override)
    a, b = original(out1, targets, normalizer), aware(out2, targets, normalizer)
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-6, atol=1e-7)
    sum(v for k, v in a.items() if k.startswith("loss_")).backward()
    sum(v for k, v in b.items() if k.startswith("loss_")).backward()
    for x, y in zip(leaves(out1), leaves(out2)):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-6, atol=1e-7)
    assert all("_coverage_allowed" not in t for t in targets)


def test_unknown_cells_zero_gradient_but_positive_and_reserved_keep_stock():
    original = stock()
    aware = CoverageSetCriterion.from_stock(original, torch.tensor([[True, False], [False, False]]))
    out1, targets = fixtures()
    out2 = copy.deepcopy(out1)
    a, b = original(out1, targets), aware(out2, targets)
    sum(v for k, v in a.items() if k.startswith("loss_ce")).backward()
    sum(v for k, v in b.items() if k.startswith("loss_ce")).backward()
    for h1, h2 in zip([out1]+out1["aux_outputs"]+[out1["enc_outputs"]],
                       [out2]+out2["aux_outputs"]+[out2["enc_outputs"]]):
        assert torch.count_nonzero(h2["pred_logits"].grad[0, :, 1]) == 0
        torch.testing.assert_close(h1["pred_logits"].grad[0, :, 0], h2["pred_logits"].grad[0, :, 0])
        torch.testing.assert_close(h1["pred_logits"].grad[:, :, 2], h2["pred_logits"].grad[:, :, 2])
        indices = original.matcher(h1, targets, group_detr=1)
        q = indices[1][0]
        torch.testing.assert_close(h1["pred_logits"].grad[1, q, 1], h2["pred_logits"].grad[1, q, 1])
        assert torch.count_nonzero(h2["pred_logits"].grad[1, :, 0]) == 0
        assert torch.count_nonzero(h2["pred_logits"].grad[1, :, 1]) == len(q)


def test_all_unknown_empty_semantic_gradient_zero_not_reserved():
    aware = CoverageSetCriterion.from_stock(stock(), torch.zeros(2, 2, dtype=torch.bool))
    out, targets = fixtures(empty=True)
    losses = aware(out, targets)
    losses["loss_ce"].backward()
    assert torch.count_nonzero(out["pred_logits"].grad[:, :, :2]) == 0
    assert torch.count_nonzero(out["pred_logits"].grad[:, :, 2]) > 0


def test_image_identity_and_configuration_fail_closed():
    aware = CoverageSetCriterion.from_stock(stock(), torch.ones(2, 2, dtype=torch.bool),
                                            valid_ids=torch.tensor([True, False]))
    out, targets = fixtures()
    with pytest.raises(ValueError, match="verified learner"):
        aware(out, targets)
    targets[1]["image_id"] = torch.tensor([3])
    with pytest.raises(ValueError, match="index space"):
        aware(out, targets)
    targets[0]["valid"] = torch.tensor([True])
    with pytest.raises(ValueError, match="padded"):
        aware(out, targets)
    other = stock()
    other.ia_bce_loss = False
    with pytest.raises(ValueError, match="IoU-aware"):
        CoverageSetCriterion.from_stock(other, torch.ones(2, 2, dtype=torch.bool))
