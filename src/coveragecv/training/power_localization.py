"""Experimental Alpha-GIoU, separate from the validated default coverage adapter.

Formula: He et al., Alpha-IoU, NeurIPS 2021 (alpha=3 recommendation).
https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html
The implementation below is original; alpha=1 recovers ordinary GIoU.
"""
import inspect

import torch
from rfdetr.models.criterion import SetCriterion
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy

from coveragecv.training.criterion import CoverageSetCriterion


def power_giou_loss(first, second, alpha=3):
    if alpha not in (1, 3) or first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("Expected aligned xyxy boxes and predeclared alpha 1 or 3")
    intersection_wh = (torch.minimum(first[..., 2:], second[..., 2:])
                       - torch.maximum(first[..., :2], second[..., :2])).clamp(min=0)
    intersection = intersection_wh.prod(-1)
    union = ((first[..., 2:]-first[..., :2]).prod(-1)
             +(second[..., 2:]-second[..., :2]).prod(-1)-intersection)
    outer_wh = torch.maximum(first[..., 2:], second[..., 2:])-torch.minimum(first[..., :2], second[..., :2])
    enclosure = outer_wh.prod(-1)
    eps = torch.finfo(first.dtype).tiny
    overlap = (intersection/union.clamp(min=eps)).clamp(0, 1)
    penalty = ((enclosure-union)/enclosure.clamp(min=eps)).clamp(0, 1)
    return 1-overlap.pow(alpha)+penalty.pow(alpha)


class PowerBoxMixin:
    alpha = 3

    def loss_boxes(self, outputs, targets, indices, num_boxes, matched_targets=None):
        if getattr(self, "pseudo_box_weight", 1.) != 1.:
            raise ValueError("Power localization is restricted to human-only experiments")
        losses = super().loss_boxes(outputs, targets, indices, num_boxes, matched_targets=matched_targets)
        idx = self._get_src_permutation_idx(indices) if matched_targets is None else matched_targets.source_indices
        predicted = outputs["pred_boxes"][idx]
        target = (torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
                  if matched_targets is None else matched_targets.boxes)
        losses["loss_giou"] = power_giou_loss(box_cxcywh_to_xyxy(predicted),
                                             box_cxcywh_to_xyxy(target), self.alpha).sum()/num_boxes
        return losses


class PowerCoverageCriterion(PowerBoxMixin, CoverageSetCriterion):
    pass


class PowerStandardCriterion(PowerBoxMixin, SetCriterion):
    pass


def with_power_localization(criterion, alpha):
    if alpha not in (1, 3):
        raise ValueError("Only the frozen alpha=1 control and alpha=3 treatment are allowed")
    if getattr(criterion, "pseudo_box_weight", 1.) != 1.:
        raise ValueError("Power localization is restricted to human-only experiments")
    if isinstance(criterion, CoverageSetCriterion):
        result = PowerCoverageCriterion.from_stock(criterion, criterion.negative_allowed, criterion.valid_ids,
                                                   contract_digest=criterion.contract_digest,
                                                   exclusive_groups=criterion.exclusive_groups)
    else:
        kwargs = {k: getattr(criterion, k) for k in inspect.signature(SetCriterion.__init__).parameters if k != "self"}
        result = PowerStandardCriterion(**kwargs)
    result.alpha = alpha
    result.train(criterion.training)
    return result
