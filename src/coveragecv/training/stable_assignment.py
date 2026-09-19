"""Observed-label positional supervision and geometry-modulated matching.

Position loss follows RF-DETR 1.10.1's exact position_supervised_loss branch,
adapted under the existing coverage mask (upstream Apache-2.0, Roboflow).
Its target is normalized IoU, not the squared-IoU example in Stable-DINO.
Matching follows Stable-DINO Eq. 4 with f2(s)=sqrt(s), s=(GIoU+1)/2:
https://arxiv.org/html/2304.04742v1#S2.SS3
https://github.com/IDEA-Research/Stable-DINO

The implementation keeps a distinct score for every query/target pair, including
multiple targets of the same class. No denoising labels, inference adapter,
checkpoint parameters, or unobserved training targets are introduced.
"""
import copy
import importlib.metadata
import inspect

import torch
import torch.nn.functional as F
from rfdetr.models import _assignment
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.utilities import box_ops
from rfdetr.utilities.misc import accuracy

from coveragecv.training.criterion import CoverageSetCriterion
from coveragecv.training.power_localization import PowerCoverageCriterion, PowerStandardCriterion

POLICY = {
    "position_target": "matched IoU / (maximum matched IoU across query/class cells per image + 1e-8)",
    "classification": "stock RF-DETR1.10.1 position BCE * abs(target-sigmoid(logit))**2 with stock focal_alpha",
    "matching_probability": "sigmoid(target_class_logit) * sqrt(clamp((pairwise_GIoU+1)/2,0,1))",
    "matching_cost": "original focal class cost on modulated probability + unchanged L1/GIoU costs",
    "matching_groups": "one-to-one Hungarian assignment independently per Group-DETR group",
    "coverage": "existing image/class negative eligibility OR observed matched positive OR declared matched-object exclusivity",
    "geometry_loss": "unchanged; only alpha1 and ordinary human-observed boxes accepted",
    "inference": "unchanged checkpoint architecture and stock postprocessing",
    "references": ["https://arxiv.org/html/2304.04742v1", "https://github.com/IDEA-Research/Stable-DINO"],
}


class PositionModulatedMatcher(HungarianMatcher):
    """Paper's pairwise modulation with the pinned stock grouped solver."""

    @classmethod
    def from_stock(cls, original):
        if type(original) is not HungarianMatcher:
            raise ValueError("Stable assignment requires the unmodified pinned Hungarian matcher")
        if original.num_keypoints_per_class or any(getattr(original, name) for name in (
                "keypoint_l1_loss_coef", "keypoint_findable_loss_coef", "keypoint_visible_loss_coef",
                "keypoint_nll_loss_coef")):
            raise ValueError("Stable assignment supports detection boxes only")
        return cls(cost_class=original.cost_class, cost_bbox=original.cost_bbox, cost_giou=original.cost_giou,
                   focal_alpha=original.focal_alpha, mask_point_sample_ratio=original.mask_point_sample_ratio,
                   cost_mask_ce=original.cost_mask_ce, cost_mask_dice=original.cost_mask_dice).train(original.training)

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1, target_side_safety=None):
        del target_side_safety  # The custom pairwise path validates its own inputs.
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        if (logits.ndim != 3 or boxes.shape != (*logits.shape[:2], 4) or len(targets) != logits.shape[0]
                or not targets or not isinstance(group_detr, int) or group_detr < 1
                or logits.shape[1] % group_detr or logits.device != boxes.device):
            raise ValueError("Invalid grouped detector dimensions")
        if any(name in outputs for name in ("pred_masks", "pred_keypoints")):
            raise ValueError("Stable assignment supports detection boxes only")
        if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all() or (boxes[..., 2:] < 0).any():
            raise ValueError("Nonfinite or negative detector geometry")
        costs, sizes = [], []
        for index, target in enumerate(targets):
            if any(name in target for name in ("masks", "keypoints", "valid")):
                raise ValueError("Masks, keypoints and padded ground truth are unsupported")
            labels, truth = target["labels"], target["boxes"]
            if (labels.dtype != torch.int64 or labels.ndim != 1 or truth.shape != (len(labels), 4)
                    or labels.device != logits.device or truth.device != boxes.device
                    or truth.dtype != boxes.dtype):
                raise ValueError("Target layout/device/dtype differs from detector")
            if (labels < 0).any() or (labels >= logits.shape[-1]-1).any():
                raise ValueError("Observed targets must be semantic classes, never the reserved slot")
            if not torch.isfinite(truth).all() or (truth[:, 2:] <= 0).any():
                raise ValueError("Observed boxes must have finite positive dimensions")
            if "is_pseudo" in target and target["is_pseudo"].any():
                raise ValueError("Stable assignment requires human-only observed labels")
            sizes.append(len(labels))
            if not len(labels):
                costs.append(boxes.new_empty((logits.shape[1], 0)))
                continue
            giou = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(boxes[index]),
                                              box_ops.box_cxcywh_to_xyxy(truth))
            # Do NOT scatter into class columns: repeated-class objects have distinct geometry.
            probability = logits[index].sigmoid()[:, labels] * ((giou+1)/2).clamp(0, 1).sqrt()
            positive = self.focal_alpha * (1-probability).pow(2) * -(probability+1e-8).log()
            negative = (1-self.focal_alpha) * probability.pow(2) * -(1-probability+1e-8).log()
            cost = (self.cost_class*(positive-negative)
                    + self.cost_bbox*torch.cdist(boxes[index], truth, p=1) - self.cost_giou*giou)
            costs.append(cost)
        if not sum(sizes):
            return [(torch.empty(0, dtype=torch.int64, device=boxes.device),
                     torch.empty(0, dtype=torch.int64, device=boxes.device)) for _ in targets]
        compact = torch.cat(costs, dim=1).float()
        if not torch.isfinite(compact).all():
            raise ValueError("Nonfinite position-modulated assignment costs")
        if compact.is_cuda:
            return _assignment.assign_many_bucketed([compact], sizes, group_detr)[0]
        return self._assign_compact_cost_matrix(compact.cpu(), sizes, group_detr)


class _StablePositionMixin:
    stable_position_targets = False
    stable_position_matching = False

    def forward(self, outputs, targets, num_boxes=None):
        for target in targets:
            if any(name in target for name in ("valid", "masks", "keypoints")):
                raise ValueError("Stable assignment supports unpadded detection targets only")
            if "is_pseudo" in target:
                pseudo = target["is_pseudo"]
                if pseudo.dtype != torch.bool or pseudo.shape != target["labels"].shape or pseudo.any():
                    raise ValueError("Stable assignment requires human-only aligned provenance")
        return super().forward(outputs, targets, num_boxes=num_boxes)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, matched_targets=None):
        if not self.stable_position_targets:
            return super().loss_labels(outputs, targets, indices, num_boxes, log=log, matched_targets=matched_targets)
        logits = outputs["pred_logits"]
        if logits.shape[-1] != self.num_classes:
            raise ValueError("Model output differs from ontology")
        if matched_targets is None:
            idx = self._get_src_permutation_idx(indices)
            labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
            boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
        else:
            idx, labels, boxes = matched_targets.source_indices, matched_targets.labels, matched_targets.boxes
        if (labels < 0).any() or (labels >= self.num_classes-1).any():
            raise ValueError("Matched labels must be semantic classes")
        quality, _ = box_ops.elementwise_box_iou(
            box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][idx].detach()), box_ops.box_cxcywh_to_xyxy(boxes))
        soft = torch.zeros_like(logits)
        positive_indices = (*idx, labels)
        soft[positive_indices] = quality.clone().detach().to(soft.dtype)
        soft = soft / (soft.view(soft.shape[0], -1, 1).amax(1, True)+1e-8)
        probability = logits.sigmoid()
        cells = F.binary_cross_entropy_with_logits(logits, soft, reduction="none") * (soft-probability).abs().pow(2)
        if self.focal_alpha >= 0:
            cells = cells * (self.focal_alpha*(soft > 0).float() + (1-self.focal_alpha)*(soft <= 0).float())
        if isinstance(self, CoverageSetCriterion):
            semantic = torch.stack([t["_coverage_allowed"] for t in targets]).to(logits.device)
            allowed = torch.cat((semantic, torch.ones((len(targets), 1), dtype=torch.bool, device=logits.device)), 1)
            eligible = allowed[:, None, :].expand_as(logits).clone()
            if self.exclusive_groups:
                eligible[idx] |= self.mutually_exclusive[labels]
            eligible[positive_indices] = True
            cells = cells * eligible
        # Preserve stock reduction order as well as its group/image denominator.
        losses = {"loss_ce": cells.mean(1).sum()/num_boxes*logits.shape[1]}
        if log:
            losses["class_error"] = 100-accuracy(logits[idx], labels)[0]
        return losses


class StableCoverageCriterion(_StablePositionMixin, CoverageSetCriterion):
    pass


class StableStandardCriterion(_StablePositionMixin, SetCriterion):
    pass


class StablePowerCoverageCriterion(_StablePositionMixin, PowerCoverageCriterion):
    pass


class StablePowerStandardCriterion(_StablePositionMixin, PowerStandardCriterion):
    pass


def with_stable_assignment(criterion, *, position_targets: bool, position_matching: bool):
    """Return an independent bounded adapter, retaining exact original box losses."""
    if importlib.metadata.version("rfdetr") != "1.10.1":
        raise ValueError("Stable assignment supports exactly RF-DETR1.10.1")
    if not isinstance(position_targets, bool) or not isinstance(position_matching, bool):
        raise TypeError("Ablation flags must be explicit Booleans")
    supported = {SetCriterion: StableStandardCriterion, CoverageSetCriterion: StableCoverageCriterion,
                 PowerStandardCriterion: StablePowerStandardCriterion, PowerCoverageCriterion: StablePowerCoverageCriterion}
    if type(criterion) not in supported:
        raise ValueError("Unknown criterion wrapper; refusing to discard custom behavior")
    if (criterion.use_varifocal_loss or criterion.use_position_supervised_loss or not criterion.ia_bce_loss
            or criterion.num_keypoints_per_class or any(x not in ("labels", "boxes", "cardinality") for x in criterion.losses)
            or getattr(criterion, "pseudo_box_weight", 1.) != 1. or getattr(criterion, "alpha", 1) != 1):
        raise ValueError("Stable assignment requires the ordinary human-only alpha1 IoU-aware detection path")
    if type(criterion.matcher) is not HungarianMatcher:
        raise ValueError("Unknown matcher; refusing to discard custom assignment behavior")
    kwargs = {name: copy.deepcopy(getattr(criterion, name))
              for name in inspect.signature(SetCriterion.__init__).parameters if name != "self"}
    cls = supported[type(criterion)]
    if isinstance(criterion, CoverageSetCriterion):
        kwargs.update(negative_allowed=criterion.negative_allowed, valid_ids=criterion.valid_ids,
                      contract_digest=criterion.contract_digest, pseudo_box_weight=1.,
                      exclusive_groups=criterion.exclusive_groups)
    result = cls(**kwargs)
    if hasattr(criterion, "alpha"):
        result.alpha = criterion.alpha
    result.stable_position_targets = position_targets
    result.stable_position_matching = position_matching
    if position_targets:
        result.ia_bce_loss, result.use_position_supervised_loss = False, True
    if position_matching:
        result.matcher = PositionModulatedMatcher.from_stock(criterion.matcher)
    result.train(criterion.training)
    return result
