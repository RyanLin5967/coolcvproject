"""Coverage-aware RF-DETR 1.10.1 IoU-aware BCE.

The unreduced loss expression is adapted from Roboflow's SetCriterion.loss_labels,
commit e3fc28795f2a4303069c6b72e80431e5ed716030, under Apache-2.0.
See THIRD_PARTY_NOTICES.md. Matching, normalization, box and auxiliary dispatch
remain upstream implementations.
"""
import importlib.metadata
import inspect
from pathlib import Path

import torch
import torch.nn.functional as F
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.math import accuracy
from rfdetr.utilities import box_ops

from coveragecv.artifacts import read_json, verify
from coveragecv.compiler import _jsonl
from coveragecv.schema import NEGATIVE_ALLOWED, DiagnosticError


def coverage_table(view: Path):
    manifest = verify(view)
    if manifest["kind"] != "training_view":
        raise DiagnosticError("INVALID_ARTIFACT_KIND", "criterion requires a verified training view")
    classes = read_json(view / "ontology.json")["classes"]
    table = torch.zeros((manifest["index_space_size"], len(classes)), dtype=torch.bool)
    valid = torch.zeros(manifest["index_space_size"], dtype=torch.bool)
    for row in _jsonl(view / "coverage.jsonl"):
        iid = row["image_id"]
        if not 0 <= iid < len(table) or valid[iid] or len(row["states"]) != len(classes):
            raise DiagnosticError("INVALID_COVERAGE_TABLE", "invalid or repeated coverage row")
        table[iid] = torch.tensor([s in NEGATIVE_ALLOWED for s in row["states"]])
        valid[iid] = True
    return table, valid, manifest


class CoverageSetCriterion(SetCriterion):
    """Apply image/class eligibility to every full unreduced classification cell."""

    def __init__(self, *args, negative_allowed: torch.Tensor, valid_ids=None, contract_digest="synthetic", **kwargs):
        if importlib.metadata.version("rfdetr") != "1.10.1":
            raise ValueError("this adapter supports exactly rfdetr==1.10.1")
        super().__init__(*args, **kwargs)
        if not self.ia_bce_loss or self.use_varifocal_loss or self.use_position_supervised_loss:
            raise ValueError("only the reviewed IoU-aware BCE detection path is supported")
        if any(loss not in ("labels", "boxes", "cardinality") for loss in self.losses):
            raise ValueError("segmentation/keypoint loss is not supported")
        if negative_allowed.dtype != torch.bool or negative_allowed.ndim != 2:
            raise ValueError("coverage must be a Boolean image x semantic-class table")
        if negative_allowed.shape[1] != self.num_classes-1:
            raise ValueError("expected K semantic coverage columns and K+1 output channels")
        self.register_buffer("negative_allowed", negative_allowed.clone(), persistent=False)
        self.register_buffer("valid_ids", (torch.ones(len(negative_allowed), dtype=torch.bool)
                                           if valid_ids is None else valid_ids.clone()), persistent=False)
        self.contract_digest = contract_digest

    @classmethod
    def from_stock(cls, stock, negative_allowed, valid_ids=None, contract_digest="synthetic"):
        fields = inspect.signature(SetCriterion.__init__).parameters
        kwargs = {k: getattr(stock, k) for k in fields if k != "self"}
        custom = cls(**kwargs, negative_allowed=negative_allowed, valid_ids=valid_ids,
                     contract_digest=contract_digest)
        custom.train(stock.training)
        return custom

    def forward(self, outputs, targets, num_boxes=None):
        if any("valid" in target for target in targets):
            raise ValueError("padded ground-truth validity is unsupported in RF-DETR 1.10.1")
        if not targets or any("image_id" not in target for target in targets):
            raise ValueError("each target must retain its verified image_id")
        ids = torch.cat([t["image_id"].reshape(-1) for t in targets]).to(self.negative_allowed.device)
        if ids.dtype != torch.int64 or ids.numel() != len(targets):
            raise ValueError("image_id must contain one int64 value per target")
        # One batched validity check, never a per-image CUDA .item() loop.
        if torch.any(ids < 0) or torch.any(ids >= len(self.negative_allowed)):
            raise ValueError("image_id is outside the artifact index space")
        if not torch.all(self.valid_ids[ids]):
            raise ValueError("image_id does not belong to the verified learner view")
        allowed = self.negative_allowed[ids]
        enriched = [{**target, "_coverage_allowed": row} for target, row in zip(targets, allowed)]
        return super().forward(outputs, enriched, num_boxes=num_boxes)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, matched_targets=None):
        logits = outputs["pred_logits"]
        if logits.shape[-1] != self.num_classes:
            raise ValueError("model output dimension differs from the coverage ontology")
        if matched_targets is None:
            idx = self._get_src_permutation_idx(indices)
            labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
            boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        else:
            idx, labels, boxes = matched_targets.source_indices, matched_targets.labels, matched_targets.boxes
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes-1):
            raise ValueError("observed labels must refer to semantic classes, never the reserved slot")
        iou, _ = box_ops.elementwise_box_iou(
            box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"][idx].detach()),
            box_ops.box_cxcywh_to_xyxy(boxes),
        )
        probability = logits.sigmoid()
        positive_weights = torch.zeros_like(logits)
        negative_weights = probability**2
        positive_indices = (*idx, labels)
        soft_target = torch.clamp(
            probability[positive_indices].pow(self.focal_alpha)*iou.clone().detach().pow(1-self.focal_alpha),
            0.01,
        ).detach()
        positive_weights[positive_indices] = soft_target.to(positive_weights.dtype)
        negative_weights[positive_indices] = 1-soft_target.to(negative_weights.dtype)
        cells = negative_weights*logits - F.logsigmoid(logits)*(positive_weights+negative_weights)
        semantic_allowed = torch.stack([t["_coverage_allowed"] for t in targets]).to(logits.device)
        allowed = torch.cat((semantic_allowed, torch.ones((len(targets), 1), dtype=torch.bool,
                                                         device=logits.device)), dim=1)
        eligible = allowed[:, None, :].expand_as(logits).clone()
        eligible[positive_indices] = True
        losses = {"loss_ce": (cells*eligible).sum()/num_boxes}
        if log:
            losses["class_error"] = 100-accuracy(logits[idx], labels)[0]
        return losses
