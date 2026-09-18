"""Preserve pseudo-label identity through the upstream instance transforms."""
import torch


class PreservePseudoProvenance:
    def __init__(self, prepare):
        self.prepare = prepare

    def __call__(self, image, target):
        annotations = [a for a in target["annotations"] if not a.get("iscrowd", 0)]
        width, height = image.size
        boxes = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        flags = torch.tensor([bool(a.get("is_pseudo", False)) for a in annotations], dtype=torch.bool)[keep]
        image, prepared = self.prepare(image, target)
        if not torch.equal(boxes[keep], prepared["boxes"]) or len(flags) != len(prepared["labels"]):
            raise ValueError("upstream COCO conversion changed instance order; pseudo provenance is unsafe")
        prepared["is_pseudo"] = flags
        # RF-DETR 1.10.1's torchvision transforms filter arbitrary instance tensors
        # with the same crop keep indices as boxes. The preservation is integration-tested.
        return image, prepared
