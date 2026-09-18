import pytest

pytest.importorskip("rfdetr")
from coveragecv.training.pseudo import consensus_boxes


def prediction(label=0, score=.9, box=(0., 0., 10., 10.)):
    return {"label": label, "score": score, "box": list(box)}


def test_pseudo_requires_unknown_coverage_both_views_and_confidence():
    a = [prediction(0), prediction(1), prediction(2), prediction(3)]
    states = ["unknown", "positive_only", "exhaustive", "verified_absent"]
    # Separate coordinates avoid geometric conflict across classes.
    for i, p in enumerate(a):
        p["box"] = [20.*i, 0., 20.*i+10, 10.]
    result = consensus_boxes(a, a, [], states)
    assert [p["label"] for p in result] == [0, 1]
    assert not consensus_boxes(a, [], [], states)
    assert not consensus_boxes([prediction(score=.65)], [prediction()], [], ["unknown"])
    assert not consensus_boxes([prediction()], [prediction(box=(30., 0., 40., 10.))], [], ["unknown"])


def test_observed_labels_win_and_duplicate_teacher_boxes_are_suppressed():
    p = prediction()
    assert not consensus_boxes([p], [p], [p["box"]], ["unknown"])
    assert len(consensus_boxes([p, p], [p, p], [], ["unknown"])) == 1
    assert not consensus_boxes([prediction(1)], [prediction(1)], [], ["unknown"])


def test_pseudo_identity_survives_real_upstream_crop_and_flip():
    import torch
    from PIL import Image
    from rfdetr.datasets._torchvision import RandomHorizontalFlip, crop
    from rfdetr.datasets.coco import ConvertCoco

    from coveragecv.training.provenance import PreservePseudoProvenance
    image = Image.new("RGB", (100, 100))
    target = {"image_id": 1, "annotations": [
        {"bbox": [1, 1, 10, 10], "category_id": 1, "area": 100},
        {"bbox": [60, 60, 10, 10], "category_id": 2, "area": 100, "is_pseudo": True}]}
    prepare = PreservePseudoProvenance(ConvertCoco(include_masks=False, cat2label={1: 0, 2: 1}))
    image, target = prepare(image, target)
    assert target["is_pseudo"].tolist() == [False, True]
    # Keep only the pseudo instance. Provenance must follow exactly the same filtering.
    image, target = crop(image, target, 50, 50, 50, 50)
    assert target["labels"].tolist() == [1]
    assert target["is_pseudo"].tolist() == [True]
    image, target = RandomHorizontalFlip(p=1.)(image, target)
    assert target["is_pseudo"].tolist() == [True]
    assert target["is_pseudo"].dtype == torch.bool
