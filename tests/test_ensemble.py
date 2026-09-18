import pytest

pytest.importorskip("ensemble_boxes")
from coveragecv.training.ensemble import fuse_predictions


def test_fusion_uses_geometry_and_class_without_annotations():
    images = [{"id": 1, "width": 100, "height": 50}, {"id": 2, "width": 100, "height": 50}]
    def box(x, label=1):
        return {"image_id": 1, "bbox": [x, 10, 20, 20], "category_id": label, "score": .9}
    predictions = fuse_predictions([[box(10), box(10, 2)], [box(12)]], images)
    same = next(p for p in predictions if p["category_id"] == 1)
    other = next(p for p in predictions if p["category_id"] == 2)
    assert same["bbox"] == pytest.approx([11, 10, 20, 20], abs=1e-4)
    assert same["score"] == pytest.approx(.9)
    assert other["score"] == pytest.approx(.45)
    assert {p["image_id"] for p in predictions} == {1}
    with pytest.raises(ValueError, match="unknown image"):
        fuse_predictions([[{**box(10), "image_id": 999}], []], images)
