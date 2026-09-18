import pytest

pytest.importorskip("rfdetr")
from coveragecv.error_analysis import diagnostic_metrics


def test_errors_distinguish_confusion_duplicate_and_localization():
    reference = {"images": [{"id": 1, "width": 100, "height": 100}],
                 "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                 "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                  "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                                 {"id": 2, "image_id": 1, "category_id": 2,
                                  "bbox": [30, 30, 10, 10], "area": 100, "iscrowd": 0}]}
    rows = [(1, [0, 0, 10, 10], .95), (1, [0, 0, 10, 10], .9), (1, [30, 30, 10, 10], .8),
            (2, [35, 30, 10, 10], .7), (2, [80, 80, 10, 10], .6)]
    evaluation = {"predictions": [{"image_id": 1, "category_id": c, "bbox": b, "score": s}
                                  for c, b, s in rows]}
    result = diagnostic_metrics(evaluation, reference)
    assert result["error_counts"] == {"true_positive": 1, "duplicate": 1, "class_confusion": 1,
                                      "localization": 1, "background_or_bad_localization": 1, "missed": 1}
    assert result["confusions"] == [{"ground_category": 2, "predicted_category": 1, "count": 1}]
    assert len(result["AP_by_IoU"]) == 10
