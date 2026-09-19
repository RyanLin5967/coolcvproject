"""Synthetic geometry contracts only; no local model or benchmark evaluation."""
import pytest

from coveragecv.training.geometry_consistency import paired_geometry


def box(x, *, image=1, label=1, score=.9):
    return {"image_id": image, "category_id": label, "score": score, "bbox": [x, 10., 20., 30.]}


def test_pairing_changes_only_geometry_and_preserves_unmatched_predictions():
    original = [box(10), box(80), box(10, label=2), box(10, image=2), box(10, score=.01)]
    flipped = [box(12)]
    corrected, pairs = paired_geometry(original, flipped)
    assert pairs == 1
    assert len(corrected) == len(original)
    assert corrected[0]["bbox"] == [11., 10., 20., 30.]
    assert corrected[1:] == original[1:]
    assert original[0]["bbox"] == [10, 10., 20., 30.]  # never mutate baseline
    for a, b in zip(original, corrected, strict=True):
        assert {k: v for k, v in a.items() if k != "bbox"} == {k: v for k, v in b.items() if k != "bbox"}


def test_one_flip_prediction_cannot_adjust_two_original_predictions():
    corrected, pairs = paired_geometry([box(10), box(10)], [box(12)])
    assert pairs == 1
    assert corrected[0]["bbox"][0] == 11
    assert corrected[1]["bbox"][0] == 10


def test_low_agreement_and_wrong_class_do_not_change_geometry():
    original = [box(10)]
    corrected, count = paired_geometry(original, [box(15), box(10, label=2)])
    assert count == 0
    assert corrected == original


def test_invalid_geometry_fails_before_matching():
    malformed = box(10)
    malformed["bbox"][2] = -1
    with pytest.raises(ValueError, match="geometry"):
        paired_geometry([malformed], [])


def test_zero_area_stock_rows_are_preserved_but_never_paired():
    empty = box(10)
    empty["bbox"][2] = 0
    corrected, count = paired_geometry([empty, box(10)], [box(12)])
    assert count == 1
    assert corrected[0] == empty
    assert len(corrected) == 2
