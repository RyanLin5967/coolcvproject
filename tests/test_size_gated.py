"""A train-only routing rule must not consume evaluation labels or rewrite boxes."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("rfdetr")
from coveragecv.artifacts import publish, stage, write_json

spec = importlib.util.spec_from_file_location(
    "size_gated", Path(__file__).resolve().parents[1] / "scripts/evaluate_size_gated.py")
size_gated = importlib.util.module_from_spec(spec)
spec.loader.exec_module(size_gated)


def view(tmp_path, *, pseudo=False):
    output = tmp_path / "views"
    staging = stage(output)
    write_json(staging / "train/_annotations.coco.json", {
        "images": [{"id": 1, "width": 640, "height": 640}],
        "categories": [{"id": 1, "name": "small"}, {"id": 2, "name": "large"},
                       {"id": 3, "name": "unobserved"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 20],
                         "is_pseudo": pseudo},
                        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 130, 130]}]})
    (staging / "valid").mkdir()
    (staging / "valid/_annotations.coco.json").write_text("must never be parsed as labels")
    return publish(staging, output, {"kind": "training_view", "schema_version": 1})


def test_gate_uses_only_scaled_train_sizes_and_unobserved_class_falls_back(tmp_path):
    rule = size_gated.training_rule(view(tmp_path))
    assert rule["tiled_category_ids"] == [1]
    assert [row["median_longest_side_at512"] for row in rule["class_statistics"]] == [80, 104, None]
    assert rule["useful"]


def test_pseudo_labels_cannot_choose_gate_classes(tmp_path):
    with pytest.raises(ValueError, match="pseudo"):
        size_gated.training_rule(view(tmp_path, pseudo=True))


def test_combination_preserves_exact_full_and_tiled_rows_for_disjoint_classes():
    full = [{"category_id": 1, "score": .5, "bbox": [1, 2, 3, 4]},
            {"category_id": 2, "score": .8, "bbox": [5, 6, 7, 8]}]
    tiled = [{"category_id": 1, "score": .7, "bbox": [9, 10, 11, 12]},
             {"category_id": 2, "score": .9, "bbox": [13, 14, 15, 16]}]
    result = size_gated.combine_predictions(full, tiled, [1])
    assert result == [full[1], tiled[0]]
    assert full[0]["score"] == .5 and tiled[1]["score"] == .9
