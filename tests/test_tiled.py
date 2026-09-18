"""Geometry and deterministic merging checks without model downloads."""
import pytest

pytest.importorskip("rfdetr")
from coveragecv.training.tiled import map_tile_box, merge_predictions, tile_windows


def prediction(box, *, score=.8, image=1, category=1):
    return {"image_id": image, "category_id": category, "bbox": box, "score": score}


def test_four_tiles_cover_640_with_128_overlap():
    assert tile_windows(640, 640) == [(0, 0, 384, 384), (256, 0, 640, 384),
                                     (0, 256, 384, 640), (256, 256, 640, 640)]


def test_uneven_dimensions_and_small_image_have_full_coverage():
    assert tile_windows(100, 200) == [(0, 0, 100, 200)]
    windows = tile_windows(937, 401)
    assert len(set(windows)) == len(windows)
    assert max(w[2] for w in windows) == 937
    assert max(w[3] for w in windows) == 401
    for x in range(937):
        for y in (0, 255, 383, 400):
            assert any(a <= x < c and b <= y < d for a, b, c, d in windows)


@pytest.mark.parametrize("dimensions", [(0, 640), (-1, 20)])
def test_invalid_dimensions_rejected(dimensions):
    with pytest.raises(ValueError):
        tile_windows(*dimensions)


def test_local_pixel_offsets_are_exact():
    assert map_tile_box([10, 20, 40, 70], (256, 256, 640, 640), (640, 640)) == [266, 276, 30, 50]


@pytest.mark.parametrize("box", [[0, 20, 40, 70], [10, 1, 40, 70], [10, 20, 383, 70], [10, 20, 40, 382]])
def test_internal_edge_fragments_rejected(box):
    assert map_tile_box(box, (128, 128, 512, 512), (640, 640)) is None


def test_real_image_boundaries_are_preserved_and_clipped():
    assert map_tile_box([-1, -2, 30, 50], (0, 0, 384, 384), (640, 640)) == [0., 0., 30., 50.]
    assert map_tile_box([100, 100, 387, 389], (256, 256, 640, 640), (640, 640)) == [356, 356, 284., 284.]


def test_nms_suppresses_only_same_image_and_class_preserves_scores():
    rows = [prediction([10, 20, 30, 40], score=.7), prediction([10, 20, 30, 40], score=.8),
            prediction([10, 20, 30, 40], image=2), prediction([10, 20, 30, 40], category=2)]
    merged = merge_predictions(rows)
    assert len(merged) == 3
    assert {p["score"] for p in merged} == {.8}
    assert rows[0]["score"] == .7


def test_equal_score_nms_retains_first_input_and_is_repeatable():
    first, second = prediction([10, 20, 30, 40]), prediction([11, 20, 30, 40])
    assert merge_predictions([first, second]) == [first]
    assert merge_predictions([first, second]) == merge_predictions([first, second])


def test_floor_and_zero_area_boxes_cannot_enter_nms():
    assert merge_predictions([prediction([0, 0, 0, 4]), prediction([0, 0, 4, 4], score=.0009)]) == []
    assert len(merge_predictions([prediction([0, 0, 4, 4], score=.001)])) == 1


def test_nonfinite_or_negative_geometry_rejected():
    for box in ([0, 0, -1, 3], [float("nan"), 0, 1, 3]):
        with pytest.raises(ValueError):
            merge_predictions([prediction(box)])


def test_untruncated_duplicate_tile_maps_to_same_full_frame_box():
    from_left = map_tile_box([300, 40, 330, 80], (0, 0, 384, 384), (640, 640))
    from_right = map_tile_box([44, 40, 74, 80], (256, 0, 640, 384), (640, 640))
    assert from_left == from_right
    assert len(merge_predictions([prediction(from_left), prediction(from_right)])) == 1
