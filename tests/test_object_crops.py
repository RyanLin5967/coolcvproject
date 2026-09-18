import copy
from collections import Counter

import pytest

pytest.importorskip("rfdetr")
import torch
from PIL import Image
from rfdetr.training import RFDETRDataModule

from coveragecv.artifacts import publish, stage, write_json
from coveragecv.training.criterion import coverage_table
from coveragecv.training.object_crops import build_crop_plan, install_object_crops
from coveragecv.training.runner import configs

CLASSES = ["helmet", "no-helmet", "no-vest", "person", "vest"]


def make_view(root, *, complete=False, pseudo=False, changed_image=False, empty=False):
    staging = stage(root)
    (staging / "train").mkdir()
    images = []
    for image_id in range(2):
        image = Image.new("RGB", (320, 240), (40+image_id*30, 100, 190))
        # Spatially nonconstant pixels make mismatched crops/flips observable.
        for x in range(320):
            image.putpixel((x, 100), (x % 256, 200, 30))
        if changed_image:
            image.putpixel((0, 0), (200, 50, 30))
        image.save(staging / "train" / f"{image_id}.png")
        images.append({"id": image_id, "file_name": f"{image_id}.png", "width": 320, "height": 240})
    annotations = [
        {"id": 1, "image_id": 0, "category_id": 1, "bbox": [135, 95, 20, 20], "area": 400},
        {"id": 2, "image_id": 0, "category_id": 2, "bbox": [160, 90, 16, 25], "area": 400},
        {"id": 3, "image_id": 0, "category_id": 3, "bbox": [0, 0, 10, 10], "area": 100},
        {"id": 4, "image_id": 0, "category_id": 4, "bbox": [250, 210, 20, 20], "area": 400},
        {"id": 5, "image_id": 0, "category_id": 5, "bbox": [160, 170, 10, 10], "area": 100},
    ]
    if pseudo:
        annotations[0]["is_pseudo"] = True
    if empty:
        annotations = []
    if complete:
        annotations.append({"id": 99, "image_id": 0, "category_id": 5, "bbox": [130, 90, 12, 12], "area": 144})
    categories = [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]
    write_json(staging / "train/_annotations.coco.json", {"images": images, "annotations": annotations,
                                                           "categories": categories})
    (staging / "valid").mkdir()
    write_json(staging / "valid/_annotations.coco.json", {"images": [], "annotations": [],
                                                           "categories": categories})
    write_json(staging / "ontology.json", {"classes": CLASSES})
    states = ["exhaustive"]*5 if complete else ["positive_only", "unknown", "exhaustive", "unknown", "positive_only"]
    rows = [{"image_id": i, "split": "train", "states": states} for i in range(2)]
    import json
    (staging / "coverage.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return publish(staging, root, {"schema_version": 1, "kind": "training_view", "index_space_size": 2,
                                  "ontology_digest": "test", "parent_digest": "test"})


def module(view, tmp_path, batch=2):
    mc, tc = configs(view, tmp_path / "out", batch=batch, epochs=1, recipe="augmented")
    dm = RFDETRDataModule(mc, tc)
    dm.setup("fit")
    return dm


def test_plan_exact_mixture_determinism_partial_anchors_and_int_rectangles(tmp_path):
    partial = make_view(tmp_path / "partial")
    plan = build_crop_plan(partial, samples=200, seed=17)
    assert plan == build_crop_plan(partial, samples=200, seed=17)
    assert Counter(row["kind"] for row in plan["rows"]) == {"full_frame": 100, "anchor_crop": 100}
    assert plan["anchor_counts"] == {str(i): 1 for i in range(1, 6)}
    assert all(row["anchor_annotation_id"] in (None, 1, 2, 3, 4, 5) for row in plan["rows"])
    class_counts = Counter(row["anchor_category_id"] for row in plan["rows"] if row["kind"] == "anchor_crop")
    assert set(class_counts) == {1, 2, 3, 4, 5}
    for row in plan["rows"]:
        top, left, height, width = row["crop_tlhw"]
        assert all(isinstance(v, int) for v in row["crop_tlhw"])
        assert 0 <= top < top+height <= 240 and 0 <= left < left+width <= 320
        if row["kind"] == "anchor_crop":
            assert height == width


def test_real_upstream_dataset_same_pixels_and_geometry_partial_complete(tmp_path):
    partial = make_view(tmp_path / "partial")
    complete = make_view(tmp_path / "complete", complete=True)
    plan = build_crop_plan(partial, samples=20)
    partial_dm, complete_dm = module(partial, tmp_path), module(complete, tmp_path)
    partial_validation = partial_dm._dataset_val
    p = install_object_crops(partial_dm, partial, partial, plan)
    c = install_object_crops(complete_dm, complete, partial, plan)
    assert partial_dm._dataset_val is partial_validation
    assert p.dataset._draft_size is None and p.dataset._transforms is None
    table, valid, _ = coverage_table(partial)
    assert table.shape == (2, 5) and valid.tolist() == [True, True]
    assert table[0].tolist() == [False, False, True, False, False]
    for i, row in enumerate(plan["rows"]):
        p_image, p_target = p[i]
        c_image, c_target = c[i]
        assert torch.equal(p_image, c_image)
        assert p_image.shape == (3, 512, 512)
        assert p_target["image_id"].item() == c_target["image_id"].item() == row["image_id"]
        # Complete-only box has label4. Every other class has one source box,
        # so its transformed geometry must be exactly identical across arms.
        for label in range(4):
            assert torch.equal(p_target["boxes"][p_target["labels"] == label],
                               c_target["boxes"][c_target["labels"] == label])
        assert torch.equal(table[p_target["image_id"]], table[c_target["image_id"]])
    assert all(row["anchor_annotation_id"] != 99 for row in c.plan["rows"])
    before = torch.random.get_rng_state()
    _ = p[0]
    assert torch.equal(before, torch.random.get_rng_state())


def test_all_targets_crop_with_correct_clipping_filtering_and_empty_frames(tmp_path):
    from rfdetr.datasets._torchvision import crop
    partial = make_view(tmp_path / "partial")
    plan = build_crop_plan(partial, samples=40, seed=18)
    wrapper = install_object_crops(module(partial, tmp_path), partial, partial, plan)
    saw_empty = saw_removed = False
    for i, row in enumerate(plan["rows"]):
        image, target = wrapper.dataset[wrapper.indices[row["image_id"]]]
        _, expected = crop(image, target, *row["crop_tlhw"])
        _, actual = wrapper[i]
        assert torch.equal(actual["labels"], expected["labels"])
        assert actual["boxes"].shape[0] == expected["boxes"].shape[0]
        # Independent arithmetic oracle for integer crop, optional flip, and
        # final normalized cxcywh; catches shared transform mistakes across arms.
        top, left, height, width = row["crop_tlhw"]
        manual = []
        for x1, y1, x2, y2 in target["boxes"].tolist():
            x1, x2 = max(0., min(width, x1-left)), max(0., min(width, x2-left))
            y1, y2 = max(0., min(height, y1-top)), max(0., min(height, y2-top))
            if x2 <= x1 or y2 <= y1:
                continue
            if row["flip"]:
                x1, x2 = width-x2, width-x1
            manual.append([(x1+x2)/(2*width), (y1+y2)/(2*height),
                           (x2-x1)/width, (y2-y1)/height])
        assert torch.allclose(actual["boxes"], torch.tensor(manual).reshape(-1, 4), atol=1e-6)
        assert torch.all(actual["boxes"] >= 0) and torch.all(actual["boxes"] <= 1)
        assert actual["image_id"].item() == row["image_id"]
        saw_empty |= actual["boxes"].shape[0] == 0
        saw_removed |= row["image_id"] == 0 and actual["boxes"].shape[0] < 5
    assert saw_empty and saw_removed


def test_loader_sequential_exact_plan_and_no_global_rng_consumption(tmp_path):
    partial = make_view(tmp_path / "partial")
    plan = build_crop_plan(partial, samples=8)
    dm = module(partial, tmp_path)
    wrapper = install_object_crops(dm, partial, partial, plan)
    loader = dm.train_dataloader()
    assert loader.num_workers == 0 and len(loader) == 4
    assert list(loader.sampler) == list(range(8))
    first = next(iter(loader))
    expected = [wrapper[i][1]["image_id"].item() for i in range(2)]
    assert [target["image_id"].item() for target in first[1]] == expected
    dm.setup("fit")
    assert dm._dataset_train is wrapper


def test_pseudo_missing_anchors_content_mismatch_and_mutated_plan_reject(tmp_path):
    partial = make_view(tmp_path / "partial")
    with pytest.raises(ValueError, match="pseudo"):
        build_crop_plan(make_view(tmp_path / "pseudo", pseudo=True), samples=2)
    with pytest.raises(ValueError, match="at least one human"):
        build_crop_plan(make_view(tmp_path / "empty", empty=True), samples=2)
    plan = build_crop_plan(partial, samples=4)
    changed = make_view(tmp_path / "changed", complete=True, changed_image=True)
    with pytest.raises(ValueError, match="content hashes"):
        install_object_crops(module(changed, tmp_path), changed, partial, plan)
    modified = copy.deepcopy(plan)
    modified["rows"][0]["image_id"] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        install_object_crops(module(partial, tmp_path), partial, partial, modified)
    missing = make_view(tmp_path / "missing", empty=True)
    with pytest.raises(ValueError, match="missing or altered"):
        install_object_crops(module(missing, tmp_path), missing, partial, plan)


def test_rectangular_oversized_anchor_fallback_and_edge_containment():
    from coveragecv.training.object_crops import _anchor_crop
    generator = torch.Generator().manual_seed(7)
    image = {"width": 100, "height": 300}
    rectangle, fallback = _anchor_crop(image, {"bbox": [10, 10, 70, 200]}, 512, generator)
    assert rectangle == [0, 0, 300, 100] and fallback == "anchor_larger_than_square"
    image = {"width": 400, "height": 240}
    for box in ([0.1, .2, 20., 30.], [379.5, 209.5, 20.5, 30.5]):
        for _ in range(10):
            rectangle, fallback = _anchor_crop(image, {"bbox": box}, 512, generator)
            top, left, height, width = rectangle
            assert fallback is None
            assert 0 <= left <= box[0] and box[0]+box[2] <= left+width <= image["width"]
            assert 0 <= top <= box[1] and box[1]+box[3] <= top+height <= image["height"]


def test_plan_does_not_parse_validation_labels(tmp_path):
    # Republish a valid inventory with deliberately invalid validation JSON.
    partial = make_view(tmp_path / "partial")
    import shutil

    from coveragecv.artifacts import read_json
    destination = tmp_path / "no-validation"
    staging = stage(destination)
    manifest = read_json(partial / "manifest.json")
    for name in manifest["files"]:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(partial / name, target)
    (staging / "valid/_annotations.coco.json").write_text("not parsed by planner")
    changed = publish(staging, destination, {key: value for key, value in manifest.items()
                                           if key not in ("files", "digest")})
    assert build_crop_plan(changed, samples=4)["validation_or_test_labels_used"] is False
