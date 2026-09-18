import copy
import json
import shutil
from collections import Counter

import pytest
from PIL import Image

from coveragecv.artifacts import (
    digest,
    file_digest,
    publish,
    read_json,
    stage,
    verify,
    write_json,
    write_jsonl,
)
from coveragecv.compiler import _jsonl
from coveragecv.training.acquisition import apply_reviews, plan_reviews

CLASSES = ["helmet", "no-helmet", "no-vest", "person", "vest"]
CHECKPOINT = "a" * 64


def test_cli_freezes_plan_then_simulates_without_overwriting(tmp_path):
    from typer.testing import CliRunner

    from coveragecv.cli import app

    partial = make_view(tmp_path / "partial")
    reference = make_view(tmp_path / "reference", reference=True)
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"fixture checkpoint identity")
    prediction_path = tmp_path / "predictions.json"
    write_json(prediction_path, {"split": "train", "view_digest": verify(partial)["digest"],
        "checkpoint_sha256": file_digest(checkpoint), "predictions": predictions()})
    output = tmp_path / "plan.json"
    runner = CliRunner()
    args = ["plan-reviews", str(partial), str(prediction_path), str(checkpoint), str(output), "--per-class", "1"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["review_units"] == 5
    frozen = output.read_bytes()
    assert runner.invoke(app, args).exit_code == 2
    assert output.read_bytes() == frozen
    applied = runner.invoke(app, ["simulate-reviews", str(partial), str(reference), str(output), str(tmp_path / "views")])
    assert applied.exit_code == 0, applied.output
    from pathlib import Path
    acquired = Path(json.loads(applied.output))
    assert read_json(acquired / "acquisition-audit.json")["review_units"] == 5


def test_cli_rejects_predictions_bound_to_validation(tmp_path):
    from typer.testing import CliRunner

    from coveragecv.cli import app

    partial = make_view(tmp_path / "partial")
    checkpoint = tmp_path / "detector.pt"
    checkpoint.write_bytes(b"fixture checkpoint identity")
    prediction_path = tmp_path / "predictions.json"
    write_json(prediction_path, {"split": "valid", "view_digest": verify(partial)["digest"],
        "checkpoint_sha256": file_digest(checkpoint), "predictions": predictions()})
    output = tmp_path / "plan.json"
    result = CliRunner().invoke(app, ["plan-reviews", str(partial), str(prediction_path), str(checkpoint), str(output)])
    assert result.exit_code == 2
    assert not output.exists()


def make_view(root, *, reference=False, change=None):
    staging = stage(root)
    (staging / "train").mkdir()
    images = []
    for iid in range(4):
        filename = f"{iid}.png"
        Image.new("RGB", (100, 80), (iid * 30, 50, 70)).save(staging / "train" / filename)
        images.append({"id": iid, "file_name": filename, "width": 100, "height": 80})
    annotations = [{"id": c+100, "image_id": 0, "category_id": c,
                    "bbox": [c*5, 4, 4, 8], "area": 32, "iscrowd": 0}
                   for c in range(1, 6)]
    if reference:
        for iid in range(3):
            for category in range(1, 6):
                if iid == 1 and category % 2 == 0:
                    continue
                annotations.append({"id": 200+iid*10+category, "image_id": iid,
                                    "category_id": category, "bbox": [category*5, 20, 4, 8],
                                    "area": 32, "iscrowd": 0})
    categories = [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]
    data = {"images": images, "annotations": annotations, "categories": categories}
    if change:
        change(data)
    ontology = {"classes": CLASSES, "categories": categories, "reserved_model_index": 5}
    write_json(staging / "train/_annotations.coco.json", data)
    # Selection and oracle application must not parse even validation labels.
    (staging / "valid").mkdir()
    (staging / "valid/_annotations.coco.json").write_text("validation is intentionally not JSON")
    write_json(staging / "ontology.json", ontology)
    coverage = [{"image_id": iid, "split": "train", "sample_key": f"sample-{iid}",
                 "states": (["exhaustive"]*5 if reference or iid == 3 else
                            ["positive_only"]*5 if iid == 0 else ["unknown"]*5)} for iid in range(4)]
    write_jsonl(staging / "coverage.jsonl", coverage)
    return publish(staging, root, {"schema_version": 1, "kind": "training_view",
                                  "ontology_digest": digest(ontology), "index_space_size": 4,
                                  "parent_digest": "test-source"})


def predictions():
    rows = [{"image_id": iid, "category_id": category, "bbox": [10, 10, 5, 6],
             "score": .9 if iid == 1 else .3} for iid in range(3) for category in range(1, 6)]
    # Max, not average, confidence must govern; duplicate low confidence cannot
    # dilute image1. High confidence on exhaustive image3 is ineligible.
    rows.extend([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 5, 6], "score": .01},
                 {"image_id": 3, "category_id": 1, "bbox": [10, 10, 5, 6], "score": 1.0}])
    return rows


def plan(view, rows=None, **kwargs):
    return plan_reviews(view, predictions() if rows is None else rows, per_class=1,
                        checkpoint_sha256=CHECKPOINT, **kwargs)


def test_reproducible_blind_selection_maxima_and_same_class_quotas(tmp_path):
    partial = make_view(tmp_path / "partial")
    guided = plan(partial)
    random = plan(partial, mode="random")
    assert guided == plan(partial, list(reversed(predictions())))
    assert random == plan(partial, mode="random")
    for selected in (guided, random):
        assert selected["query_count"] == 5
        assert Counter(row["category_id"] for row in selected["queries"]) == dict.fromkeys(range(1, 6), 1)
        assert selected["validation_or_test_labels_used"] is False
        assert selected["reference_annotations_used_for_selection"] is False
        assert selected["prediction_count"] == len(predictions())
        assert all(row["image_id"] != 3 for row in selected["queries"])
    assert all(row["image_id"] == 1 for row in guided["queries"])
    assert all(row["max_prediction_confidence"] == .9 for row in guided["queries"])
    assert guided["digest"] != random["digest"]
    assert all(row["max_prediction_confidence"] == 0 for row in plan(partial, [])['queries'])
    # No reference exists at selection time, and broken validation JSON was never parsed.
    assert not (tmp_path / "reference").exists()


def test_reveal_only_queried_labels_coverage_and_canonical_identity(tmp_path):
    partial = make_view(tmp_path / "partial")
    reference = make_view(tmp_path / "reference", reference=True)
    before_partial, before_reference = verify(partial), verify(reference)
    guided = plan(partial)
    frozen = copy.deepcopy(guided)
    output = apply_reviews(partial, reference, guided, tmp_path / "acquired")
    assert output == apply_reviews(partial, reference, guided, tmp_path / "acquired")
    assert guided == frozen
    manifest = verify(output)
    assert manifest["kind"] == "training_view" and manifest["index_space_size"] == 4
    assert manifest["digest"] not in (before_partial["digest"], before_reference["digest"])
    assert verify(partial) == before_partial and verify(reference) == before_reference
    audit = read_json(output / "acquisition-audit.json")
    assert audit["boxes_before"] == 5 and audit["boxes_after"] == 8 and audit["added_boxes"] == 3
    assert audit["review_units"] == 5 and audit["reviewed_unique_images"] == 1
    assert audit["actual_new_human_annotation"] is False
    assert audit["same_label_budget_as_original_experiment"] is False
    assert audit["reference_training_annotations_sha256"] == before_reference["files"]["train/_annotations.coco.json"]
    original = read_json(partial / "train/_annotations.coco.json")
    new = read_json(output / "train/_annotations.coco.json")
    assert new["images"] == original["images"] and new["categories"] == original["categories"]
    mapping = {row["original_id"]: row["new_id"] for row in audit["observed_annotation_id_map"]}
    indexed = {ann["id"]: ann for ann in new["annotations"]}
    for old in original["annotations"]:
        expected = {**old, "id": mapping[old["id"]]}
        assert indexed[expected["id"]] == expected
    acquired = [indexed[i] for i in audit["acquired_annotation_ids"]]
    assert {(ann["image_id"], ann["category_id"]) for ann in acquired} == {(1, 1), (1, 3), (1, 5)}
    assert all(a["annotation_origin"] == "published_training_annotation_oracle" for a in acquired)
    rows = {row["image_id"]: row for row in _jsonl(output / "coverage.jsonl")}
    original_rows = {row["image_id"]: row for row in _jsonl(partial / "coverage.jsonl")}
    assert rows[1]["states"] == ["exhaustive", "verified_absent", "exhaustive", "verified_absent", "exhaustive"]
    assert all(rows[iid] == original_rows[iid] for iid in (0, 2, 3))
    assert (output / "valid/_annotations.coco.json").read_bytes() == (partial / "valid/_annotations.coco.json").read_bytes()


def test_unrequested_reference_box_values_never_consumed(tmp_path):
    partial = make_view(tmp_path / "partial")
    # Known reference quality problems outside queried pairs cannot influence an
    # acquisition output. Leave one unrequested box malformed and marked pseudo.
    def corrupt_unrequested(data):
        data["annotations"][0]["bbox"] = [-100, 0, -5, 7]
        data["annotations"][0]["is_pseudo"] = True
    reference = make_view(tmp_path / "reference", reference=True, change=corrupt_unrequested)
    result = apply_reviews(partial, reference, plan(partial), tmp_path / "out")
    assert read_json(result / "acquisition-audit.json")["added_boxes"] == 3


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p.update(image_id=99), "unknown image/category"),
    (lambda p: p.update(category_id=0), "unknown image/category"),
    (lambda p: p.update(category_id=True), "unknown image/category"),
    (lambda p: p.update(score=float("nan")), "finite"),
    (lambda p: p.update(score=1.01), "between"),
    (lambda p: p.update(bbox=[0, 0, 0, 8]), "geometry"),
    (lambda p: p.update(bbox=[99, 0, 2, 8]), "geometry"),
    (lambda p: p.update(bbox=[0, 0, float("inf"), 8]), "finite"),
    (lambda p: p.update(is_pseudo=True), "pseudo"),
])
def test_reject_invalid_predictions(tmp_path, mutation, match):
    partial = make_view(tmp_path / "partial")
    prediction = predictions()[0]
    mutation(prediction)
    with pytest.raises(ValueError, match=match):
        plan(partial, [prediction])


@pytest.mark.parametrize("mutation,match", [
    (lambda ann: ann.update(is_pseudo=True), "pseudo"),
    (lambda ann: ann.update(bbox=[0, 0, 0, 8]), "geometry"),
    (lambda ann: ann.update(category_id=8), "unknown image/category"),
])
def test_reject_invalid_observed_labels(tmp_path, mutation, match):
    partial = make_view(tmp_path / "partial", change=lambda data: mutation(data["annotations"][0]))
    with pytest.raises(ValueError, match=match):
        plan(partial)


def test_invalid_frozen_quota_tamper_and_parent_binding(tmp_path):
    partial = make_view(tmp_path / "partial")
    reference = make_view(tmp_path / "reference", reference=True)
    with pytest.raises(ValueError, match="eligible"):
        plan_reviews(partial, [], per_class=4, checkpoint_sha256=CHECKPOINT)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        plan_reviews(partial, [], per_class=1)
    tampered = plan(partial)
    tampered["queries"][0]["image_id"] = 0
    with pytest.raises(ValueError, match="digest mismatch"):
        apply_reviews(partial, reference, tampered, tmp_path / "out")
    wrong_quota = plan(partial)
    wrong_quota["queries"].pop()
    wrong_quota["digest"] = digest({k: v for k, v in wrong_quota.items() if k != "digest"})
    with pytest.raises(ValueError, match="quota"):
        apply_reviews(partial, reference, wrong_quota, tmp_path / "out")
    wrong_parent = plan(partial)
    wrong_parent["partial_view_digest"] = "f"*64
    wrong_parent["digest"] = digest({k: v for k, v in wrong_parent.items() if k != "digest"})
    with pytest.raises(ValueError, match="bound"):
        apply_reviews(partial, reference, wrong_parent, tmp_path / "out")


def test_image_ontology_mismatch_and_missing_observed_oracle_fail_closed(tmp_path):
    partial = make_view(tmp_path / "partial")
    guided = plan(partial)
    changed = make_view(tmp_path / "dimensions", reference=True,
                        change=lambda data: data["images"][0].update(width=101))
    with pytest.raises(ValueError, match="image IDs"):
        apply_reviews(partial, changed, guided, tmp_path / "out")
    changed_category = make_view(tmp_path / "categories", reference=True,
                                 change=lambda data: data["categories"][0].update(id=99))
    with pytest.raises(ValueError, match="category IDs"):
        apply_reviews(partial, changed_category, guided, tmp_path / "out")
    # Force selection of image0, whose five human labels must survive verbatim.
    p = [{**row, "score": 1.0 if row["image_id"] == 0 else .1} for row in predictions()]
    plan0 = plan(partial, p)
    missing = make_view(tmp_path / "missing", reference=True,
                        change=lambda data: data["annotations"].pop(0))
    with pytest.raises(ValueError, match="missing or altered"):
        apply_reviews(partial, missing, plan0, tmp_path / "out")


def test_training_coverage_contract_compatibility(tmp_path):
    pytest.importorskip("torch")
    from coveragecv.training.criterion import coverage_table
    partial = make_view(tmp_path / "partial")
    reference = make_view(tmp_path / "reference", reference=True)
    result = apply_reviews(partial, reference, plan(partial), tmp_path / "out")
    table, valid, metadata = coverage_table(result)
    assert table.shape == (4, 5) and valid.tolist() == [True]*4
    assert table.tolist() == [[False]*5, [True]*5, [False]*5, [True]*5]
    assert metadata["digest"] == verify(result)["digest"]


def test_reference_image_content_and_incomplete_coverage_rejected(tmp_path):
    partial = make_view(tmp_path / "partial")
    reference = make_view(tmp_path / "reference", reference=True)

    def republish(change, root):
        staging = stage(root)
        manifest = verify(reference)
        for name in manifest["files"]:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference / name, target)
        change(staging)
        return publish(staging, root, {k: v for k, v in manifest.items() if k not in ("files", "digest")})

    mismatch = republish(lambda root: Image.new("RGB", (100, 80), (255, 0, 0)).save(root / "train/0.png"),
                         tmp_path / "mismatch")
    with pytest.raises(ValueError, match="content hashes"):
        apply_reviews(partial, mismatch, plan(partial), tmp_path / "out")
    def break_coverage(root):
        rows = _jsonl(root / "coverage.jsonl")
        rows[1]["states"][0] = "unknown"
        write_jsonl(root / "coverage.jsonl", rows)
    incomplete = republish(break_coverage, tmp_path / "incomplete")
    with pytest.raises(ValueError, match="exhaustive"):
        apply_reviews(partial, incomplete, plan(partial), tmp_path / "out")
    tampered = json.loads((partial / "manifest.json").read_text())
    tampered["index_space_size"] = 999
    (partial / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="identity"):
        plan(partial)
