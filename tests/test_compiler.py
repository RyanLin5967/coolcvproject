import copy
import json

import pytest
from PIL import Image

from coveragecv.artifacts import read_json, verify
from coveragecv.compiler import compile_bundle, materialize_view
from coveragecv.schema import CompileSpec, DiagnosticError


@pytest.fixture
def source(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 24), (i*60, 50, 20)).save(images / f"{i}.png")
    data = {"images": [{"id": i, "file_name": f"{i}.png", "width": 32, "height": 24} for i in range(3)],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 4, "bbox": [1, 2, 10, 12]}],
            "categories": [{"id": 4, "name": "cat"}, {"id": 9, "name": "dog"}]}
    spec = {"schema_version": 1, "classes": ["cat", "dog"], "sources": [
        {"id": "a", "revision": "v1", "annotations": "coco.json", "images": "images", "split": "train",
         "class_map": {"cat": "cat", "dog": "dog"}, "coverage": {"cat": "exhaustive"},
         "evidence": "test declaration", "attribution": "test images created for this test"}]}
    (tmp_path / "coco.json").write_text(json.dumps(data))
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    return tmp_path, data, spec


def test_deterministic_semantic_order_and_view_custody(source):
    root, data, spec = source
    a = compile_bundle(root / "spec.json", root / "bundles")
    data["images"].reverse()
    data["categories"].reverse()
    (root / "coco.json").write_text(json.dumps(data, indent=4))
    b = compile_bundle(root / "spec.json", root / "bundles")
    assert a == b
    rows = [json.loads(r) for r in (a / "coverage.jsonl").read_text().splitlines()]
    assert all(row["states"] == ["exhaustive", "unknown"] for row in rows)
    view = materialize_view(a, root / "views")
    m = verify(view)
    assert m["parent_digest"] == verify(a)["digest"]
    assert not (view / "test").exists()
    assert not any("test" in f for f in m["files"])
    assert [x["id"] for x in read_json(view / "train/_annotations.coco.json")["images"]] == list(range(3))
    spec["sources"][0]["coverage"]["dog"] = "exhaustive"
    (root / "spec.json").write_text(json.dumps(spec))
    assert compile_bundle(root / "spec.json", root / "bundles") != a


def test_absence_contradiction(source):
    root, _, spec = source
    spec["sources"][0]["coverage"]["cat"] = "verified_absent"
    (root / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(DiagnosticError) as e:
        compile_bundle(root / "spec.json", root / "bundles")
    assert e.value.code == "COVERAGE_ABSENCE_CONTRADICTION"


@pytest.mark.parametrize("corrupt", [False, True])
def test_platform_roundtrip_rejects_coordinate_shift(source, corrupt):
    from coveragecv.artifacts import write_json
    from coveragecv.providers import verify_roboflow_export
    root, _, _ = source
    bundle = compile_bundle(root / "spec.json", root / "bundles")
    view = materialize_view(bundle, root / "views")
    for split in ("train", "valid"):
        data = read_json(view / split / "_annotations.coco.json")
        for im in data["images"]:
            im["extra"] = {"user_metadata": {"coveragecv_image_id": im["id"],
                                               "coveragecv_view": verify(view)["digest"]}}
        if corrupt and data["annotations"]:
            data["annotations"][0]["bbox"][0] -= 1
        write_json(root / "export" / split / "_annotations.coco.json", data)
    if corrupt:
        with pytest.raises(ValueError, match="geometry"):
            verify_roboflow_export(view, root / "export", root / "roundtrip.json")
    else:
        assert verify_roboflow_export(view, root / "export", root / "roundtrip.json")["status"] == "passed"


@pytest.mark.parametrize("case", ["split", "coverage", "duplicate"])
def test_duplicate_policy(source, case):
    root, _, spec = source
    s = copy.deepcopy(spec["sources"][0])
    s["id"] = "b"
    if case == "split":
        s["split"] = "test"
    if case == "coverage":
        s["coverage"]["dog"] = "exhaustive"
    spec["sources"].append(s)
    (root / "spec.json").write_text(json.dumps(spec))
    if case != "duplicate":
        with pytest.raises(DiagnosticError, match="different splits|conflicting"):
            compile_bundle(root / "spec.json", root / "bundles")
    else:
        bundle = compile_bundle(root / "spec.json", root / "bundles")
        assert read_json(bundle / "diagnostics.json")["coalesced_observations"] == 3


def test_corruption_and_extra_payload_rejected(source):
    root, _, _ = source
    bundle = compile_bundle(root / "spec.json", root / "bundles")
    (bundle / "extra.txt").write_text("not declared")
    with pytest.raises(DiagnosticError, match="inventory"):
        verify(bundle)
    (bundle / "extra.txt").unlink()
    (bundle / "ontology.json").write_text("{}")
    with pytest.raises(DiagnosticError, match="modified"):
        verify(bundle)


@pytest.mark.parametrize("field,value", [("bbox", [0, 0, -1, 2]), ("bbox", [0, 0, float('nan'), 1]),
                                         ("iscrowd", 1)])
def test_invalid_annotations(source, field, value):
    root, data, _ = source
    data["annotations"][0][field] = value
    (root / "coco.json").write_text(json.dumps(data))
    with pytest.raises(DiagnosticError):
        compile_bundle(root / "spec.json", root / "bundles")
    assert not list((root / "bundles").glob(".staging-*"))


def test_unsafe_image_path_and_unknown_schema(source):
    root, data, spec = source
    data["images"][0]["file_name"] = "../elsewhere.png"
    (root / "coco.json").write_text(json.dumps(data))
    with pytest.raises(DiagnosticError, match="relative path"):
        compile_bundle(root / "spec.json", root / "bundles")
    spec["sources"][0]["exhaustive_by_default"] = True
    with pytest.raises(ValueError):
        CompileSpec.model_validate_json(json.dumps(spec))
