import pytest

from coveragecv.providers import voc_annotation
from coveragecv.report import validate_run_evidence


@pytest.mark.parametrize("mutation", ["wrong_arm", "incomplete", "wrong_view", "stale_checkpoint", "wrong_reference", "test_split"])
def test_report_rejects_misleading_comparisons(mutation):
    run = {"arm": "naive", "status": "completed", "view_digest": "partial", "detector_sha256": "checkpoint"}
    evaluation = {"checkpoint_sha256": "checkpoint", "bundle_digest": "complete", "split": "valid"}
    if mutation == "wrong_arm":
        run["arm"] = "aware"
    elif mutation == "incomplete":
        run["status"] = "incomplete"
    elif mutation == "wrong_view":
        run["view_digest"] = "complete"
    elif mutation == "stale_checkpoint":
        evaluation["checkpoint_sha256"] = "old"
    elif mutation == "wrong_reference":
        evaluation["bundle_digest"] = "partial"
    else:
        evaluation["split"] = "test"
    with pytest.raises(ValueError):
        validate_run_evidence("naive", run, evaluation, view_digest="partial", bundle_digest="complete",
                              checkpoint_digest="checkpoint")


def test_voc_preserves_fractional_geometry_and_escapes_names():
    from xml.etree.ElementTree import fromstring
    image = {"file_name": "a&b.jpg", "width": 640, "height": 480}
    annotation = {"category_id": 4, "bbox": [10.25, 11.5, 12.75, 13.5]}
    xml = fromstring(voc_annotation(image, [annotation], [{"id": 4, "name": "a&b"}])["rawText"])
    assert xml.findtext("filename") == "a&b.jpg"
    assert xml.findtext("object/name") == "a&b"
    box = xml.find("object/bndbox")
    assert [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")] == [11.25, 12.5, 24, 26]
