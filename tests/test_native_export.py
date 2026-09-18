"""The ordinary upstream inference API must reload our plain detector exports."""
import gc

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rfdetr")
from rfdetr import RFDETR
from rfdetr.training import RFDETRModelModule

from coveragecv.artifacts import write_json
from coveragecv.training.runner import configs, parameter_digest, save_detector


@pytest.mark.parametrize("recipe,model_name,resolution,pe", [
    ("augmented_fresh", "RFDETRNano", 512, 24),
    ("large_fresh", "RFDETRLarge", 704, 44),
])
def test_public_checkpoint_loader_preserves_model_and_custom_geometry(
        tmp_path, monkeypatch, recipe, model_name, resolution, pe):
    # These are real models, constructed without pretrained weights or training.
    # Guard the public constructor against an accidental network download.
    def local_checkpoint_only(path):
        assert path == str(tmp_path / "detector.pt")

    monkeypatch.setattr("rfdetr.detr.download_pretrain_weights", local_checkpoint_only)
    classes = ["forklift", "person", "pallet"]
    write_json(tmp_path / "ontology.json", {"classes": classes})
    mc, tc = configs(tmp_path, tmp_path / "unused-training-output", recipe=recipe)
    module = RFDETRModelModule(mc, tc)
    expected_digest = parameter_digest(module.model)
    save_detector(module, mc, tc, tmp_path / "detector.pt")
    del module
    gc.collect()

    # Generic filename: no accidental success via upstream's filename fallback.
    restored = RFDETR.from_checkpoint(tmp_path / "detector.pt")
    assert type(restored).__name__ == model_name
    assert restored.class_names == classes
    assert (restored.model_config.resolution, restored.model_config.positional_encoding_size) == (resolution, pe)
    assert restored.model.resolution == resolution
    assert parameter_digest(restored.model.model) == expected_digest
    del restored
    gc.collect()
