import pytest

pytest.importorskip("rfdetr")
from coveragecv.artifacts import write_json
from coveragecv.training.runner import checkpoint_config, configs


def test_large_recipe_and_checkpoint_preserve_actual_architecture(tmp_path):
    write_json(tmp_path / "ontology.json", {"classes": ["forklift", "person", "pallet"]})
    large, train = configs(tmp_path, tmp_path / "run", recipe="large_fresh")
    assert (large.resolution, large.dec_layers, large.positional_encoding_size) == (704, 4, 44)
    assert train.use_ema and train.scale_jitter and train.lr_scheduler == "cosine"
    restored = checkpoint_config({"model_variant": "large", "model_config": large.model_dump(mode="json")})
    assert restored.model_dump() == large.model_dump()
    nano, train = configs(tmp_path, tmp_path / "run", recipe="augmented")
    assert (nano.resolution, nano.dec_layers, nano.positional_encoding_size) == (512, 2, 24)
    assert not train.use_ema
    with pytest.raises(ValueError, match="unsupported"):
        checkpoint_config({"model_variant": "made-up", "model_config": {}})
