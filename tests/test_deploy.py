import pytest

torch = pytest.importorskip("torch")

from coveragecv.training.deploy import reorder_classifier_rows


def test_serving_layout_preserves_all_semantic_scores_and_reserved_row():
    torch.manual_seed(17)
    classes = ["forklift", "person", "pallet"]
    state = {"class_embed.weight": torch.randn(4, 8), "class_embed.bias": torch.randn(4),
             "transformer.enc_out_class_embed.0.weight": torch.randn(4, 8),
             "transformer.enc_out_class_embed.0.bias": torch.randn(4), "backbone.weight": torch.randn(5, 5)}
    converted, changed = reorder_classifier_rows(state, classes)
    features = torch.randn(2, 11, 8)
    for head in ("class_embed", "transformer.enc_out_class_embed.0"):
        native = torch.nn.functional.linear(features, state[head+".weight"], state[head+".bias"])
        exported = torch.nn.functional.linear(features, converted[head+".weight"], converted[head+".bias"])
        torch.testing.assert_close(native[..., :3], exported[..., 1:])
        torch.testing.assert_close(native[..., 3], exported[..., 0])
    assert len(changed) == 4
    assert converted["backbone.weight"] is state["backbone.weight"]
    assert not torch.equal(converted["class_embed.weight"], state["class_embed.weight"])


def test_wrong_export_ontology_fails_closed():
    with pytest.raises(ValueError, match="dimensions"):
        reorder_classifier_rows({"class_embed.weight": torch.zeros(4, 8)}, ["cat", "dog"])
