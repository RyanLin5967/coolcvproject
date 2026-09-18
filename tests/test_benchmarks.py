import pytest

from coveragecv.benchmarks import paired_delta, stats, summarize


def row(method, seed, score=.5, **kwargs):
    return {"method": method, "seed": seed, "steps": 2000, "batch": 4, "resolution": 384,
            "recipe": "pilot", "device": "cuda", "original_view_digest": "view",
            "initial_parameter_digest": f"initial-{seed}", "warm_start_sha256": f"warm-{seed}",
            "metrics": {"AP": score, "AP50": score, "AP75": score, "recall_at_threshold": score,
                        "precision_at_threshold": score, "false_positives_per_image": 0,
                        "per_class_AP": {"object": score}}, **kwargs}


def test_seed_summary_uses_sample_variance_and_rejects_duplicate_or_mixed_budgets():
    s = stats([.4, .5, .6])
    assert s["mean"] == .5 and s["sd"] == pytest.approx(.1)
    assert stats([.4])["sd"] is None
    assert summarize([row("aware", 1), row("aware", 2)])["n"] == 2
    with pytest.raises(ValueError, match="Repeated seeds"):
        summarize([row("aware", 1), row("aware", 1)])
    with pytest.raises(ValueError, match="different training contracts"):
        summarize([row("aware", 1), row("aware", 2, steps=4000)])


def test_paired_comparison_uses_shared_seeds_and_requires_matching_checkpoints():
    rows = [row("aware", 1, .6), row("naive", 1, .4), row("aware", 2, .8)]
    result = paired_delta(rows, "aware", "naive")
    assert result["seeds"] == [1]
    assert result["AP_points"]["mean"] == pytest.approx(20.)
    with pytest.raises(ValueError, match="unmatched"):
        paired_delta([row("aware", 1), row("naive", 1, initial_parameter_digest="wrong")], "aware", "naive")
    with pytest.raises(ValueError, match="exact starting checkpoint"):
        paired_delta([row("aware_teacher_512", 1), row("aware_augmented_512", 1, warm_start_sha256="wrong")],
                     "aware_teacher_512", "aware_augmented_512")
