"""Registry naming invariants (phase 02): legacy names byte-identical, seed
suffix and results_root strictly opt-in."""

import pytest

from src.training.registry import get_spec, resolve_run
from src.training.specs import LEGACY_SPECS

# Golden legacy stems — hardcoded on purpose: a spec regression must fail here,
# not be silently reproduced by deriving the expectation from the spec table.
GOLDEN_STEMS = {
    ("codet5", "fmft"): "codet5_fmft",
    ("codet5", "ia3"): "codet5_ia3",
    ("codet5", "lora"): "codet5_lora",
    ("codet5", "qlora"): "codet5_qlora",
    ("codet5", "ptuning"): "codet5_ptuning_v2",
    ("plbart", "fmft"): "plbart_fmft",
    ("plbart", "ia3"): "plbart_ia3",
    ("plbart", "lora"): "plbart_lora",
    ("plbart", "qlora"): "plbart_qlora",
    ("plbart", "ptuning"): "plbart_ptuning_v2",
    ("qwen15", "fmft"): "qwen15_fmft",
    ("qwen15", "ia3"): "qwen15_ia3",
    ("qwen15", "lora"): "qwen15_lora",
    ("qwen15", "qlora"): "qwen15_qlora",
    ("qwen15", "ptuning"): "qwen15_ptuning_v2",
    ("qwen25", "fmft"): "qwen25_fmft",
    ("qwen25", "ia3"): "qwen25_ia3",
    ("qwen25", "lora"): "qwen25_lora",
    ("qwen25", "qlora"): "qwen25_qlora",
    ("qwen25", "ptuning"): "qwen25_ptuning_v2",
    ("qwen25-0.5b", "fmft"): "qwen25_0.5b_fmft",
    ("qwen25-0.5b", "ia3"): "qwen25_0.5b_ia3",
    ("qwen25-0.5b", "lora"): "qwen25_0.5b_lora",
    ("qwen25-0.5b", "qlora"): "qwen25_0.5b_qlora",
}


def test_golden_legacy_names():
    assert set(GOLDEN_STEMS) == set(LEGACY_SPECS)
    for key, stem in GOLDEN_STEMS.items():
        spec = LEGACY_SPECS[key]
        official = resolve_run(spec, learning_rate=5e-05)
        sweep = resolve_run(spec, experiment_id=0)
        assert official["experiment_name"] == f"{stem}_official_5e-05"
        assert official["experiment_results_dir"] == spec["results_dir"] + f"{stem}_official_5e-05"
        assert sweep["experiment_name"] == f"{stem}_0"


def test_seed_suffix_only_when_set():
    spec = LEGACY_SPECS[("codet5", "lora")]
    assert resolve_run(spec, learning_rate=5e-05)["experiment_name"] == "codet5_lora_official_5e-05"
    named = resolve_run(spec, learning_rate=5e-05, seed=1)
    assert named["experiment_name"] == "codet5_lora_official_5e-05_seed1"
    assert named["experiment_results_dir"].endswith("codet5_lora_official_5e-05_seed1")


def test_results_root_joins():
    spec = LEGACY_SPECS[("codet5", "lora")]
    run = resolve_run(spec, learning_rate=5e-05, seed=2, results_root="scratch/revision/tr")
    assert run["experiment_results_dir"] == "scratch/revision/tr/codet5_lora_official_5e-05_seed2"


def test_learning_rate_zero_is_official():
    run = resolve_run(LEGACY_SPECS[("codet5", "lora")], learning_rate=0.0)
    assert run["experiment_name"] == "codet5_lora_official_0.0"
    assert run["learning_rate"] == 0.0


def test_get_spec_errors():
    with pytest.raises(ValueError, match="unknown model"):
        get_spec("nope", "lora")
    with pytest.raises(ValueError, match="unsupported combination"):
        get_spec("qwen25-0.5b", "ptuning")
