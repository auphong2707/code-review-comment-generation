"""Derived revision specs (phase 06, D-026): the diff to the parent spec is
exactly ALLOWED_DIFF, legacy specs are untouched, and the galore spec builds
real TrainingArguments with the GaLore optimizer fields."""

import pytest

from src.training.registry import SPECS, get_spec, resolve_run
from src.training.specs import LEGACY_SPECS
from src.training.specs_revision import (
    ALLOWED_DIFF,
    DORA_MODELS,
    GALORE_MODELS,
    REMOVED,
    VARIES,
    derive_specs,
    explain,
    peft_pin_for,
    spec_diff,
)

DERIVED = derive_specs(LEGACY_SPECS)


def test_legacy_specs_are_untouched_in_registry():
    for key, spec in LEGACY_SPECS.items():
        assert SPECS[key] is spec  # same object: derive_specs never mutates


def test_derived_key_sets_match_parent():
    for key, spec in DERIVED.items():
        parent_key, _ = explain(key)
        assert set(spec) == set(LEGACY_SPECS[parent_key])


@pytest.mark.parametrize("model", DORA_MODELS)
def test_dora_diff_is_allowed(model):
    parent_key, diff = explain((model, "dora"))
    assert parent_key == (model, "lora")
    allowed = ALLOWED_DIFF["dora"]
    assert set(diff) <= set(allowed)
    assert diff["peft_config.kwargs.use_dora"] is True
    assert diff["name_stem"].endswith("_dora")


@pytest.mark.parametrize("model", GALORE_MODELS)
def test_galore_diff_is_allowed(model):
    parent_key, diff = explain((model, "galore"))
    assert parent_key == (model, "fmft")
    allowed = ALLOWED_DIFF["galore"]
    assert set(diff) <= set(allowed)
    for path, value in diff.items():
        if allowed[path] != VARIES:
            assert value == allowed[path], path
    assert diff["training_args.optim"] == "galore_adamw"
    if "deepspeed" in LEGACY_SPECS[(model, "fmft")]["training_args"]:
        assert diff["training_args.deepspeed"] == REMOVED


def test_derived_naming():
    assert resolve_run(SPECS[("qwen25", "dora")], learning_rate=5e-05)["experiment_name"] \
        == "qwen25_dora_official_5e-05"
    assert resolve_run(SPECS[("qwen25", "galore")], learning_rate=5e-05)["experiment_name"] \
        == "qwen25_galore_official_5e-05"
    assert get_spec("qwen25-0.5b", "dora")[2]["name_stem"] == "qwen25_0.5b_dora"


def test_galore_spec_builds_training_args(tmp_path):
    from src.training.peft_setup import build_training_args, static_marker_context

    spec = SPECS[("qwen25-0.5b", "galore")]
    run_info = resolve_run(spec, learning_rate=5e-05)
    run_info["experiment_results_dir"] = str(tmp_path)
    args = build_training_args(spec, static_marker_context(run_info, spec))
    assert str(args.optim).endswith("GALORE_ADAMW") or args.optim == "galore_adamw"
    assert args.optim_target_modules == ["q_proj", "k_proj", "v_proj",
                                         "gate_proj", "up_proj", "down_proj"]
    assert args.optim_args == "rank=128,update_proj_gap=200,scale=0.25"
    assert args.gradient_checkpointing is True


def test_peft_pin_for():
    pin = peft_pin_for("trans_review", "codet5", "ia3")
    assert pin == {"target_modules": ["v", "wo", "k"], "feedforward_modules": ["wo"]}
    assert peft_pin_for("acra_by_ls_pt", "codet5", "ia3") == {}
    assert peft_pin_for("trans_review", "codet5", "lora") == {}
    # the pin never leaks into the frozen spec dict
    assert "feedforward_modules" not in str(
        LEGACY_SPECS[("codet5", "ia3")]["peft_config"]["kwargs"].get("target_modules", ""))


def test_spec_diff_detects_removal_and_addition():
    parent = {"a": 1, "nested": {"x": 1, "y": 2}}
    derived = {"a": 1, "nested": {"x": 5, "z": 3}}
    assert spec_diff(parent, derived) == {"nested.x": 5, "nested.y": REMOVED, "nested.z": 3}
