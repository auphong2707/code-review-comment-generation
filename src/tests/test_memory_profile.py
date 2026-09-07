"""The opt-in profile must conserve protocol identity while reducing residency."""
from copy import deepcopy
import pytest

from src.training.memory_profile import apply_memory_profile
from src.training.registry import get_spec
from src.training.protocols import overrides_for
from src.train import main


@pytest.mark.parametrize("model,method", [
    ("codet5", "fmft"), ("plbart", "lora"), ("qwen25-0.5b", "fmft"),
    ("qwen25", "ia3"), ("qwen15", "lora"), ("qwen25-14b", "qlora"),
])
@pytest.mark.parametrize("dataset", ["trans_review", "acra_by_ls_pt"])
def test_effective_batch_and_frozen_spec(model, method, dataset):
    _, _, spec = get_spec(model, method)
    before = deepcopy(spec)
    protocol = overrides_for(dataset, model, "archived")
    resolved = {**spec["training_args"], **protocol}
    derived, runtime = apply_memory_profile(spec, protocol, "a5000", model, method)
    assert spec == before
    assert runtime["per_device_train_batch_size"] == 1
    assert runtime["per_device_eval_batch_size"] == 1
    assert runtime["gradient_accumulation_steps"] == (
        resolved["per_device_train_batch_size"] * resolved.get("gradient_accumulation_steps", 1))
    assert runtime["gradient_checkpointing"] is True
    assert derived["grad_checkpoint_enable"] is True
    for key in ("name_stem", "generation_config", "peft_config", "bnb_config"):
        assert derived[key] == before[key]
    assert runtime.get("num_train_epochs", spec["training_args"]["num_train_epochs"]) == resolved["num_train_epochs"]


@pytest.mark.parametrize("model,method", [("qwen25-14b", "lora"), ("qwen25", "galore"),
                                                   ("qwen25", "fmft"), ("qwen15", "fmft"),
                                                   ("codet5", "ptuning")])
def test_unsafe_training_is_rejected(model, method):
    _, _, spec = get_spec(model, method)
    with pytest.raises(ValueError, match="A5000"):
        apply_memory_profile(spec, {}, "a5000", model, method)


def test_default_profile_is_identity():
    _, _, spec = get_spec("qwen25", "lora")
    derived, overrides = apply_memory_profile(spec, {}, None, "qwen25", "lora")
    assert derived is spec
    assert overrides == {}


def test_profile_requires_separate_root():
    with pytest.raises(SystemExit, match="isolated"):
        main(["--model", "codet5", "--method", "fmft", "--memory-profile", "a5000", "--dry-run"])


def test_profile_dry_run_reports_resolved_settings(capsys, tmp_path):
    import json
    main(["--model", "qwen25", "--method", "ia3", "--memory-profile", "a5000",
          "--dataset", "trans_review", "--results-root", str(tmp_path), "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory_profile"] == "a5000"
    assert payload["protocol_overrides"]["per_device_train_batch_size"] == 1
    assert payload["protocol_overrides"]["gradient_checkpointing"]
