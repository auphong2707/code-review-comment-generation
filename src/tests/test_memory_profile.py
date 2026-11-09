"""The opt-in A100 profile keeps the native protocol and the frozen spec."""
from copy import deepcopy
import pytest

from src.training.memory_profile import apply_memory_profile, metric_resources
from src.training.registry import get_spec
from src.training.protocols import overrides_for
from src.train import main


@pytest.mark.parametrize("model,method", [
    ("codet5", "fmft"), ("plbart", "qlora"), ("qwen25-0.5b", "fmft"),
    ("qwen25", "ia3"), ("qwen15", "lora"), ("qwen25-14b", "qlora"), ("qwen25", "dora"),
])
@pytest.mark.parametrize("dataset", ["trans_review", "acra_by_ls_pt"])
def test_native_protocol_and_frozen_spec(model, method, dataset):
    _, _, spec = get_spec(model, method)
    before = deepcopy(spec)
    protocol = overrides_for(dataset, model, "archived")
    derived, runtime = apply_memory_profile(spec, protocol, "a100", model, method)
    assert spec == before
    # Native batch sizes, no accumulation rewrite, checkpointing exactly as the spec says.
    assert runtime == {**protocol, "save_total_limit": 3}
    assert derived["grad_checkpoint_enable"] == before["grad_checkpoint_enable"]
    for key in ("name_stem", "generation_config", "peft_config", "bnb_config", "training_args"):
        assert derived[key] == before[key]
    if "device_map" in before["model_kwargs"]:
        assert derived["model_kwargs"]["device_map"] == {"": 0}


@pytest.mark.parametrize("model,method", [("qwen25-14b", "lora"), ("qwen25", "galore"),
                                                   ("qwen25", "fmft"), ("qwen15", "fmft"),
                                                   ("codet5", "ptuning")])
def test_unsafe_training_is_rejected(model, method):
    _, _, spec = get_spec(model, method)
    with pytest.raises(ValueError, match="A100"):
        apply_memory_profile(spec, {}, "a100", model, method)


def test_unknown_profile_is_rejected():
    _, _, spec = get_spec("qwen25", "lora")
    with pytest.raises(ValueError, match="Unknown"):
        apply_memory_profile(spec, {}, "h100", "qwen25", "lora")


def test_default_profile_is_identity():
    _, _, spec = get_spec("qwen25", "lora")
    derived, overrides = apply_memory_profile(spec, {}, None, "qwen25", "lora")
    assert derived is spec
    assert overrides == {}
    assert metric_resources(None) is None
    assert metric_resources("a100") == {"device": "cuda", "batch_size": 64}


def test_profile_requires_separate_root():
    with pytest.raises(SystemExit, match="isolated"):
        main(["--model", "codet5", "--method", "fmft", "--memory-profile", "a100", "--dry-run"])


def test_profile_dry_run_reports_resolved_settings(capsys, tmp_path):
    import json
    main(["--model", "qwen25", "--method", "ia3", "--memory-profile", "a100",
          "--dataset", "trans_review", "--results-root", str(tmp_path), "--gen-batch-size", "16",
          "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory_profile"] == "a100"
    assert payload["protocol_overrides"]["per_device_train_batch_size"] == 2
    assert "gradient_accumulation_steps" not in payload["protocol_overrides"]
