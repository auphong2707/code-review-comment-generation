"""Resources block (phase 03): callback hooks must exist, builder is total;
batched validation generation must mask padding like the per-sample path."""

from collections import namedtuple
from types import SimpleNamespace

from src.training.resources import collect_resources


def test_callback_hooks_exist_on_trainer_callback():
    # The pre-phase-03 bug class: on_evaluate_begin/on_evaluate_end are not
    # TrainerCallback hooks and silently never fire
    from transformers import TrainerCallback

    from src.utils.custom_modified_api import TimeMemoryCallback

    for name in vars(TimeMemoryCallback):
        if name.startswith("on_"):
            assert hasattr(TrainerCallback, name), f"{name} is not a TrainerCallback hook"


def test_collect_resources_from_stub(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x" * 128)
    (tmp_path / "notes.txt").write_text("ignored")
    TrainOutput = namedtuple("TrainOutput", "global_step training_loss metrics")
    out = TrainOutput(2, 0.5, {"train_runtime": 12.5, "train_samples_per_second": 3.2})
    summary = {"train_step_time_s": [1.0, 1.1], "eval_time_s": [0.4],
               "peak_vram_allocated_gb": 2.5, "peak_vram_reserved_gb": 3.0}

    res = collect_resources(out, summary, test_generation_time_s=7.0,
                            trainable_params=10, total_params=1000, hardware="stub-gpu",
                            experiment_name="codet5_lora_official_5e-05_seed1",
                            dataset="trans_review", seed=1, protocol="archived",
                            protocol_overrides={"num_train_epochs": 15},
                            eval_max_samples=None, gen_batch_size=4,
                            results_dir=str(tmp_path))

    assert res["train_runtime_s"] == 12.5
    assert res["train_step_time_s"] == [1.0, 1.1]
    assert res["trainable_pct"] == 1.0
    assert res["adapter_size_bytes"] == 128
    assert res["peak_vram_reserved_gb"] == 3.0
    assert res["protocol_overrides"] == {"num_train_epochs": 15}
    assert res["seed"] == 1 and res["gen_batch_size"] == 4
    assert res["dataset"] == "trans_review" and res["hardware"] == "stub-gpu"


def test_generate_batched_masks_all_padding():
    """The batched val path must mask both each row's own left padding and the
    chunk padding — matching the per-sample path, where generate() infers
    input_ids.ne(pad) because pad != eos for the qwen tokenizers."""
    import torch

    from src.utils.custom_modified_api import LLMCustomTrainer

    trainer = object.__new__(LLMCustomTrainer)  # bypass Trainer.__init__
    trainer.tokenizer_gen = SimpleNamespace(
        pad_token_id=0, decode=lambda ids, skip_special_tokens=True: "text")
    trainer.args = SimpleNamespace(device="cpu")
    trainer.generation_config = SimpleNamespace(to_dict=lambda: {"max_new_tokens": 2})
    captured = {}

    def fake_generate(input_ids=None, attention_mask=None, **kwargs):
        captured["attention_mask"] = attention_mask
        return torch.cat([input_ids, torch.full((input_ids.shape[0], 2), 5)], dim=1)

    trainer.model = SimpleNamespace(generate=fake_generate)
    prompts = [torch.tensor([0, 0, 7, 8]),  # row with its own left padding
               torch.tensor([9])]           # short row -> chunk padding added
    texts = trainer._generate_batched(prompts)
    assert captured["attention_mask"].tolist() == [[0, 0, 1, 1], [0, 0, 0, 1]]
    assert texts == ["text", "text"]
