"""Resolve spec markers and build the real config objects.

Spec values are plain data with `<marker>` strings (see specs.py); everything
torch/transformers/peft is imported lazily inside the builders so that
registry/spec handling stays import-light.
"""

import os


def static_marker_context(run_info, spec):
    """Markers resolvable before any model/tokenizer exists."""
    import torch
    from peft import PeftType, TaskType

    return {
        "<torch.bfloat16>": torch.bfloat16,
        "<TaskType.CAUSAL_LM>": TaskType.CAUSAL_LM,
        "<TaskType.SEQ_2_SEQ_LM>": TaskType.SEQ_2_SEQ_LM,
        "<PeftType.P_TUNING>": PeftType.P_TUNING,
        "<EXPERIMENT_NAME>": run_info["experiment_name"],
        "<LR>": run_info["learning_rate"],
        "<EXPERIMENT_RESULTS_DIR>": run_info["experiment_results_dir"],
        "<EXPERIMENT_RESULTS_DIR>/logs": run_info["experiment_results_dir"] + "/logs",
        "<RESULTS_DIR>": spec["results_dir"],
    }


def add_tokenizer_markers(ctx, tokenizer):
    ctx["<tokenizer.eos_token_id>"] = tokenizer.eos_token_id
    ctx["<tokenizer.pad_token_id>"] = tokenizer.pad_token_id
    return ctx


def resolve_markers(value, ctx):
    if isinstance(value, dict):
        return {key: resolve_markers(item, ctx) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_markers(item, ctx) for item in value]
    if isinstance(value, str) and value in ctx:
        return ctx[value]
    return value


def build_bnb_config(spec, ctx):
    if not spec["bnb_config"]:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(**resolve_markers(spec["bnb_config"], ctx))


def build_peft_config(spec, ctx):
    if not spec["peft_config"]:
        return None
    import peft

    cls = getattr(peft, spec["peft_config"]["class"])
    return cls(**resolve_markers(spec["peft_config"]["kwargs"], ctx))


def build_generation_config(spec, ctx):
    if not spec["generation_config"]:
        return None
    from transformers import GenerationConfig

    return GenerationConfig(**resolve_markers(spec["generation_config"], ctx))


def build_training_args(spec, ctx, smoke=False):
    import transformers

    cls = getattr(transformers, spec["training_args_class"])
    kwargs = resolve_markers(spec["training_args"], ctx)

    if "<GENERATION_CONFIG>" in ctx and kwargs.get("generation_config") == "<GENERATION_CONFIG>":
        kwargs["generation_config"] = ctx["<GENERATION_CONFIG>"]

    if "deepspeed" in kwargs:
        if smoke:
            # deepspeed needs its launcher; never used for smoke runs
            kwargs.pop("deepspeed")
        elif not os.path.exists(kwargs["deepspeed"]):
            raise FileNotFoundError(
                f"deepspeed config '{kwargs['deepspeed']}' does not exist (KI-003: the "
                "original ZeRO-3 json was never committed). Provide the file or run a "
                "non-deepspeed combination."
            )

    if smoke:
        kwargs.update(
            max_steps=2,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )

    return cls(**kwargs)


def load_base_model(spec, ctx, attn_implementation=None):
    import transformers

    cls = getattr(transformers, spec["model_class"])
    kwargs = resolve_markers(spec["model_kwargs"], ctx)
    bnb = build_bnb_config(spec, ctx)
    if bnb is not None:
        kwargs["quantization_config"] = bnb
    if attn_implementation and "attn_implementation" in kwargs:
        kwargs["attn_implementation"] = attn_implementation
    return cls.from_pretrained(spec["model_id"], **kwargs)


def latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if not ckpts:
        return None
    last = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(output_dir, last)
