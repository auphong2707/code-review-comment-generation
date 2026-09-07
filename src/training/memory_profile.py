"""Opt-in hardware overrides, separate from the frozen experiment specs."""

from copy import deepcopy


def apply_memory_profile(spec, protocol_overrides, profile, model, method):
    """Return a copied spec and overrides; preserve effective single-GPU batch.

    These are conservative starting settings, not a certificate of GPU fit.
    Lifecycle probes must qualify each campaign configuration on the target GPU.
    """
    if profile is None:
        return spec, dict(protocol_overrides)
    if profile != "a5000":
        raise ValueError(f"Unknown memory profile: {profile}")
    if model == "qwen25-14b" and method != "qlora":
        raise ValueError("A5000: 14B BF16 weights exceed 24 GB; only probe-qualified QLoRA is admitted")
    if model in ("qwen15", "qwen25") and method == "fmft":
        raise ValueError("A5000: unoffloaded 7B full fine-tuning is not admitted")
    if method == "ptuning":
        raise ValueError("A5000: experimental P-Tuning is outside the qualified campaign methods")
    if method == "galore":
        raise ValueError("A5000: GaLore requires a separately qualified optimizer/resume path")

    derived = deepcopy(spec)
    resolved = {**spec["training_args"], **protocol_overrides}
    effective_batch = (resolved["per_device_train_batch_size"]
                       * resolved.get("gradient_accumulation_steps", 1))
    overrides = {
        **protocol_overrides,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": effective_batch,
        "eval_accumulation_steps": 1,
        "save_total_limit": 3,  # retain latest recovery as well as best epoch checkpoint
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    derived["grad_checkpoint_enable"] = True
    # Training must stay on the allocated device, never inference auto-offload.
    if "device_map" in derived["model_kwargs"]:
        derived["model_kwargs"]["device_map"] = {"": 0}
    return derived, overrides
