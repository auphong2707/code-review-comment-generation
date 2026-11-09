"""Opt-in A100 80 GB profile, separate from the frozen experiment specs."""

from copy import deepcopy

PROFILES = ("a100",)
# Corpus metrics run on the GPU in batches (same metric semantics, recorded in resources).
METRIC_RESOURCES = {"a100": {"device": "cuda", "batch_size": 64}}


def metric_resources(profile):
    """Metric device/batch for a profile; None keeps the legacy library defaults."""
    return METRIC_RESOURCES.get(profile) if profile else None


def apply_memory_profile(spec, protocol_overrides, profile, model, method):
    """Return a copied spec and overrides for the opt-in A100 profile.

    The native protocol: frozen spec/protocol batch sizes, no accumulation rewrite,
    gradient checkpointing only where the spec itself enables it, the model pinned to
    the allocated GPU. Not a certificate of fit: lifecycle probes qualify each campaign
    configuration on the target GPU.
    """
    if profile is None:
        return spec, dict(protocol_overrides)
    if profile not in PROFILES:
        raise ValueError(f"Unknown memory profile: {profile}")
    if model == "qwen25-14b" and method != "qlora":
        raise ValueError("A100: 14B BF16 training is not admitted; only probe-qualified QLoRA")
    if model in ("qwen15", "qwen25") and method == "fmft":
        raise ValueError("A100: 7B full fine-tuning needs the missing ZeRO-3 config (KI-003)")
    if method == "ptuning":
        raise ValueError("A100: experimental P-Tuning is outside the qualified campaign methods")
    if method == "galore":
        raise ValueError("A100: GaLore requires a separately qualified optimizer/resume path")

    derived = deepcopy(spec)
    # Training must stay on the allocated device, never inference auto-offload.
    if "device_map" in derived["model_kwargs"]:
        derived["model_kwargs"]["device_map"] = {"": 0}
    # Keep the latest recovery point alongside the best epoch checkpoint.
    return derived, {**protocol_overrides, "save_total_limit": 3}
