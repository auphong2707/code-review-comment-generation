"""Revision specs derived at import from LEGACY_SPECS (phase 06, D-026).

`specs.py` stays byte-identical; new methods are deepcopies of their parent
spec with a small, test-enforced diff (`ALLOWED_DIFF`):

- `dora` (D-024): the parent LoRA spec with `use_dora=True`, for all five models.
- `galore` (D-021): memory-efficient full-parameter FT from the parent FMFT
  spec — deepspeed dropped, `optim="galore_adamw"` over the attention/MLP
  projections, gradient checkpointing on. Primary subject is qwen25
  (Qwen2.5-Coder-7B); qwen15 optional; qwen25-0.5b for local smokes.

`PEFT_PIN` (applied as a runtime override, never into the spec dict) pins the
archived PEFT-0.16 IA3 module resolution for TransReview reruns (KI-015 note:
codet5 IA3 resolved to ['v','wo','k']/['wo']).
"""

from copy import deepcopy

REMOVED = "<removed>"
VARIES = "<varies>"

DORA_MODELS = ("codet5", "plbart", "qwen15", "qwen25", "qwen25-0.5b")
GALORE_MODELS = ("qwen25", "qwen15", "qwen25-0.5b")
QWEN25_14B_MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
QWEN25_14B_METHODS = ("lora", "qlora")
GALORE_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
GALORE_OPTIM_ARGS = "rank=128,update_proj_gap=200,scale=0.25"

# Every path a derived spec may differ from its parent on (test-enforced;
# not every path must appear — e.g. deepspeed only exists on the 7B parents)
ALLOWED_DIFF = {
    "dora": {
        "name_stem": VARIES,
        "peft_config.kwargs.use_dora": True,
    },
    "galore": {
        "name_stem": VARIES,
        "training_args.deepspeed": REMOVED,
        "training_args.optim": "galore_adamw",
        "training_args.optim_target_modules": GALORE_TARGET_MODULES,
        "training_args.optim_args": GALORE_OPTIM_ARGS,
        "training_args.gradient_checkpointing": True,
        "grad_checkpoint_enable": True,
    },
    # qwen25-14b (D-022, phase 07): the qwen25 spec scaled to 14B; only the
    # model identity, naming/output location, and (for LoRA) gradient
    # checkpointing for VRAM headroom may change
    "qwen25-14b": {
        "model_id": QWEN25_14B_MODEL_ID,
        "tokenizer_id": QWEN25_14B_MODEL_ID,
        "name_stem": VARIES,
        "results_dir": "assets/results/qwen25-14b/",
        "training_args.gradient_checkpointing": True,
        "grad_checkpoint_enable": True,
    },
}

# (dataset, model, method) -> peft_config.kwargs overrides for reruns
PEFT_PIN = {
    ("trans_review", "codet5", "ia3"): {
        "target_modules": ["v", "wo", "k"],
        "feedforward_modules": ["wo"],
    },
}


def _replace_stem(stem, old, new):
    if not stem.endswith(old):
        raise ValueError(f"cannot derive spec: stem '{stem}' does not end in '{old}'")
    return stem[: -len(old)] + new


def derive_specs(legacy):
    specs = {}
    for model in DORA_MODELS:
        parent = legacy[(model, "lora")]
        spec = deepcopy(parent)
        spec["peft_config"]["kwargs"]["use_dora"] = True
        spec["name_stem"] = _replace_stem(parent["name_stem"], "_lora", "_dora")
        specs[(model, "dora")] = spec

    for model in GALORE_MODELS:
        parent = legacy[(model, "fmft")]
        spec = deepcopy(parent)
        spec["training_args"].pop("deepspeed", None)
        spec["training_args"].update(
            optim="galore_adamw",
            optim_target_modules=GALORE_TARGET_MODULES,
            optim_args=GALORE_OPTIM_ARGS,
            gradient_checkpointing=True,
        )
        spec["grad_checkpoint_enable"] = True
        spec["name_stem"] = _replace_stem(parent["name_stem"], "_fmft", "_galore")
        specs[(model, "galore")] = spec

    for method in QWEN25_14B_METHODS:
        parent = legacy[("qwen25", method)]
        spec = deepcopy(parent)
        spec["model_id"] = QWEN25_14B_MODEL_ID
        spec["tokenizer_id"] = QWEN25_14B_MODEL_ID
        spec["name_stem"] = f"qwen25_14b_{method}"
        spec["results_dir"] = "assets/results/qwen25-14b/"
        if method == "lora":
            spec["training_args"]["gradient_checkpointing"] = True
            spec["grad_checkpoint_enable"] = True
        specs[("qwen25-14b", method)] = spec

    return specs


def peft_pin_for(dataset_name, model_key, method):
    return dict(PEFT_PIN.get((dataset_name, model_key, method), {}))


def _flatten(mapping, prefix=""):
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def spec_diff(parent, derived):
    """Flat {path: new_value_or_REMOVED} of every difference to the parent."""
    flat_parent = _flatten(parent)
    flat_derived = _flatten(derived)
    diff = {}
    for path in set(flat_parent) | set(flat_derived):
        if flat_parent.get(path, REMOVED) != flat_derived.get(path, REMOVED):
            diff[path] = flat_derived.get(path, REMOVED)
    return diff


def explain(spec_key):
    """(parent_key, diff) for a derived spec key — the D-026 audit trail."""
    from .specs import LEGACY_SPECS

    model, method = spec_key
    if model == "qwen25-14b":
        parent_key = ("qwen25", method)
    else:
        parent_key = (model, {"dora": "lora", "galore": "fmft"}[method])
    derived = derive_specs(LEGACY_SPECS)[spec_key]
    return parent_key, spec_diff(LEGACY_SPECS[parent_key], derived)
