"""Registry resolving (model, method) to a legacy-equivalent training spec.

Torch-free by design: `python -m src.train --dry-run` must work without any
heavy import (no torch, no logins).
"""

from .specs import LEGACY_SPECS

# CLI aliases; spec keys use the canonical names below
MODEL_ALIASES = {
    "qwen25-0.5b-instruct": "qwen25-0.5b",
    "qwen25_0.5b": "qwen25-0.5b",
}
METHOD_ALIASES = {"ptuning_v2": "ptuning"}


def supported_models():
    return sorted({model for model, _ in LEGACY_SPECS})


def supported_methods():
    return sorted({method for _, method in LEGACY_SPECS})


def get_spec(model, method):
    model = MODEL_ALIASES.get(model, model)
    method = METHOD_ALIASES.get(method, method)
    if (model, method) not in LEGACY_SPECS:
        if model not in supported_models():
            raise ValueError(f"unknown model '{model}'; choose from {supported_models()}")
        methods = sorted(me for mo, me in LEGACY_SPECS if mo == model)
        raise ValueError(
            f"unsupported combination: {model} + {method}; "
            f"supported methods for {model}: {methods}"
        )
    return model, method, LEGACY_SPECS[(model, method)]


def resolve_run(spec, experiment_id=None, learning_rate=None):
    """Reproduce the legacy experiment naming exactly.

    Official runs: `<stem>_official_<str(lr)>` with lr the argparse float
    (str(5e-05) -> "5e-05", str(0.001) -> "0.001" — matches produced dirs).
    Sweep runs: `<stem>_<experiment_id>` with lr = LR[experiment_id].
    """
    if learning_rate:
        lr = learning_rate
        experiment_name = spec["name_stem"] + "_official_" + str(learning_rate)
    else:
        from src.utils.constants import LR

        lr = LR[experiment_id]
        experiment_name = f"{spec['name_stem']}_{experiment_id}"
    return {
        "experiment_name": experiment_name,
        "learning_rate": lr,
        "experiment_results_dir": spec["results_dir"] + experiment_name,
    }
