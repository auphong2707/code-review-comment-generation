"""Registry resolving (model, method) to a legacy-equivalent training spec.

Torch-free by design: `python -m src.train --dry-run` must work without any
heavy import (no torch, no logins).
"""

import os

from .specs import LEGACY_SPECS
from .specs_revision import derive_specs

# Legacy table + revision specs derived at import (D-026); LEGACY_SPECS stays
# importable for the parity harness
SPECS = {**LEGACY_SPECS, **derive_specs(LEGACY_SPECS)}

# CLI aliases; spec keys use the canonical names below
MODEL_ALIASES = {
    "qwen25-0.5b-instruct": "qwen25-0.5b",
    "qwen25_0.5b": "qwen25-0.5b",
    "qwen25_14b": "qwen25-14b",
}
METHOD_ALIASES = {"ptuning_v2": "ptuning"}


def supported_models():
    return sorted({model for model, _ in SPECS})


def supported_methods():
    return sorted({method for _, method in SPECS})


def get_spec(model, method):
    model = MODEL_ALIASES.get(model, model)
    method = METHOD_ALIASES.get(method, method)
    if (model, method) not in SPECS:
        if model not in supported_models():
            raise ValueError(f"unknown model '{model}'; choose from {supported_models()}")
        methods = sorted(me for mo, me in SPECS if mo == model)
        raise ValueError(
            f"unsupported combination: {model} + {method}; "
            f"supported methods for {model}: {methods}"
        )
    return model, method, SPECS[(model, method)]


def resolve_run(spec, experiment_id=None, learning_rate=None, seed=None, results_root=None):
    """Reproduce the legacy experiment naming exactly.

    Official runs: `<stem>_official_<str(lr)>` with lr the argparse float
    (str(5e-05) -> "5e-05", str(0.001) -> "0.001" — matches produced dirs).
    Sweep runs: `<stem>_<experiment_id>` with lr = LR[experiment_id].

    `seed` adds a `_seed<N>` suffix (only for non-default seeds; KI-012 —
    without it a rerun silently resumes the seed-42 checkpoint).
    `results_root` relocates the run dir (dataset lives in the root, never in
    the leaf name; D-026); default keeps the legacy string concat verbatim.
    """
    if learning_rate is not None:
        lr = learning_rate
        experiment_name = spec["name_stem"] + "_official_" + str(learning_rate)
    else:
        from src.utils.constants import LR

        lr = LR[experiment_id]
        experiment_name = f"{spec['name_stem']}_{experiment_id}"
    if seed is not None:
        experiment_name += f"_seed{seed}"
    if results_root:
        experiment_results_dir = os.path.join(results_root, experiment_name)
    else:
        experiment_results_dir = spec["results_dir"] + experiment_name
    return {
        "experiment_name": experiment_name,
        "learning_rate": lr,
        "experiment_results_dir": experiment_results_dir,
    }
