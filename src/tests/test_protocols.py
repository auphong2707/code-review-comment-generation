"""Protocol overrides (phase 03, D-025): table shape + harvest test.

The harvest test parses every archived training_args.txt under assets/ and
asserts it equals spec + overrides on the protocol-relevant keys — the
evidence that `--protocol archived` reproduces the produced TR artifacts and
that CodeReviewer runs match the HEAD specs untouched.
"""

import re
from pathlib import Path

import pytest

from src.training.protocols import ARCHIVED, PARENT_MODEL, overrides_for
from src.training.specs import LEGACY_SPECS

ASSETS_ROOTS = {
    "trans_review": Path("assets/models/review-code-generation"),
    "acra_by_ls_pt": Path("assets/models/review-code-generation-code-changes"),
}
PREFIXES = {"qwen25_0.5b": "qwen25-0.5b", "qwen25": "qwen25", "qwen15": "qwen15",
            "codet5": "codet5", "plbart": "plbart"}
HARVEST_KEYS = ("num_train_epochs", "per_device_train_batch_size",
                "per_device_eval_batch_size", "optim", "bf16", "fp16",
                "learning_rate", "lr_scheduler_type", "warmup_ratio",
                "max_grad_norm", "gradient_accumulation_steps")
# TrainingArguments defaults for keys the specs leave unset
TA_DEFAULTS = {"optim": "adamw_torch", "lr_scheduler_type": "linear",
               "warmup_ratio": 0.0, "max_grad_norm": 1.0,
               "gradient_accumulation_steps": 1, "bf16": False, "fp16": False}


def test_overrides_table_shape():
    for (dataset, model), row in ARCHIVED.items():
        assert dataset == "trans_review"
        assert set(row) == {"num_train_epochs", "per_device_train_batch_size",
                            "per_device_eval_batch_size"}
    for model, _ in LEGACY_SPECS:
        assert overrides_for("acra_by_ls_pt", model) == {}
        assert overrides_for("trans_review", model, protocol="head") == {}
        assert overrides_for("trans_review", model) == dict(
            ARCHIVED[("trans_review", model)])


def test_derived_models_inherit_parent_row():
    for child, parent in PARENT_MODEL.items():
        assert overrides_for("trans_review", child) == overrides_for("trans_review", parent)


def test_unknown_protocol_raises():
    with pytest.raises(ValueError, match="unknown protocol"):
        overrides_for("trans_review", "codet5", protocol="nope")


def parse_dir_name(name):
    for stem in sorted(PREFIXES, key=len, reverse=True):
        if name.startswith(stem + "_"):
            m = re.fullmatch(r"(\w+?)_official(?:_rank_\d+)?_([\d.e-]+)", name[len(stem) + 1:])
            if m:
                return PREFIXES[stem], m.group(1), float(m.group(2))
    return None


def parse_training_args_txt(path):
    values = {}
    for key in HARVEST_KEYS:
        m = re.search(rf"^{key}=(.*?),?$", path.read_text(errors="replace"), re.M)
        if not m:
            continue
        raw = m.group(1)
        if raw in ("True", "False"):
            values[key] = raw == "True"
        elif re.fullmatch(r"[A-Za-z][\w.]*\.[A-Z0-9_]+", raw):  # enum repr, e.g. OptimizerNames.ADAMW_TORCH
            values[key] = raw.rsplit(".", 1)[1].lower()
        else:
            try:
                values[key] = float(raw)
            except ValueError:
                values[key] = raw
    return values


def harvest_cases():
    cases = []
    for dataset, root in ASSETS_ROOTS.items():
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            args_txt = path / "training_args.txt"
            if path.is_dir() and args_txt.is_file() and parse_dir_name(path.name):
                cases.append(pytest.param(dataset, path, id=f"{dataset[:2]}-{path.name}"))
    return cases


@pytest.mark.parametrize("dataset,path", harvest_cases())
def test_harvested_args_match_spec_plus_overrides(dataset, path):
    model, method, lr = parse_dir_name(path.name)
    spec_args = LEGACY_SPECS[(model, method)]["training_args"]
    overrides = overrides_for(dataset, model)
    harvested = parse_training_args_txt(path / "training_args.txt")
    for key in HARVEST_KEYS:
        expected = overrides.get(key, spec_args.get(key, TA_DEFAULTS.get(key)))
        expected = lr if expected == "<LR>" else expected
        got = harvested.get(key, TA_DEFAULTS.get(key))
        if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
            assert float(got) == pytest.approx(float(expected)), f"{key}: {got} != {expected}"
        else:
            assert got == expected, f"{key}: {got} != {expected}"


def test_harvest_saw_assets():
    if not ASSETS_ROOTS["trans_review"].is_dir():
        pytest.skip("assets/ not present")
    assert len(harvest_cases()) >= 50  # 28 TR + 23 CR dirs with training_args.txt
