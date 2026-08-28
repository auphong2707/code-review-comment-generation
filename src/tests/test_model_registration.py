"""Model-registration consistency (phase 07): every causal registry model must
be wired at all eight sites, so the *next* model addition cannot be partial.

Text-based checks are used for the three entry points whose lists live in
argparse/case statements (re_eval_model.py, icl_inference.py, run_training.sh).
"""

from pathlib import Path

import pytest

from src.training.registry import MODEL_ALIASES, SPECS, get_spec
from src.training.protocols import overrides_for
from src.utils.helper import CAUSAL_MODELS, normalize_model_name
from src.utils import constants

REPO = Path(__file__).resolve().parents[2]

# registry model key -> constants.py model-id constant
CONSTANTS_BY_MODEL = {
    "qwen15": "MODEL_QWEN15",
    "qwen25": "MODEL_QWEN25",
    "qwen25-0.5b": "MODEL_QWEN25_0_5B",
    "qwen25-14b": "MODEL_QWEN25_14B",
}

CAUSAL_KEYS = sorted({model for (model, _), spec in SPECS.items()
                      if spec["family"] == "causal"})


def test_expected_causal_models():
    assert CAUSAL_KEYS == ["qwen15", "qwen25", "qwen25-0.5b", "qwen25-14b"]


@pytest.mark.parametrize("model", CAUSAL_KEYS)
def test_eight_sites_agree(model):
    # 1. registry resolves the key and its underscore alias
    assert get_spec(model, "lora")[0] == model
    underscore = model.replace("-", "_")
    if underscore != model:
        assert MODEL_ALIASES.get(underscore, model) == model

    # 2. train.py choices (derived from supported_models, plus alias extras)
    from src.train import build_parser

    choices = next(a for a in build_parser()._actions if a.dest == "model").choices
    assert model in choices

    # 3. run_training.sh whitelist
    sh = (REPO / "run_training.sh").read_text()
    assert f'"{model}"' in sh

    # 4. helper: normalized name is a causal model
    assert normalize_model_name(model) in CAUSAL_MODELS

    # 5. re_eval_model.py: --model_name choices and the causal branch list
    re_eval = (REPO / "src/re_eval_model.py").read_text()
    assert f"'{model}'" in re_eval or f'"{model}"' in re_eval
    assert f'"{normalize_model_name(model)}"' in re_eval

    # 6. constants.py model block
    model_id = getattr(constants, CONSTANTS_BY_MODEL[model])
    assert model_id == SPECS[(model, "lora")]["model_id"]

    # 7. icl_inference.py model_mapping + choices
    icl = (REPO / "src/icl_inference.py").read_text()
    assert f'"{model}"' in icl

    # 8. protocols: an archived-TR row resolves (directly or via PARENT_MODEL)
    assert overrides_for("trans_review", model) != {}


def test_14b_spec_identity():
    for method in ("lora", "qlora"):
        spec = SPECS[("qwen25-14b", method)]
        assert spec["model_id"] == "Qwen/Qwen2.5-Coder-14B-Instruct"
        assert spec["name_stem"] == f"qwen25_14b_{method}"
        assert spec["results_dir"] == "assets/results/qwen25-14b/"
    assert SPECS[("qwen25-14b", "lora")]["training_args"]["gradient_checkpointing"] is True


def test_14b_diff_is_allowed():
    from src.training.specs_revision import ALLOWED_DIFF, VARIES, explain

    for method in ("lora", "qlora"):
        parent_key, diff = explain(("qwen25-14b", method))
        assert parent_key == ("qwen25", method)
        allowed = ALLOWED_DIFF["qwen25-14b"]
        assert set(diff) <= set(allowed)
        for path, value in diff.items():
            if allowed[path] != VARIES:
                assert value == allowed[path], path
