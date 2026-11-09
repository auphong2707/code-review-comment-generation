import json
import math
import sys
import types
import importlib.util
from pathlib import Path

import pytest

from src.llm_as_judge.cache import make_cache_key
from src.llm_as_judge.cli import score_file
from src.llm_as_judge.config import JudgeConfig
from src.llm_as_judge.parser import parse_judge_output
from src.llm_as_judge.qwen_judge import score_record
from src.llm_as_judge.scorer import aggregate_scores, score_components


def test_parse_judge_output_accepts_valid_compact_json():
    assert parse_judge_output('{"c":4,"g":4,"a":3,"s":4,"q":5}') == {
        "c": 4,
        "g": 4,
        "a": 3,
        "s": 4,
        "q": 5,
    }


def test_parse_judge_output_extracts_first_json_object():
    assert parse_judge_output('Scores: {"c":5,"g":4,"a":3,"s":2,"q":1}\nDone.') == {
        "c": 5,
        "g": 4,
        "a": 3,
        "s": 2,
        "q": 1,
    }


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"c":0,"g":4,"a":3,"s":4,"q":5}',
        '{"c":4,"g":4,"a":3,"s":4}',
        '{"c":4,"g":4,"a":3,"s":4,"q":5,"extra":1}',
        '{"c":4.5,"g":4,"a":3,"s":4,"q":5}',
    ],
)
def test_parse_judge_output_rejects_invalid_json(text):
    with pytest.raises(ValueError):
        parse_judge_output(text)


def test_score_components_matches_weighted_formula_without_caps():
    scores = {"c": 4, "g": 4, "a": 3, "s": 4, "q": 5}
    expected = 0.35 * 4 + 0.25 * 4 + 0.20 * 3 + 0.10 * 4 + 0.10 * 5

    assert score_components(scores) == pytest.approx(expected)


def test_score_components_applies_hard_caps():
    assert score_components({"c": 2, "g": 5, "a": 5, "s": 5, "q": 5}) == 2.5
    assert score_components({"c": 5, "g": 5, "a": 2, "s": 5, "q": 5}) == 3.5
    assert score_components({"c": 5, "g": 5, "a": 5, "s": 5, "q": 2}) == 4.0


def test_aggregation_computes_summary_scores():
    result = aggregate_scores(
        [4.0, 4.0, 3.0],
        k=3,
        parse_errors=1,
        model_path="judge",
        prompt_version="v1",
        penalty_lambda=0.25,
    )

    assert result["qwen_judge_mean"] == pytest.approx(11 / 3)
    assert result["qwen_judge_median"] == 4.0
    assert result["qwen_judge_std"] == pytest.approx(0.4714045)
    assert result["qwen_judge_agreement"] == pytest.approx(2 / 3)
    assert result["qwen_judge_score"] == pytest.approx((11 / 3) - 0.25 * 0.4714045)
    assert result["qwen_judge_norm"] == pytest.approx((result["qwen_judge_score"] - 1) / 4)
    assert result["qwen_judge_valid_runs"] == 3
    assert result["qwen_judge_parse_errors"] == 1


def test_aggregation_returns_nan_when_all_runs_fail():
    result = aggregate_scores(
        [],
        k=5,
        parse_errors=5,
        model_path="judge",
        prompt_version="v1",
        penalty_lambda=0.25,
    )

    assert math.isnan(result["qwen_judge_score"])
    assert math.isnan(result["qwen_judge_mean"])
    assert result["qwen_judge_valid_runs"] == 0
    assert result["qwen_judge_parse_errors"] == 5


def test_cache_key_is_stable_and_sensitive_to_run_index():
    config = JudgeConfig(model_path="judge", k=2, temperature=0.0)
    record = {"code_or_diff": "code", "prediction": "pred", "reference": "ref"}

    assert make_cache_key(record, config, 0) == make_cache_key(dict(record), config, 0)
    assert make_cache_key(record, config, 0) != make_cache_key(record, config, 1)


def test_cache_key_changes_with_generation_settings():
    record = {"code_or_diff": "code", "prediction": "pred", "reference": "ref"}
    first = JudgeConfig(model_path="judge", temperature=0.0)
    second = JudgeConfig(model_path="judge", temperature=0.3)

    assert make_cache_key(record, first, 0) != make_cache_key(record, second, 0)


def test_cli_scores_jsonl_with_mock_judge(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "code_or_diff": "def f(): return 1",
                "prediction": "Return a named constant instead.",
                "reference": "Consider replacing the magic number.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    score_file(str(input_path), str(output_path), JudgeConfig(k=1, mock=True))

    output = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert output[0]["qwen_judge_score"] == 4.0
    assert output[0]["qwen_judge_valid_runs"] == 1
    assert output[0]["qwen_judge_parse_errors"] == 0
    assert "qwen_judge_raw_outputs" not in output[0]


def test_cli_scores_csv_with_explicit_columns(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    input_path.write_text(
        "src_col,pred_col,ref_col\n"
        "\"def f(): return 1\",Use a constant.,Avoid magic values.\n",
        encoding="utf-8",
    )

    score_file(
        str(input_path),
        str(output_path),
        JudgeConfig(
            k=1,
            mock=True,
            code_column="src_col",
            prediction_column="pred_col",
            reference_column="ref_col",
        ),
    )

    rows = output_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("src_col,pred_col,ref_col,qwen_judge_score")
    assert "4.0" in rows[1]


def test_missing_columns_raise_clear_value_error():
    with pytest.raises(ValueError, match="available columns"):
        score_record({"prediction": "comment", "reference": "target"}, JudgeConfig(mock=True))


def test_evaluation_scripts_import_with_judge_disabled(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    monkeypatch.syspath_prepend(str(src_root))
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.delitem(sys.modules, "llm_as_judge", raising=False)

    helper_module = types.ModuleType("utils.helper")
    helper_module.set_seed = lambda *args, **kwargs: None
    helper_module.load_model_and_tokenizer = lambda *args, **kwargs: None
    helper_module.load_dataset_for_eval = lambda *args, **kwargs: []
    helper_module.load_jsonl = lambda *args, **kwargs: []
    helper_module.extract_input_target = lambda *args, **kwargs: ("", "")
    helper_module.normalize_model_name = lambda name: name

    constants_module = types.ModuleType("utils.constants")
    constants_module.TEST_DATA_DIR = "test.jsonl"
    constants_module.TRAIN_DATA_DIR = "train.jsonl"
    constants_module.SEED = 42
    constants_module.DATASET_NAME = "acra_by_ls_pt"
    constants_module.OUTPUT_MAX_LENGTH = 256
    constants_module.OUTPUT_MIN_LENGTH = 2
    constants_module.NUM_BEAMS = 3
    constants_module.INPUT_MAX_LENGTH = 512
    constants_module.HUGGINGFACE_REPO_ID = "repo"

    utils_module = types.ModuleType("utils")
    utils_module.helper = helper_module
    utils_module.constants = constants_module

    src_utils_module = types.ModuleType("src.utils")
    src_utils_module.helper = helper_module
    src_utils_module.constants = constants_module

    evaluate_module = types.ModuleType("evaluate")
    evaluate_module.load = lambda *args, **kwargs: None

    torch_module = types.ModuleType("torch")
    torch_module.no_grad = lambda: pytest.fail("torch.no_grad should not run during import")
    torch_module.tensor = lambda *args, **kwargs: pytest.fail("torch.tensor should not run during import")
    torch_module.cuda = types.SimpleNamespace(
        amp=types.SimpleNamespace(
            autocast=lambda *args, **kwargs: pytest.fail("autocast should not run during import")
        )
    )
    torch_module.bfloat16 = object()

    transformers_module = types.ModuleType("transformers")

    class _NoLoadModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("model/tokenizer loading should not run during import")

    transformers_module.AutoModelForCausalLM = _NoLoadModel
    transformers_module.AutoModelForSeq2SeqLM = _NoLoadModel
    transformers_module.AutoTokenizer = _NoLoadModel

    huggingface_hub_module = types.ModuleType("huggingface_hub")
    huggingface_hub_module.login = lambda *args, **kwargs: None
    huggingface_hub_module.HfApi = lambda: pytest.fail("HfApi should not run during import")

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **kwargs: iterable

    monkeypatch.setitem(sys.modules, "utils", utils_module)
    monkeypatch.setitem(sys.modules, "utils.helper", helper_module)
    monkeypatch.setitem(sys.modules, "utils.constants", constants_module)
    monkeypatch.setitem(sys.modules, "src.utils", src_utils_module)
    monkeypatch.setitem(sys.modules, "src.utils.helper", helper_module)
    monkeypatch.setitem(sys.modules, "src.utils.constants", constants_module)
    monkeypatch.setitem(sys.modules, "evaluate", evaluate_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub_module)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)

    for module_name, script_name in (
        ("re_eval_model_disabled_smoke", "re_eval_model.py"),
        ("icl_inference_disabled_smoke", "icl_inference.py"),
    ):
        spec = importlib.util.spec_from_file_location(module_name, src_root / script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    assert "llm_as_judge" not in sys.modules
