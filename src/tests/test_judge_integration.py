import argparse
import math

import pytest

from src.llm_as_judge.cache import make_cache_key
from src.llm_as_judge.config import JudgeConfig
from src.llm_as_judge.integration import (
    JUDGE_COLUMNS,
    append_judge_results,
    append_judge_results_to_icl,
    build_judge_config,
    failed_judge_result,
    judge_rows,
    add_judge_arguments,
)
from src.llm_as_judge.qwen_judge import MockJudge, QwenJudge


def make_args(**overrides):
    parser = argparse.ArgumentParser()
    add_judge_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def sample_rows():
    rows = [
        {"prediction": "Add a null check.", "reference": "Handle null input."},
        {"prediction": "Rename this variable.", "reference": "Use a clearer name."},
        {"prediction": "Add a docstring.", "reference": "Document this."},
    ]
    samples = [
        {"patch": "def f(x): return x + 1", "msg": "Handle null input."},
        {"patch": "x = 1", "msg": "Use a clearer name."},
        {"patch": "def g(): pass", "msg": "Document this."},
    ]
    return rows, samples


def test_build_judge_config_forwards_device_batch_and_mock():
    args = make_args(judge_device="cpu", judge_batch_size=4, judge_mock=True,
                     judge_repair_retry=True, judge_emit_debug=True)
    config = build_judge_config(args)
    assert config.device == "cpu"
    assert config.batch_size == 4
    assert config.mock is True
    assert config.repair_retry is True
    assert config.emit_debug_columns is True
    # Column overrides stay in args (resolved by the glue), not in config.
    assert config.code_column is None


def test_judge_rows_mock_appends_all_columns():
    rows, samples = sample_rows()
    config = JudgeConfig(mock=True, device="cpu")
    judge_rows(rows, samples, "acra_by_ls_pt", config, make_args())
    for row in rows:
        for column in JUDGE_COLUMNS:
            assert column in row
        assert row["qwen_judge_score"] == pytest.approx(4.0)
        assert row["qwen_judge_valid_runs"] == config.k


def test_judge_rows_single_vs_batched_parity():
    config_b1 = JudgeConfig(mock=True, device="cpu", batch_size=1)
    config_b3 = JudgeConfig(mock=True, device="cpu", batch_size=3)

    rows_a, samples = sample_rows()
    rows_b, _ = sample_rows()
    judge_rows(rows_a, samples, "acra_by_ls_pt", config_b1, make_args())
    judge_rows(rows_b, samples, "acra_by_ls_pt", config_b3, make_args())
    assert rows_a == rows_b


def test_judge_rows_missing_sample_produces_failed_result():
    rows = [{"prediction": "p", "reference": "r"}]
    config = JudgeConfig(mock=True, device="cpu")
    # Empty raw sample -> no code column detectable -> failed_judge_result
    judge_rows(rows, [], "acra_by_ls_pt", config, make_args())
    assert math.isnan(rows[0]["qwen_judge_score"])
    assert rows[0]["qwen_judge_valid_runs"] == 0
    assert rows[0]["qwen_judge_parse_errors"] == config.k


def test_append_judge_results_reads_jsonl_and_scores(tmp_path):
    rows, samples = sample_rows()
    test_file = tmp_path / "test.jsonl"
    import json

    with open(test_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    config = JudgeConfig(mock=True, device="cpu")
    append_judge_results(rows, str(test_file), "acra_by_ls_pt", config, make_args())
    assert all(row["qwen_judge_score"] == pytest.approx(4.0) for row in rows)


def test_append_judge_results_to_icl_builds_raw_data(tmp_path):
    _, samples = sample_rows()
    test_file = tmp_path / "test.jsonl"
    import json

    with open(test_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    results = {
        "predictions": ["Add a null check.", "Rename this variable."],
        "references": ["Handle null input.", "Use a clearer name."],
    }
    config = JudgeConfig(mock=True, device="cpu")
    append_judge_results_to_icl(results, str(test_file), 2, "acra_by_ls_pt", config, make_args())
    assert len(results["raw_data"]) == 2
    assert results["raw_data"][0]["prediction"] == "Add a null check."
    assert results["raw_data"][0]["qwen_judge_score"] == pytest.approx(4.0)


def test_mock_judge_emit_debug_columns():
    config = JudgeConfig(mock=True, device="cpu", emit_debug_columns=True, k=3)
    result = MockJudge(config).score_record({"code_or_diff": "x", "prediction": "p", "reference": "r"})
    assert len(result["qwen_judge_raw_outputs"]) == 3
    assert result["qwen_judge_component_scores"][0] == {"c": 4, "g": 4, "a": 4, "s": 4, "q": 4}


def test_failed_judge_result_includes_debug_columns_when_enabled():
    config = JudgeConfig(mock=True, device="cpu", emit_debug_columns=True)
    result = failed_judge_result(config)
    assert result["qwen_judge_raw_outputs"] == []
    assert result["qwen_judge_component_scores"] == []
    config_off = JudgeConfig(mock=True, device="cpu")
    assert "qwen_judge_raw_outputs" not in failed_judge_result(config_off)


def _stubbed_qwen_judge(config, outputs_by_call):
    """QwenJudge with generation stubbed: outputs_by_call is a list of lists,
    one per _generate_batch/_generate_repair_batch invocation (in order)."""
    judge = QwenJudge(config)
    calls = {"n": 0, "repair": 0}

    def fake_generate_batch(records):
        result = outputs_by_call[calls["n"]]
        calls["n"] += 1
        return result[: len(records)]

    def fake_repair_batch(items):
        calls["repair"] += 1
        return ['{"c":3,"g":3,"a":3,"s":3,"q":3}'] * len(items)

    judge._generate_batch = fake_generate_batch
    judge._generate_repair_batch = fake_repair_batch
    return judge, calls


def test_repair_retry_off_counts_parse_error():
    config = JudgeConfig(k=1, device="cpu")
    judge, calls = _stubbed_qwen_judge(config, [["not json"]])
    result = judge.score_records([{"code_or_diff": "x", "prediction": "p", "reference": "r"}])[0]
    assert result["qwen_judge_parse_errors"] == 1
    assert result["qwen_judge_valid_runs"] == 0
    assert calls["repair"] == 0


def test_repair_retry_on_recovers_parse_failure():
    config = JudgeConfig(k=1, device="cpu", repair_retry=True)
    judge, calls = _stubbed_qwen_judge(config, [["not json"]])
    result = judge.score_records([{"code_or_diff": "x", "prediction": "p", "reference": "r"}])[0]
    assert calls["repair"] == 1
    assert result["qwen_judge_parse_errors"] == 0
    assert result["qwen_judge_valid_runs"] == 1
    # repaired output {"c":3,...} -> 3.0 weighted score
    assert result["qwen_judge_score"] == pytest.approx(3.0)


def test_qwen_judge_emit_debug_columns_with_repair():
    config = JudgeConfig(k=1, device="cpu", repair_retry=True, emit_debug_columns=True)
    judge, _ = _stubbed_qwen_judge(config, [["not json"]])
    result = judge.score_records([{"code_or_diff": "x", "prediction": "p", "reference": "r"}])[0]
    # both the failed raw output and the repaired one are recorded
    assert result["qwen_judge_raw_outputs"] == ["not json", '{"c":3,"g":3,"a":3,"s":3,"q":3}']
    assert result["qwen_judge_component_scores"] == [{"c": 3, "g": 3, "a": 3, "s": 3, "q": 3}]


GOLDEN_RECORD = {
    "code_or_diff": "def f(x): return x + 1",
    "prediction": "Add a docstring.",
    "reference": "Document this function.",
}
# Pinned 2026-08-26 against the config defaults that produced Tables 1-5.
GOLDEN_DEFAULT_KEY = "eb086069fdbb7703be0c7f9841ff5e6b0c8cb2b48a52c35e76d71ced13126560"


def test_default_cache_key_is_stable():
    assert make_cache_key(GOLDEN_RECORD, JudgeConfig(), 0) == GOLDEN_DEFAULT_KEY


def test_new_config_fields_do_not_change_cache_keys():
    with_new_fields = JudgeConfig(repair_retry=True, emit_debug_columns=True)
    assert make_cache_key(GOLDEN_RECORD, with_new_fields, 0) == GOLDEN_DEFAULT_KEY
