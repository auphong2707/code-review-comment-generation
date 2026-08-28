"""Phase-05 backend tests: prefix-aware columns, HFChatJudge generation
details, the jobs-file/subset/reuse driver paths, and the CR subset tool."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm_as_judge.config import JudgeConfig
from src.llm_as_judge.hf_chat_judge import HFChatJudge
from src.llm_as_judge.integration import failed_judge_result
from src.llm_as_judge.qwen_judge import MockJudge, QwenJudge, make_judge
from src.llm_as_judge.scorer import (
    DEFAULT_CAPS,
    DEFAULT_WEIGHTS,
    JUDGE_COLUMNS,
    judge_columns,
    score_components,
)


def test_judge_columns_prefix():
    assert JUDGE_COLUMNS == judge_columns("qwen_judge_")
    llama = judge_columns("llama_judge_")
    assert llama[0] == "llama_judge_score"
    assert len(llama) == 11 and all(c.startswith("llama_judge_") for c in llama)


def test_default_weights_and_caps_pinned():
    # frozen rubric literals (paper Tables 1-5)
    assert DEFAULT_WEIGHTS == {"c": 0.35, "g": 0.25, "a": 0.20, "s": 0.10, "q": 0.10}
    assert DEFAULT_CAPS == ((("c", "g"), 2, 2.5), (("a",), 2, 3.5), (("q",), 2, 4.0))
    scores = {"c": 4, "g": 4, "a": 3, "s": 4, "q": 5}
    assert score_components(scores) == pytest.approx(
        0.35 * 4 + 0.25 * 4 + 0.20 * 3 + 0.10 * 4 + 0.10 * 5)
    assert score_components({"c": 2, "g": 5, "a": 5, "s": 5, "q": 5}) == 2.5


def test_make_judge_dispatch():
    assert isinstance(make_judge(JudgeConfig(mock=True, backend="hf-chat")), MockJudge)
    assert isinstance(make_judge(JudgeConfig(backend="mock")), MockJudge)
    assert isinstance(make_judge(JudgeConfig(backend="hf-chat")), HFChatJudge)
    assert type(make_judge(JudgeConfig())) is QwenJudge


def test_prefixed_columns_end_to_end_with_mock():
    config = JudgeConfig(mock=True, k=1, column_prefix="llama_judge_")
    record = {"code_or_diff": "x", "prediction": "p", "reference": "r"}
    result = make_judge(config).score_record(record)
    assert result["llama_judge_score"] == 4.0
    assert not any(key.startswith("qwen_judge_") for key in result)
    failed = failed_judge_result(config)
    assert set(failed) == set(judge_columns("llama_judge_"))


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 7

    def __init__(self):
        self.last_add_special_tokens = None

    def __call__(self, prompts, padding=None, add_special_tokens=True, return_tensors=None):
        import torch

        self.last_add_special_tokens = add_special_tokens
        ids = torch.ones((len(prompts), 3), dtype=torch.long)
        return FakeBatch(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, skip_special_tokens=True):
        return '{"c":3,"g":3,"a":3,"s":3,"q":3}'


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=[7, 8])

    def __init__(self):
        self.last_kwargs = None

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        import torch

        self.last_kwargs = kwargs
        return torch.cat([input_ids, torch.full((input_ids.shape[0], 2), 5)], dim=1)


def make_stubbed_hf_chat(config):
    import torch

    judge = HFChatJudge(config)
    judge.tokenizer = FakeTokenizer()
    judge.model = FakeModel()
    judge._torch = torch
    judge._target_device = torch.device("cpu")
    return judge


def test_hf_chat_judge_generation_details():
    config = JudgeConfig(backend="hf-chat", top_k=20, repetition_penalty=1.05, seed=42)
    judge = make_stubbed_hf_chat(config)
    outputs = judge._generate_from_prompts(["p1", "p2"], run_index=1)
    assert outputs == ['{"c":3,"g":3,"a":3,"s":3,"q":3}'] * 2
    # no double BOS: the rendered chat template is tokenised without specials
    assert judge.tokenizer.last_add_special_tokens is False
    kwargs = judge.model.last_kwargs
    assert kwargs["eos_token_id"] == [7, 8]  # the generation-config list
    assert kwargs["top_k"] == 20
    assert kwargs["repetition_penalty"] == 1.05
    assert kwargs["do_sample"] is True and kwargs["temperature"] == config.temperature


def test_hf_chat_eos_falls_back_to_tokenizer():
    judge = make_stubbed_hf_chat(JudgeConfig(backend="hf-chat"))
    judge.model.generation_config = SimpleNamespace(eos_token_id=None)
    judge._generate_from_prompts(["p"])
    assert judge.model.last_kwargs["eos_token_id"] == [7]


# ---------------------------------------------------------------------------
# Driver: --jobs-file / --sample-index-file / --reuse-inputs with --mock
# ---------------------------------------------------------------------------

TR_RESULT = Path("assets/models/review-code-generation/codet5_lora_official_0.001/complete_results.json")


@pytest.mark.skipif(not TR_RESULT.is_file(), reason="assets/ not present")
def test_driver_jobs_file_subset_and_reuse(tmp_path, monkeypatch):
    import tools.update_llm_judge_tables as driver

    jobs = [{"tables": "rev", "source": "peft", "dataset": "trans_review",
             "model": "CodeT5", "method": "LoRA", "result": str(TR_RESULT),
             "kind": "complete"}]
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs))
    subset_file = tmp_path / "subset.json"
    subset_file.write_text(json.dumps({"trans_review": [0, 2, 5]}))

    def run(argv):
        monkeypatch.setattr(sys, "argv", ["update_llm_judge_tables.py"] + argv)
        driver.main()

    out_a = tmp_path / "out-a"
    run(["--mock", "--jobs-file", str(jobs_file), "--sample-index-file", str(subset_file),
         "--out-root", str(out_a)])
    scored = [json.loads(line) for line in
              (out_a / "scored/tables_rev_trans_review_codet5_lora_peft.jsonl").read_text().splitlines()]
    assert [row["sample_index"] for row in scored] == [0, 2, 5]
    assert scored[0]["qwen_judge_score"] == 4.0

    # reuse the inputs with the second-judge prefix: identical inputs, llama_ columns
    out_b = tmp_path / "out-b"
    run(["--mock", "--jobs-file", str(jobs_file), "--reuse-inputs", str(out_a / "inputs"),
         "--column-prefix", "llama_judge_", "--out-root", str(out_b)])
    scored_b = [json.loads(line) for line in
                (out_b / "scored/tables_rev_trans_review_codet5_lora_peft.jsonl").read_text().splitlines()]
    assert [row["sample_index"] for row in scored_b] == [0, 2, 5]
    assert scored_b[0]["llama_judge_score"] == 4.0
    assert "qwen_judge_score" not in scored_b[0]
    summary = (out_b / "summary.csv").read_text().splitlines()
    assert "llama_judge_score" in summary[0]


def test_cr_subset_stratification_deterministic():
    from tools.make_cr_judge_subset import stratified_indices

    langs = ["py"] * 60 + ["java"] * 40
    indices, per_lang = stratified_indices(langs, 10, seed=42)
    assert per_lang == {"java": 4, "py": 6}
    assert len(indices) == 10 == len(set(indices))
    assert indices == sorted(indices) and all(0 <= i < 100 for i in indices)
    assert stratified_indices(langs, 10, seed=42)[0] == indices
    assert stratified_indices(langs, 10, seed=1)[0] != indices
