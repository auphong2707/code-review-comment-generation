import pytest
from src.llm_as_judge.config import JudgeConfig
from src.llm_as_judge.qwen_judge import QwenJudge
from src.llm_as_judge.cache import make_cache_key


def test_guard_does_not_change_cache_identity():
    record = {"code_or_diff": "x", "prediction": "p", "reference": "r"}
    assert make_cache_key(record, JudgeConfig(), 0) == make_cache_key(
        record, JudgeConfig(input_token_limit=4096), 0)


def test_budget_rejects_before_gpu_transfer():
    from types import SimpleNamespace
    judge = QwenJudge(JudgeConfig(input_token_limit=4))
    with pytest.raises(ValueError, match="qualified"):
        judge._check_input_budget({"input_ids": SimpleNamespace(shape=(1, 5))})
    assert judge._check_input_budget({"input_ids": SimpleNamespace(shape=(1, 4))})


def test_long_rows_split_the_batch_into_qualified_shapes():
    """Qualified shapes: 16 rows x 4096 tokens, or one row up to 4 x 4096. A padded call
    outside them is halved (order kept) until it fits; a longer single row is refused."""
    from types import SimpleNamespace
    judge = QwenJudge(JudgeConfig(input_token_limit=4096, batch_size=16))
    fits = lambda rows, length: judge._check_input_budget({"input_ids": SimpleNamespace(shape=(rows, length))})
    assert fits(16, 4096) and fits(1, 16384) and fits(1, 5000)
    assert not fits(16, 4097) and not fits(4, 16384) and not fits(2, 4097) and not fits(17, 100)
    with pytest.raises(ValueError, match="16384"):
        fits(1, 16385)
    lengths = [100] * 15 + [10000]
    calls = []

    def generate(prompts, **kwargs):
        if not fits(len(prompts), max(lengths[p] for p in prompts)):
            return judge._split_generate(prompts, **kwargs)
        calls.append(len(prompts))
        return [f"out{p}" for p in prompts]

    judge._generate_from_prompts = generate
    assert generate(list(range(16)), run_index=0) == [f"out{p}" for p in range(16)]
    assert calls == [8, 4, 2, 1, 1]


def test_qualified_driver_does_not_publish_failed_generation(tmp_path, monkeypatch):
    import tools.update_llm_judge_tables as driver
    import json
    source = tmp_path / "input.jsonl"
    output = tmp_path / "scored.jsonl"
    source.write_text(json.dumps({"code_or_diff": "x", "prediction": "p", "reference": "r"}) + "\n")
    class FailedJudge:
        def score_records(self, rows):
            raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(driver, "make_judge", lambda _: FailedJudge())
    job = {"input_path": str(source), "output_path": str(output), "dataset": "TransReview",
           "model": "test", "method": "lora", "source": "peft", "job_id": "test", "tables": [1]}
    with pytest.raises(RuntimeError, match="out of memory"):
        driver.score_manifest([job], JudgeConfig(input_token_limit=4096))
    assert not output.exists()
