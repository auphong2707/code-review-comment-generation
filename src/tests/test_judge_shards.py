"""Server-smoke tests for opt-in whole-artifact sharding and final coverage gates."""
import json
import sys

import pytest

from tools import update_llm_judge_tables as driver


def test_shards_cover_every_artifact_once():
    jobs = [{"id": index} for index in range(133)]
    shards = [driver.shard_jobs(jobs, index, 48) for index in range(48)]
    assert all(2 <= len(shard) <= 3 for shard in shards)
    assert sorted(job["id"] for shard in shards for job in shard) == list(range(133))
    assert driver.shard_jobs(jobs, None, None) is jobs


@pytest.mark.parametrize("index,count", [(None, 2), (0, None), (-1, 2), (2, 2), (0, 0), (1, 2)])
def test_invalid_or_empty_shards_fail(index, count):
    with pytest.raises(ValueError):
        driver.shard_jobs([{"id": 0}], index, count)


@pytest.mark.parametrize("prefix", ["qwen_judge_", "llama_judge_"])
def test_final_gate_rejects_partial_stale_or_unscored_outputs(tmp_path, prefix):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "scored.jsonl"
    expected = {"prediction": "p", "reference": "r", "sample_index": 0}
    source.write_text(json.dumps(expected) + "\n")
    scored = dict(expected, **{
        prefix + key.removeprefix("qwen_judge_"): 1 for key in driver.JUDGE_COLUMNS
    })
    manifest = [{"job_id": "test", "input_path": str(source), "output_path": str(target)}]
    with pytest.raises(FileNotFoundError):
        driver.require_complete(manifest, prefix)
    for rows in [[], [expected], [dict(scored, prediction="stale")], [scored, scored]]:
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError):
            driver.require_complete(manifest, prefix)
    target.write_text(json.dumps(scored) + "\n")
    driver.require_complete(manifest, prefix)


def test_mock_shards_preserve_outputs_and_publish_full_summary(tmp_path, monkeypatch):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps({"code_tokens": ["return", "x"]}) + "\n")
    monkeypatch.setattr(driver, "DATASETS", {
        "trans_review": {"label": "TransReview", "test": dataset, "code_column": "code_tokens"}
    })
    result = tmp_path / "complete.json"
    result.write_text(json.dumps({"raw_data": [{"prediction": "p", "reference": "r"}]}))
    jobs = [{"tables": "revision", "source": "peft", "dataset": "trans_review",
             "model": "test", "method": f"method{i}", "result": str(result), "kind": "complete"}
            for i in range(3)]
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs))
    out = tmp_path / "out"

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["driver", "--mock", "--judge-k", "3",
                            "--jobs-file", str(jobs_file), "--out-root", str(out), *extra])
        driver.main()

    run("--job-shard-index", "0", "--job-shard-count", "2")
    first_outputs = {path.name: path.read_bytes() for path in (out / "scored").glob("*.jsonl")}
    assert len(first_outputs) == 2
    assert not (out / "summary.csv").exists()
    with pytest.raises(FileNotFoundError):
        run("--summary-only", "--require-complete")
    run("--job-shard-index", "1", "--job-shard-count", "2")
    assert all((out / "scored" / name).read_bytes() == content for name, content in first_outputs.items())
    assert len(list(out.glob("manifest_all_shard_*.jsonl"))) == 2
    run("--summary-only", "--require-complete")
    assert len((out / "summary.csv").read_text().splitlines()) == 4
