#!/usr/bin/env python3
"""Build and score LLM-as-Judge inputs for proposal Tables 1-5."""

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_as_judge import JudgeConfig
from src.llm_as_judge.qwen_judge import MockJudge, QwenJudge


DEFAULT_JUDGE_MODEL = "assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUT_ROOT = "assets/models/llm-as-judge/table-results"
JOB_GROUPS = (
    "all",
    "trans-review-finetuned",
    "code-reviewer-finetuned",
    "trans-review-icl",
    "code-reviewer-icl",
)
JUDGE_COLUMNS = (
    "qwen_judge_score",
    "qwen_judge_mean",
    "qwen_judge_median",
    "qwen_judge_std",
    "qwen_judge_agreement",
    "qwen_judge_norm",
    "qwen_judge_k",
    "qwen_judge_valid_runs",
    "qwen_judge_parse_errors",
    "qwen_judge_model",
    "qwen_judge_prompt_version",
)
LOGGER = logging.getLogger("llm_judge_tables")

DATASETS = {
    "trans_review": {
        "label": "TransReview",
        "test": Path("assets/data/Trans_Review/test.jsonl"),
        "code_column": "code_tokens",
    },
    "code_reviewer": {
        "label": "CodeReviewer",
        "test": Path("assets/data/Code_Review/msg-test.jsonl"),
        "code_column": "patch",
    },
}

TR = "assets/models/review-code-generation"
CR = "assets/models/review-code-generation-code-changes"

JOBS = [
    # Table 1 PEFT rows, TransReview.
    {"tables": "1,2", "source": "peft", "dataset": "trans_review", "model": "CodeT5", "method": "IA3", "result": f"{TR}/codet5_ia3_official_0.0001/complete_results.json", "kind": "complete"},
    {"tables": "1,2", "source": "peft", "dataset": "trans_review", "model": "CodeT5", "method": "LoRA", "result": f"{TR}/codet5_lora_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,2", "source": "peft", "dataset": "trans_review", "model": "CodeT5", "method": "QLoRA", "result": f"{TR}/codet5_qlora_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,2", "source": "peft", "dataset": "trans_review", "model": "PLBART", "method": "IA3", "result": f"{TR}/plbart_ia3_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,2", "source": "peft", "dataset": "trans_review", "model": "PLBART", "method": "LoRA", "result": f"{TR}/plbart_lora_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,2", "source": "peft", "dataset": "trans_review", "model": "PLBART", "method": "QLoRA", "result": f"{TR}/plbart_qlora_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,2,4", "source": "peft", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "IA3", "result": f"{TR}/qwen25_0.5b_ia3_official_0.0005/complete_results.json", "kind": "complete"},
    {"tables": "1,2,4", "source": "peft", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "LoRA", "result": f"{TR}/qwen25_0.5b_lora_official_0.0005/complete_results.json", "kind": "complete"},
    {"tables": "1,2,4", "source": "peft", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "QLoRA", "result": f"{TR}/qwen25_0.5b_qlora_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,4", "source": "peft", "dataset": "trans_review", "model": "Qwen2.5-7B", "method": "IA3", "result": f"{TR}/qwen25_ia3_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,4", "source": "peft", "dataset": "trans_review", "model": "Qwen2.5-7B", "method": "LoRA", "result": f"{TR}/qwen25_lora_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,4", "source": "peft", "dataset": "trans_review", "model": "Qwen2.5-7B", "method": "QLoRA", "result": f"{TR}/qwen25_qlora_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,4", "source": "peft", "dataset": "trans_review", "model": "CodeQwen1.5-7B", "method": "IA3", "result": f"{TR}/qwen15_ia3_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,4", "source": "peft", "dataset": "trans_review", "model": "CodeQwen1.5-7B", "method": "LoRA", "result": f"{TR}/qwen15_lora_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,4", "source": "peft", "dataset": "trans_review", "model": "CodeQwen1.5-7B", "method": "QLoRA", "result": f"{TR}/qwen15_qlora_official_5e-05/complete_results.json", "kind": "complete"},

    # Table 1 PEFT rows, CodeReviewer.
    {"tables": "1,3", "source": "peft", "dataset": "code_reviewer", "model": "CodeT5", "method": "IA3", "result": f"{CR}/codet5_ia3_official_0.0001/complete_results.json", "kind": "complete"},
    {"tables": "1,3", "source": "peft", "dataset": "code_reviewer", "model": "CodeT5", "method": "LoRA", "result": f"{CR}/codet5_lora_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,3", "source": "peft", "dataset": "code_reviewer", "model": "CodeT5", "method": "QLoRA", "result": f"{CR}/codet5_qlora_official_0.0005/complete_results.json", "kind": "complete"},
    {"tables": "1,3", "source": "peft", "dataset": "code_reviewer", "model": "PLBART", "method": "IA3", "result": f"{CR}/plbart_ia3_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,3", "source": "peft", "dataset": "code_reviewer", "model": "PLBART", "method": "LoRA", "result": f"{CR}/plbart_lora_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,3", "source": "peft", "dataset": "code_reviewer", "model": "PLBART", "method": "QLoRA", "result": f"{CR}/plbart_qlora_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,3,5", "source": "peft", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "IA3", "result": f"{CR}/qwen25_0.5b_ia3_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,3,5", "source": "peft", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "LoRA", "result": f"{CR}/qwen25_0.5b_lora_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,3,5", "source": "peft", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "QLoRA", "result": f"{CR}/qwen25_0.5b_qlora_official_5e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,5", "source": "peft", "dataset": "code_reviewer", "model": "Qwen2.5-7B", "method": "IA3", "result": f"{CR}/qwen25_ia3_official_0.001/complete_results.json", "kind": "complete"},
    {"tables": "1,5", "source": "peft", "dataset": "code_reviewer", "model": "Qwen2.5-7B", "method": "LoRA", "result": f"{CR}/qwen25_lora_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,5", "source": "peft", "dataset": "code_reviewer", "model": "Qwen2.5-7B", "method": "QLoRA", "result": f"{CR}/qwen25_qlora_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,5", "source": "peft", "dataset": "code_reviewer", "model": "CodeQwen1.5-7B", "method": "IA3", "result": f"{CR}/qwen15_ia3_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "1,5", "source": "peft", "dataset": "code_reviewer", "model": "CodeQwen1.5-7B", "method": "LoRA", "result": f"{CR}/qwen15_lora_official_0.0005/complete_results.json", "kind": "complete"},
    {"tables": "1,5", "source": "peft", "dataset": "code_reviewer", "model": "CodeQwen1.5-7B", "method": "QLoRA", "result": f"{CR}/qwen15_qlora_official_1e-05/complete_results.json", "kind": "complete"},

    # Table 2 FMFT rows, TransReview.
    {"tables": "2", "source": "fmft", "dataset": "trans_review", "model": "CodeT5", "method": "FMFT", "result": f"{TR}/codet5_fmft_official_0.0005/complete_results.json", "kind": "complete"},
    {"tables": "2", "source": "fmft", "dataset": "trans_review", "model": "PLBART", "method": "FMFT", "result": f"{TR}/plbart_fmft_official_0.0001/complete_results.json", "kind": "complete"},
    {"tables": "2", "source": "fmft", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "FMFT", "result": f"{TR}/qwen25_0.5b_fmft_official_5e-05/complete_results.json", "kind": "complete"},

    # Table 3 FMFT rows, CodeReviewer.
    {"tables": "3", "source": "fmft", "dataset": "code_reviewer", "model": "CodeT5", "method": "FMFT", "result": f"{CR}/codet5_fmft_official_0.0001/complete_results.json", "kind": "complete"},
    {"tables": "3", "source": "fmft", "dataset": "code_reviewer", "model": "PLBART", "method": "FMFT", "result": f"{CR}/plbart_fmft_official_1e-05/complete_results.json", "kind": "complete"},
    {"tables": "3", "source": "fmft", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "FMFT", "result": f"{CR}/qwen25_0.5b_fmft_official_5e-05/complete_results.json", "kind": "complete"},

    # Table 4 ICL rows, TransReview.
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "CodeQwen1.5-7B", "method": "3-shot", "result": f"{TR}/qwen15_icl_3ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "CodeQwen1.5-7B", "method": "5-shot", "result": f"{TR}/qwen15_icl_5ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "CodeQwen1.5-7B", "method": "10-shot", "result": f"{TR}/qwen15_icl_10ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "3-shot", "result": f"{TR}/qwen25-0.5b_icl_3ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "5-shot", "result": f"{TR}/qwen25-0.5b_icl_5ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "Qwen2.5-0.5B", "method": "10-shot", "result": f"{TR}/qwen25-0.5b_icl_10ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "Qwen2.5-7B", "method": "3-shot", "result": f"{TR}/qwen25_icl_3ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "Qwen2.5-7B", "method": "5-shot", "result": f"{TR}/qwen25_icl_5ex_random.json", "kind": "icl"},
    {"tables": "4", "source": "icl", "dataset": "trans_review", "model": "Qwen2.5-7B", "method": "10-shot", "result": f"{TR}/qwen25_icl_10ex_random.json", "kind": "icl"},

    # Table 5 ICL rows, CodeReviewer.
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "CodeQwen1.5-7B", "method": "3-shot", "result": f"{CR}/qwen15_icl_3ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "CodeQwen1.5-7B", "method": "5-shot", "result": f"{CR}/qwen15_icl_5ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "CodeQwen1.5-7B", "method": "10-shot", "result": f"{CR}/qwen15_icl_10ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "3-shot", "result": f"{CR}/qwen25-0.5b_icl_3ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "5-shot", "result": f"{CR}/qwen25-0.5b_icl_5ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "Qwen2.5-0.5B", "method": "10-shot", "result": f"{CR}/qwen25-0.5b_icl_10ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "Qwen2.5-7B", "method": "3-shot", "result": f"{CR}/qwen25_icl_3ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "Qwen2.5-7B", "method": "5-shot", "result": f"{CR}/qwen25_icl_5ex_random.json", "kind": "icl"},
    {"tables": "5", "source": "icl", "dataset": "code_reviewer", "model": "Qwen2.5-7B", "method": "10-shot", "result": f"{CR}/qwen25_icl_10ex_random.json", "kind": "icl"},
]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def code_or_diff(sample: Mapping, dataset_info: Mapping) -> str:
    value = sample[dataset_info["code_column"]]
    if isinstance(value, list):
        return " ".join(str(part) for part in value)
    return str(value)


def iter_result_rows(job: Mapping) -> Iterable[tuple[int, str, str]]:
    with Path(job["result"]).open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if job["kind"] == "complete":
        for index, row in enumerate(result["raw_data"]):
            yield index, row["prediction"], row["reference"]
        return

    predictions = result["predictions"]
    references = result["references"]
    for index, prediction in enumerate(predictions):
        reference = references[index] if index < len(references) else ""
        yield index, prediction, reference


def filter_jobs(job_group: str, limit_jobs: int | None) -> list[dict]:
    jobs = [dict(job) for job in JOBS]
    if job_group == "trans-review-finetuned":
        jobs = [
            job for job in jobs
            if job["dataset"] == "trans_review" and job["source"] in {"peft", "fmft"}
        ]
    elif job_group == "code-reviewer-finetuned":
        jobs = [
            job for job in jobs
            if job["dataset"] == "code_reviewer" and job["source"] in {"peft", "fmft"}
        ]
    elif job_group == "trans-review-icl":
        jobs = [
            job for job in jobs
            if job["dataset"] == "trans_review" and job["source"] == "icl"
        ]
    elif job_group == "code-reviewer-icl":
        jobs = [
            job for job in jobs
            if job["dataset"] == "code_reviewer" and job["source"] == "icl"
        ]
    elif job_group != "all":
        raise ValueError(f"unknown job group: {job_group}")

    if limit_jobs is not None:
        jobs = jobs[:limit_jobs]
    return jobs


def output_stem(job_group: str) -> str:
    return "" if job_group == "all" else f"_{job_group}"


def job_label(job: Mapping) -> str:
    return (
        f"tables={job['tables']} dataset={job['dataset']} "
        f"model={job['model']} method={job['method']} source={job['source']}"
    )


def verify_sources(jobs: Iterable[Mapping]) -> None:
    LOGGER.info("Verifying source data and result artifacts")
    missing = []
    for dataset_info in DATASETS.values():
        if not dataset_info["test"].exists():
            missing.append(str(dataset_info["test"]))
    for job in jobs:
        if not Path(job["result"]).exists():
            missing.append(job["result"])
    if missing:
        raise SystemExit("Missing required source paths:\n" + "\n".join(missing))
    LOGGER.info("Source verification complete")


def build_inputs(
    out_root: Path,
    jobs: list[dict],
    limit_samples: int | None,
    job_group: str,
) -> list[dict]:
    LOGGER.info("Building judge input JSONL files")
    input_dir = out_root / "inputs"
    scored_dir = out_root / "scored"
    input_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = {name: load_jsonl(info["test"]) for name, info in DATASETS.items()}
    manifest_rows = []

    for job in tqdm(jobs, desc="Building judge inputs", unit="job"):
        LOGGER.info("Building input for %s", job_label(job))
        dataset_info = DATASETS[job["dataset"]]
        raw_samples = raw_cache[job["dataset"]]
        job_id = (
            f"tables_{job['tables'].replace(',', '_')}_{job['dataset']}_"
            f"{slug(job['model'])}_{slug(job['method'])}_{job['source']}"
        )
        input_path = input_dir / f"{job_id}.jsonl"
        output_path = scored_dir / f"{job_id}.jsonl"
        rows = []

        for index, prediction, reference in iter_result_rows(job):
            if index >= len(raw_samples):
                break
            if limit_samples is not None and len(rows) >= limit_samples:
                break
            rows.append({
                "tables": job["tables"],
                "dataset": dataset_info["label"],
                "model": job["model"],
                "method": job["method"],
                "source": job["source"],
                "source_result": job["result"],
                "sample_index": index,
                "code_or_diff": code_or_diff(raw_samples[index], dataset_info),
                "prediction": prediction,
                "reference": reference,
            })

        write_jsonl(input_path, rows)
        LOGGER.info("Wrote %s rows to %s", len(rows), input_path)
        manifest_rows.append({
            "job_id": job_id,
            "tables": job["tables"],
            "dataset": dataset_info["label"],
            "model": job["model"],
            "method": job["method"],
            "source": job["source"],
            "source_result": job["result"],
            "input_path": str(input_path),
            "output_path": str(output_path),
            "rows": len(rows),
        })

    manifest_path = out_root / f"manifest{output_stem(job_group)}.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    LOGGER.info("Wrote manifest to %s", manifest_path)
    return manifest_rows


def failed_judge_result(config: JudgeConfig) -> dict:
    return {
        "qwen_judge_score": math.nan,
        "qwen_judge_mean": math.nan,
        "qwen_judge_median": math.nan,
        "qwen_judge_std": math.nan,
        "qwen_judge_agreement": math.nan,
        "qwen_judge_norm": math.nan,
        "qwen_judge_k": config.k,
        "qwen_judge_valid_runs": 0,
        "qwen_judge_parse_errors": config.k,
        "qwen_judge_model": config.model_path,
        "qwen_judge_prompt_version": config.prompt_version,
    }


def score_manifest(manifest_rows: Iterable[Mapping], config: JudgeConfig) -> None:
    LOGGER.info("Starting LLM-as-Judge scoring")
    judge = MockJudge(config) if config.mock else QwenJudge(config)
    for job in manifest_rows:
        input_path = Path(job["input_path"])
        output_path = Path(job["output_path"])
        rows = load_jsonl(input_path)
        scored_rows = []
        desc = f"{job['dataset']} {job['model']} {job['method']}"
        LOGGER.info("Scoring %s (%s rows)", job_label(job), len(rows))

        with tqdm(total=len(rows), desc=desc, unit="sample") as progress:
            for start in range(0, len(rows), config.batch_size):
                batch = rows[start:start + config.batch_size]
                try:
                    judge_results = judge.score_records(batch)
                    if len(judge_results) != len(batch):
                        raise RuntimeError(
                            f"judge returned {len(judge_results)} results for {len(batch)} rows"
                        )
                except Exception as exc:
                    LOGGER.warning(
                        "Batch judge failed for %s samples=%s-%s: %s",
                        job["job_id"],
                        batch[0].get("sample_index") if batch else None,
                        batch[-1].get("sample_index") if batch else None,
                        exc,
                    )
                    judge_results = []
                    for row in batch:
                        try:
                            judge_results.append(judge.score_record(row))
                        except Exception as row_exc:
                            LOGGER.warning(
                                "Judge failed for %s sample=%s: %s",
                                job["job_id"],
                                row.get("sample_index"),
                                row_exc,
                            )
                            judge_results.append(failed_judge_result(config))

                for row, judge_result in zip(batch, judge_results):
                    updated = dict(row)
                    updated.update(judge_result)
                    scored_rows.append(updated)
                progress.update(len(batch))

        write_jsonl(output_path, scored_rows)
        LOGGER.info("Scored %s rows: %s", len(scored_rows), output_path)


def mean_finite(rows: Iterable[Mapping], column: str) -> float:
    values = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(value)
    return statistics.fmean(values) if values else math.nan


def write_summary(out_root: Path, manifest_rows: Iterable[Mapping], job_group: str) -> Path:
    LOGGER.info("Writing summary CSV")
    summary_path = out_root / f"summary{output_stem(job_group)}.csv"
    summary_rows = []

    for job in tqdm(list(manifest_rows), desc="Summarizing judged outputs", unit="job"):
        scored_path = Path(job["output_path"])
        if scored_path.exists():
            rows = load_jsonl(scored_path)
        else:
            LOGGER.warning("Missing judged output for %s: %s", job_label(job), scored_path)
            rows = []
        valid_rows = sum(1 for row in rows if row.get("qwen_judge_valid_runs", 0) > 0)
        summary_rows.append({
            "tables": job["tables"],
            "dataset": job["dataset"],
            "model": job["model"],
            "method": job["method"],
            "source": job["source"],
            "rows": len(rows),
            "valid_rows": valid_rows,
            "qwen_judge_score": mean_finite(rows, "qwen_judge_score"),
            "qwen_judge_mean": mean_finite(rows, "qwen_judge_mean"),
            "qwen_judge_std": mean_finite(rows, "qwen_judge_std"),
            "qwen_judge_agreement": mean_finite(rows, "qwen_judge_agreement"),
            "source_result": job["source_result"],
            "judged_output": job["output_path"],
        })

    fieldnames = [
        "tables",
        "dataset",
        "model",
        "method",
        "source",
        "rows",
        "valid_rows",
        "qwen_judge_score",
        "qwen_judge_mean",
        "qwen_judge_std",
        "qwen_judge_agreement",
        "source_result",
        "judged_output",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    LOGGER.info("Wrote summary CSV to %s", summary_path)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update proposal Tables 1-5 with LLM-as-Judge scores."
    )
    parser.add_argument("--judge-model-path", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-k", type=int, default=5)
    parser.add_argument("--judge-cache-path", default=None)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--job-group", default="all", choices=JOB_GROUPS,
                        help="Subset of table jobs to run")
    parser.add_argument("--summary-only", action="store_true",
                        help="Build manifest and summary from existing scored files without judging")
    parser.add_argument("--judge-local-files-only", action="store_true")
    parser.add_argument("--judge-device", default="cuda",
                        help="Judge device: cuda, cuda:N, cpu, or auto (default: cuda)")
    parser.add_argument("--judge-batch-size", type=int, default=1,
                        help="Number of judge records to generate per batch")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock judge scores")
    parser.add_argument("--limit-jobs", type=int, default=None, help="Smoke-test only: score first N jobs")
    parser.add_argument("--limit-samples", type=int, default=None, help="Smoke-test only: score first N samples per job")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def preflight_device(args: argparse.Namespace) -> None:
    if args.mock or not args.judge_device.startswith("cuda"):
        if args.mock:
            LOGGER.info("Mock mode enabled; skipping CUDA preflight")
        return

    LOGGER.info("Checking CUDA device availability for %s", args.judge_device)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; LLM-as-Judge is configured to run on GPU.")

    target = "cuda:0" if args.judge_device == "cuda" else args.judge_device
    if ":" in target:
        device_index = int(target.split(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise SystemExit(
                f"Requested {target}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    out_root = Path(args.out_root)
    jobs = filter_jobs(args.job_group, args.limit_jobs)
    LOGGER.info("Preparing %s table job(s)", len(jobs))
    LOGGER.info("Job group: %s", args.job_group)
    LOGGER.info("Output root: %s", out_root)
    LOGGER.info("Judge model: %s", args.judge_model_path)
    LOGGER.info("Judge k: %s", args.judge_k)
    LOGGER.info("Judge batch size: %s", args.judge_batch_size)

    if not args.summary_only and not args.mock and not Path(args.judge_model_path).is_dir():
        raise SystemExit(
            f"Missing local judge model: {args.judge_model_path}\n"
            "Place the Hugging Face-compatible model at this path or run with --mock for smoke tests."
        )

    verify_sources(jobs)
    manifest_rows = build_inputs(out_root, jobs, args.limit_samples, args.job_group)
    LOGGER.info("Prepared %s job(s)", len(manifest_rows))

    if args.summary_only:
        LOGGER.info("Summary-only mode enabled; skipping judge scoring")
        summary_path = write_summary(out_root, manifest_rows, args.job_group)
        LOGGER.info("Done. Summary: %s", summary_path)
        return

    preflight_device(args)
    cache_path = args.judge_cache_path or str(out_root / "qwen_judge_cache.jsonl")
    LOGGER.info("Judge cache: %s", cache_path)
    config = JudgeConfig(
        model_path=args.judge_model_path,
        k=args.judge_k,
        cache_path=cache_path,
        local_files_only=args.judge_local_files_only,
        device=args.judge_device,
        batch_size=args.judge_batch_size,
        code_column="code_or_diff",
        prediction_column="prediction",
        reference_column="reference",
        mock=args.mock,
    )

    score_manifest(manifest_rows, config)
    summary_path = write_summary(out_root, manifest_rows, args.job_group)
    LOGGER.info("Done. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
