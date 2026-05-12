#!/usr/bin/env python3
"""Build and score LLM-as-Judge inputs for proposal Tables 1-5."""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_as_judge import JudgeConfig
from src.llm_as_judge.qwen_judge import MockJudge, QwenJudge


DEFAULT_JUDGE_MODEL = "assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUT_ROOT = "assets/models/llm-as-judge/table-results"
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


def selected_jobs(limit_jobs: int | None) -> list[dict]:
    jobs = [dict(job) for job in JOBS]
    if limit_jobs is not None:
        jobs = jobs[:limit_jobs]
    return jobs


def verify_sources(jobs: Iterable[Mapping]) -> None:
    missing = []
    for dataset_info in DATASETS.values():
        if not dataset_info["test"].exists():
            missing.append(str(dataset_info["test"]))
    for job in jobs:
        if not Path(job["result"]).exists():
            missing.append(job["result"])
    if missing:
        raise SystemExit("Missing required source paths:\n" + "\n".join(missing))


def build_inputs(out_root: Path, jobs: list[dict], limit_samples: int | None) -> list[dict]:
    input_dir = out_root / "inputs"
    scored_dir = out_root / "scored"
    input_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    raw_cache = {name: load_jsonl(info["test"]) for name, info in DATASETS.items()}
    manifest_rows = []

    for job in jobs:
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

    write_jsonl(out_root / "manifest.jsonl", manifest_rows)
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
    judge = MockJudge(config) if config.mock else QwenJudge(config)
    for job in manifest_rows:
        input_path = Path(job["input_path"])
        output_path = Path(job["output_path"])
        rows = load_jsonl(input_path)
        scored_rows = []

        for row in rows:
            updated = dict(row)
            try:
                updated.update(judge.score_record(row))
            except Exception as exc:
                print(f"WARNING: judge failed for {job['job_id']} sample {row.get('sample_index')}: {exc}")
                updated.update(failed_judge_result(config))
            scored_rows.append(updated)

        write_jsonl(output_path, scored_rows)
        print(f"Scored {len(scored_rows)} rows: {output_path}")


def mean_finite(rows: Iterable[Mapping], column: str) -> float:
    values = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(value)
    return statistics.fmean(values) if values else math.nan


def write_summary(out_root: Path, manifest_rows: Iterable[Mapping]) -> Path:
    summary_path = out_root / "summary.csv"
    summary_rows = []

    for job in manifest_rows:
        scored_path = Path(job["output_path"])
        rows = load_jsonl(scored_path) if scored_path.exists() else []
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

    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update proposal Tables 1-5 with LLM-as-Judge scores."
    )
    parser.add_argument("--judge-model-path", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-k", type=int, default=5)
    parser.add_argument("--judge-cache-path", default=None)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--judge-local-files-only", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock judge scores")
    parser.add_argument("--limit-jobs", type=int, default=None, help="Smoke-test only: score first N jobs")
    parser.add_argument("--limit-samples", type=int, default=None, help="Smoke-test only: score first N samples per job")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    jobs = selected_jobs(args.limit_jobs)

    if not args.mock and not Path(args.judge_model_path).is_dir():
        raise SystemExit(
            f"Missing local judge model: {args.judge_model_path}\n"
            "Place the Hugging Face-compatible model at this path or run with --mock for smoke tests."
        )

    verify_sources(jobs)
    cache_path = args.judge_cache_path or str(out_root / "qwen_judge_cache.jsonl")
    config = JudgeConfig(
        model_path=args.judge_model_path,
        k=args.judge_k,
        cache_path=cache_path,
        local_files_only=args.judge_local_files_only,
        code_column="code_or_diff",
        prediction_column="prediction",
        reference_column="reference",
        mock=args.mock,
    )

    manifest_rows = build_inputs(out_root, jobs, args.limit_samples)
    print(f"Wrote {len(manifest_rows)} jobs: {out_root / 'manifest.jsonl'}")
    score_manifest(manifest_rows, config)
    summary_path = write_summary(out_root, manifest_rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
