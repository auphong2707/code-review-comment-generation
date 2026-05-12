import argparse
import csv
import json
import os
from typing import List

from .config import JudgeConfig
from .qwen_judge import score_records


def _load_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(path: str) -> tuple[str, List[dict]]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return "csv", _load_csv(path)
    if suffix == ".jsonl":
        return "jsonl", _load_jsonl(path)
    raise ValueError(f"unsupported input format '{suffix}'; expected .csv or .jsonl")


def write_rows(path: str, kind: str, rows: List[dict]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if kind == "csv":
        _write_csv(path, rows)
        return
    if kind == "jsonl":
        _write_jsonl(path, rows)
        return
    raise ValueError(f"unsupported output format kind: {kind}")


def score_file(input_path: str, output_path: str, config: JudgeConfig) -> List[dict]:
    kind, rows = load_rows(input_path)
    judge_results = score_records(rows, config)
    scored_rows = []
    for row, judge_result in zip(rows, judge_results):
        updated = dict(row)
        updated.update(judge_result)
        scored_rows.append(updated)
    write_rows(output_path, kind, scored_rows)
    return scored_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score CSV/JSONL review-comment results with LLM-as-Judge")
    parser.add_argument("input_path", help="Input .csv or .jsonl file")
    parser.add_argument("output_path", help="Output .csv or .jsonl file")
    parser.add_argument("--judge-model-path", default=JudgeConfig.model_path)
    parser.add_argument("--judge-k", type=int, default=JudgeConfig.k)
    parser.add_argument("--judge-cache-path", default=None)
    parser.add_argument("--judge-local-files-only", action="store_true")
    parser.add_argument("--judge-temperature", type=float, default=JudgeConfig.temperature)
    parser.add_argument("--judge-top-p", type=float, default=JudgeConfig.top_p)
    parser.add_argument("--judge-max-new-tokens", type=int, default=JudgeConfig.max_new_tokens)
    parser.add_argument("--judge-penalty-lambda", type=float, default=JudgeConfig.penalty_lambda)
    parser.add_argument("--judge-no-sample", action="store_true")
    parser.add_argument("--code-column", default=None)
    parser.add_argument("--prediction-column", default=None)
    parser.add_argument("--reference-column", default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock scores for smoke tests")
    return parser


def config_from_args(args) -> JudgeConfig:
    return JudgeConfig(
        model_path=args.judge_model_path,
        k=args.judge_k,
        penalty_lambda=args.judge_penalty_lambda,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_new_tokens=args.judge_max_new_tokens,
        do_sample=not args.judge_no_sample,
        cache_path=args.judge_cache_path,
        local_files_only=args.judge_local_files_only,
        code_column=args.code_column,
        prediction_column=args.prediction_column,
        reference_column=args.reference_column,
        mock=args.mock,
    )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    score_file(args.input_path, args.output_path, config_from_args(args))
    print(f"Judged results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
