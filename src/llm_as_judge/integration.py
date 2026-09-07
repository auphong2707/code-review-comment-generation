"""Single copy of the LLM-as-Judge glue shared by re_eval_model, icl_inference
and tools/update_llm_judge_tables.py (previously duplicated in each).

The ``--judge-*-column`` overrides are deliberately NOT forwarded into
``JudgeConfig`` here: this glue resolves columns against the *raw dataset
sample* first and hands the judge canonical ``code_or_diff``/``prediction``/
``reference`` records, so setting them on the config would make
``normalize_record`` reject those records.
"""

import json
import os

from tqdm import tqdm

from .config import JudgeConfig
from .qwen_judge import CODE_COLUMNS, MockJudge, QwenJudge
from .qwen_judge import make_judge as _make_judge
from .scorer import JUDGE_COLUMNS  # re-exported for consumers

JUDGE_CODE_COLUMNS = CODE_COLUMNS  # legacy alias


def add_judge_arguments(parser):
    parser.add_argument("--llm-judge", dest="llm_judge", action="store_true",
                        help="Enable optional LLM-as-Judge scoring")
    parser.add_argument("--judge-model-path", dest="judge_model_path",
                        default=JudgeConfig.model_path,
                        help="Path to local Hugging Face-compatible judge model")
    parser.add_argument("--judge-k", dest="judge_k", type=int, default=JudgeConfig.k,
                        help="Number of judge runs per sample")
    parser.add_argument("--judge-cache-path", dest="judge_cache_path", default=None,
                        help="Optional JSONL cache path for judge runs")
    parser.add_argument("--judge-output-path", dest="judge_output_path", default=None,
                        help="Optional output path for judged results")
    parser.add_argument("--judge-local-files-only", dest="judge_local_files_only",
                        action="store_true", help="Load judge model with local_files_only=True")
    parser.add_argument("--judge-debug", dest="judge_debug", action="store_true",
                        help="Print per-sample judge errors")
    parser.add_argument("--judge-code-column", dest="judge_code_column", default=None,
                        help="Optional raw dataset column to use as judge code/diff input")
    parser.add_argument("--judge-prediction-column", dest="judge_prediction_column", default=None,
                        help="Optional result/raw column to use as judge prediction")
    parser.add_argument("--judge-reference-column", dest="judge_reference_column", default=None,
                        help="Optional result/raw column to use as judge reference")
    parser.add_argument("--judge-device", dest="judge_device", default=JudgeConfig.device,
                        help="Judge device: cuda, cuda:N, cpu, or auto (default: cuda)")
    parser.add_argument("--judge-input-token-limit", type=int, default=None,
                        help="Fail before GPU generation above this qualified token budget; no truncation")
    parser.add_argument("--judge-batch-size", dest="judge_batch_size", type=int,
                        default=JudgeConfig.batch_size,
                        help="Number of judge records to generate per batch")
    parser.add_argument("--judge-mock", dest="judge_mock", action="store_true",
                        help="Use deterministic mock judge scores (no model load)")
    parser.add_argument("--judge-repair-retry", dest="judge_repair_retry", action="store_true",
                        help="Retry parse failures once with a deterministic repair prompt")
    parser.add_argument("--judge-emit-debug", dest="judge_emit_debug", action="store_true",
                        help="Emit qwen_judge_raw_outputs/qwen_judge_component_scores columns")
    parser.add_argument("--judge-backend", dest="judge_backend", default=JudgeConfig.backend,
                        choices=["qwen", "hf-chat", "mock"],
                        help="Judge backend (default qwen; hf-chat for non-Qwen chat models)")
    parser.add_argument("--judge-column-prefix", dest="judge_column_prefix",
                        default=JudgeConfig.column_prefix,
                        help="Output column prefix (e.g. llama_judge_ for the second judge)")
    parser.add_argument("--judge-top-k", dest="judge_top_k", type=int, default=None,
                        help="hf-chat backend: top_k sampling parameter (default: unset)")
    parser.add_argument("--judge-repetition-penalty", dest="judge_repetition_penalty",
                        type=float, default=None,
                        help="hf-chat backend: repetition penalty (default: unset)")
    parser.add_argument("--judge-seed", dest="judge_seed", type=int, default=None,
                        help="hf-chat backend: torch.manual_seed(seed + run_index) per batch")


def build_judge_config(args):
    return JudgeConfig(
        model_path=args.judge_model_path,
        k=args.judge_k,
        cache_path=args.judge_cache_path,
        local_files_only=args.judge_local_files_only,
        device=args.judge_device,
        batch_size=args.judge_batch_size,
        input_token_limit=getattr(args, "judge_input_token_limit", None),
        mock=args.judge_mock,
        repair_retry=args.judge_repair_retry,
        emit_debug_columns=args.judge_emit_debug,
        backend=getattr(args, "judge_backend", JudgeConfig.backend),
        column_prefix=getattr(args, "judge_column_prefix", JudgeConfig.column_prefix),
        top_k=getattr(args, "judge_top_k", None),
        repetition_penalty=getattr(args, "judge_repetition_penalty", None),
        seed=getattr(args, "judge_seed", None),
    )


def load_raw_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def value_as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def get_sample_code_for_judge(sample, dataset_name, explicit_column=None):
    if explicit_column:
        if explicit_column not in sample:
            raise ValueError(
                f"missing judge code column '{explicit_column}'; available columns: {sorted(sample.keys())}"
            )
        return value_as_text(sample[explicit_column])

    if dataset_name == "trans_review" and "code_tokens" in sample:
        return value_as_text(sample["code_tokens"])
    if dataset_name == "acra_by_ls_pt" and "patch" in sample:
        return value_as_text(sample["patch"])

    for column in JUDGE_CODE_COLUMNS:
        if column in sample:
            return value_as_text(sample[column])
    raise ValueError(f"could not detect judge code/diff column; available columns: {sorted(sample.keys())}")


def get_judge_text(row, sample, explicit_column, default_column, label):
    if explicit_column:
        if explicit_column in row:
            return value_as_text(row[explicit_column])
        if explicit_column in sample:
            return value_as_text(sample[explicit_column])
        raise ValueError(
            f"missing judge {label} column '{explicit_column}'; "
            f"available result columns: {sorted(row.keys())}; available raw columns: {sorted(sample.keys())}"
        )
    return value_as_text(row[default_column])


def failed_judge_result(judge_config):
    prefix = judge_config.column_prefix
    result = {
        f"{prefix}score": float("nan"),
        f"{prefix}mean": float("nan"),
        f"{prefix}median": float("nan"),
        f"{prefix}std": float("nan"),
        f"{prefix}agreement": float("nan"),
        f"{prefix}norm": float("nan"),
        f"{prefix}k": judge_config.k,
        f"{prefix}valid_runs": 0,
        f"{prefix}parse_errors": judge_config.k,
        f"{prefix}model": judge_config.model_path,
        f"{prefix}prompt_version": judge_config.prompt_version,
    }
    if judge_config.emit_debug_columns:
        result[f"{prefix}raw_outputs"] = []
        result[f"{prefix}component_scores"] = []
    return result


def save_json_result(path, results):
    results_dir = os.path.dirname(path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def make_judge(judge_config):
    return _make_judge(judge_config)


def judge_rows(rows, raw_samples, dataset_name, judge_config, args, judge=None):
    """Score result rows in place. Each row needs prediction/reference (or the
    --judge-*-column overrides); raw_samples align positionally with rows."""
    judge = judge or make_judge(judge_config)

    if len(raw_samples) != len(rows):
        print(
            f"WARNING: LLM-as-Judge alignment mismatch: {len(rows)} result rows vs "
            f"{len(raw_samples)} raw samples; extra rows are judged with empty samples."
        )

    pending = []
    for index, row in enumerate(rows):
        sample = raw_samples[index] if index < len(raw_samples) else {}
        try:
            record = {
                "code_or_diff": get_sample_code_for_judge(sample, dataset_name, args.judge_code_column),
                "prediction": get_judge_text(row, sample, args.judge_prediction_column, "prediction", "prediction"),
                "reference": get_judge_text(row, sample, args.judge_reference_column, "reference", "reference"),
            }
            pending.append((index, record))
        except Exception as exc:
            if args.judge_debug:
                print(f"WARNING: LLM-as-Judge failed for sample {index}: {exc}")
            row.update(failed_judge_result(judge_config))

    batch_size = max(1, judge_config.batch_size)
    with tqdm(total=len(pending), desc="LLM-as-Judge scoring", unit="sample") as progress:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            chunk_records = [record for _, record in chunk]
            try:
                judge_results = judge.score_records(chunk_records)
                if len(judge_results) != len(chunk_records):
                    raise RuntimeError(
                        f"judge returned {len(judge_results)} results for {len(chunk_records)} rows"
                    )
            except Exception as exc:
                # Fall back to per-row scoring so one bad batch cannot fail
                # every sample in it (legacy per-row behavior).
                if args.judge_debug:
                    print(f"WARNING: LLM-as-Judge batch failed, retrying per row: {exc}")
                judge_results = []
                for _, record in chunk:
                    try:
                        judge_results.append(judge.score_record(record))
                    except Exception as row_exc:
                        if args.judge_debug:
                            print(f"WARNING: LLM-as-Judge failed for a sample: {row_exc}")
                        judge_results.append(failed_judge_result(judge_config))

            for (index, _), judge_result in zip(chunk, judge_results):
                rows[index].update(judge_result)
            progress.update(len(chunk))

    return rows


def append_judge_results(raw_data, test_dataset_directory, dataset_name, judge_config, args):
    """re_eval_model path: score complete_results raw_data rows in place."""
    print("LLM-as-Judge enabled. Scoring generated predictions...")
    raw_samples = load_raw_jsonl(test_dataset_directory)
    judge_rows(raw_data, raw_samples, dataset_name, judge_config, args)
    return raw_data


def append_judge_results_to_icl(results, test_data_path, max_samples, dataset_name, judge_config, args):
    """icl_inference path: build raw_data rows from predictions/references."""
    print("LLM-as-Judge enabled. Scoring generated ICL predictions...")
    raw_samples = load_raw_jsonl(test_data_path)
    if max_samples:
        raw_samples = raw_samples[:max_samples]

    predictions = results.get("predictions", [])
    references = results.get("references", [])
    rows = [
        {
            "prediction": prediction,
            "reference": references[index] if index < len(references) else "",
        }
        for index, prediction in enumerate(predictions)
    ]
    judge_rows(rows, raw_samples, dataset_name, judge_config, args)

    results["raw_data"] = rows
    return results
