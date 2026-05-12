import math
import statistics
from collections import Counter
from typing import Iterable, Mapping


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


def score_components(scores: Mapping[str, int]) -> float:
    score = (
        0.35 * scores["c"]
        + 0.25 * scores["g"]
        + 0.20 * scores["a"]
        + 0.10 * scores["s"]
        + 0.10 * scores["q"]
    )
    if scores["c"] <= 2 or scores["g"] <= 2:
        score = min(score, 2.5)
    if scores["a"] <= 2:
        score = min(score, 3.5)
    if scores["q"] <= 2:
        score = min(score, 4.0)
    return score


def _round_half_up(score: float) -> int:
    return int(math.floor(score + 0.5))


def aggregate_scores(
    run_scores: Iterable[float],
    *,
    k: int,
    parse_errors: int,
    model_path: str,
    prompt_version: str,
    penalty_lambda: float,
) -> dict:
    valid_scores = list(run_scores)
    valid_runs = len(valid_scores)

    if valid_runs == 0:
        return {
            "qwen_judge_score": math.nan,
            "qwen_judge_mean": math.nan,
            "qwen_judge_median": math.nan,
            "qwen_judge_std": math.nan,
            "qwen_judge_agreement": math.nan,
            "qwen_judge_norm": math.nan,
            "qwen_judge_k": k,
            "qwen_judge_valid_runs": 0,
            "qwen_judge_parse_errors": parse_errors,
            "qwen_judge_model": model_path,
            "qwen_judge_prompt_version": prompt_version,
        }

    mean_score = statistics.fmean(valid_scores)
    median_score = statistics.median(valid_scores)
    std_score = statistics.pstdev(valid_scores)
    rounded_counts = Counter(_round_half_up(score) for score in valid_scores)
    agreement = rounded_counts.most_common(1)[0][1] / valid_runs
    final_score = mean_score - penalty_lambda * std_score

    return {
        "qwen_judge_score": final_score,
        "qwen_judge_mean": mean_score,
        "qwen_judge_median": median_score,
        "qwen_judge_std": std_score,
        "qwen_judge_agreement": agreement,
        "qwen_judge_norm": (final_score - 1) / 4,
        "qwen_judge_k": k,
        "qwen_judge_valid_runs": valid_runs,
        "qwen_judge_parse_errors": parse_errors,
        "qwen_judge_model": model_path,
        "qwen_judge_prompt_version": prompt_version,
    }
