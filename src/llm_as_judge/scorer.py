import math
import statistics
from collections import Counter
from typing import Iterable, Mapping

_COLUMN_SUFFIXES = (
    "score",
    "mean",
    "median",
    "std",
    "agreement",
    "norm",
    "k",
    "valid_runs",
    "parse_errors",
    "model",
    "prompt_version",
)

# Frozen rubric weights and hard caps (paper Tables 1-5; test-pinned). A cap
# is (fields, threshold, max_score): if any field is <= threshold, the score
# is capped.
DEFAULT_WEIGHTS = {"c": 0.35, "g": 0.25, "a": 0.20, "s": 0.10, "q": 0.10}
DEFAULT_CAPS = (
    (("c", "g"), 2, 2.5),
    (("a",), 2, 3.5),
    (("q",), 2, 4.0),
)


def judge_columns(prefix: str) -> tuple:
    return tuple(prefix + suffix for suffix in _COLUMN_SUFFIXES)


JUDGE_COLUMNS = judge_columns("qwen_judge_")


def score_components(scores: Mapping[str, int], weights=None, caps=None) -> float:
    weights = DEFAULT_WEIGHTS if weights is None else weights
    caps = DEFAULT_CAPS if caps is None else caps
    score = sum(weight * scores[field] for field, weight in weights.items())
    for fields, threshold, max_score in caps:
        if any(scores[field] <= threshold for field in fields):
            score = min(score, max_score)
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
    column_prefix: str = "qwen_judge_",
) -> dict:
    valid_scores = list(run_scores)
    valid_runs = len(valid_scores)
    p = column_prefix

    if valid_runs == 0:
        return {
            f"{p}score": math.nan,
            f"{p}mean": math.nan,
            f"{p}median": math.nan,
            f"{p}std": math.nan,
            f"{p}agreement": math.nan,
            f"{p}norm": math.nan,
            f"{p}k": k,
            f"{p}valid_runs": 0,
            f"{p}parse_errors": parse_errors,
            f"{p}model": model_path,
            f"{p}prompt_version": prompt_version,
        }

    mean_score = statistics.fmean(valid_scores)
    median_score = statistics.median(valid_scores)
    std_score = statistics.pstdev(valid_scores)
    rounded_counts = Counter(_round_half_up(score) for score in valid_scores)
    agreement = rounded_counts.most_common(1)[0][1] / valid_runs
    final_score = mean_score - penalty_lambda * std_score

    return {
        f"{p}score": final_score,
        f"{p}mean": mean_score,
        f"{p}median": median_score,
        f"{p}std": std_score,
        f"{p}agreement": agreement,
        f"{p}norm": (final_score - 1) / 4,
        f"{p}k": k,
        f"{p}valid_runs": valid_runs,
        f"{p}parse_errors": parse_errors,
        f"{p}model": model_path,
        f"{p}prompt_version": prompt_version,
    }
