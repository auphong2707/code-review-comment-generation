"""Standalone LLM-as-Judge utilities for code review comments."""

from .config import JudgeConfig
from .parser import parse_judge_output
from .scorer import aggregate_scores, score_components


def score_record(record, config):
    """Score one record with lazy judge imports."""
    from .qwen_judge import score_record as _score_record

    return _score_record(record, config)


def score_records(records, config):
    """Score records with lazy judge imports."""
    from .qwen_judge import score_records as _score_records

    return _score_records(records, config)


__all__ = [
    "JudgeConfig",
    "aggregate_scores",
    "parse_judge_output",
    "score_components",
    "score_record",
    "score_records",
]
