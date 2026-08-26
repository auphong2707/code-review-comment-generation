import os

import pytest

from src.evaluation import metrics as metrics_mod
from src.evaluation.metrics import (
    EMPTY_PLACEHOLDER,
    as_eval_results,
    as_icl_results,
    as_test_results,
    compute_generation_metrics,
    guard_empty,
)


class _StubMetric:
    """Records compute() kwargs and returns a canned result."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def stub_metrics(monkeypatch):
    stubs = {
        "bleu": _StubMetric({"bleu": 0.25}),
        "exact_match": _StubMetric({"exact_match": 0.5}),
        "rouge": _StubMetric({"rouge1": 0.4, "rougeL": 0.3}),
        "bertscore": _StubMetric(
            {"precision": [0.8, 0.6], "recall": [0.7, 0.5], "f1": [0.75, 0.55]}
        ),
        "meteor": _StubMetric({"meteor": 0.35}),
    }
    monkeypatch.setattr(metrics_mod, "_metric", lambda name: stubs[name])
    return stubs


def test_compute_generation_metrics_structure_and_call_semantics(stub_metrics):
    preds = ["fix the null check", "add docstring"]
    refs = ["handle null here", "document this"]

    result = compute_generation_metrics(preds, refs)

    assert result == {
        "bleu_4": 0.25,
        "exact_match": 0.5,
        "rouge": {"rouge1": 0.4, "rougeL": 0.3},
        "bertscore": {"precision": 0.7, "recall": 0.6, "f1": 0.65},
        "meteor": 0.35,
    }
    # Frozen call semantics
    assert stub_metrics["bleu"].calls[0]["references"] == [[r] for r in refs]
    assert stub_metrics["exact_match"].calls[0]["references"] == refs
    assert stub_metrics["rouge"].calls[0]["use_stemmer"] is True
    assert stub_metrics["bertscore"].calls[0]["lang"] == "en"


def test_key_mappers_preserve_legacy_schemas(stub_metrics):
    m = compute_generation_metrics(["a"], ["b"])

    assert set(as_test_results(m)) == {
        "test_bleu_4", "test_exact_match", "test_rouge", "test_bertscore", "test_meteor",
    }
    assert set(as_eval_results(m)) == {
        "eval_bleu_4", "eval_exact_match", "eval_rouge", "eval_bertscore", "eval_meteor",
    }
    icl = as_icl_results(m)
    assert set(icl) == {
        "bleu_score", "exact_match_score", "rouge_score", "bertscore", "meteor_score",
    }
    assert as_test_results(m)["test_bleu_4"] == m["bleu_4"]
    assert icl["bertscore"] == m["bertscore"]


def test_guard_empty():
    assert guard_empty("") == EMPTY_PLACEHOLDER
    assert guard_empty("   ") == EMPTY_PLACEHOLDER
    assert guard_empty(None) == EMPTY_PLACEHOLDER
    assert guard_empty("ok") == "ok"


@pytest.mark.skipif(
    not os.environ.get("RUN_METRIC_PARITY"),
    reason="real-metric parity needs network + metric downloads; set RUN_METRIC_PARITY=1",
)
def test_real_metric_parity_against_legacy_inline_block():
    """Reference implementation = the legacy inline block, copied verbatim."""
    import evaluate

    preds = ["please add a null check here", EMPTY_PLACEHOLDER]
    refs = ["add a null check", "rename this variable"]

    # --- legacy inline block (do not modernize) ---
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    rouge_metric = evaluate.load("rouge")
    bert_score_metric = evaluate.load("bertscore")
    meteor_metric = evaluate.load("meteor")
    bleu_score = bleu_metric.compute(
        predictions=preds, references=[[ref] for ref in refs]
    )["bleu"]
    exact_match_score = exact_match_metric.compute(
        predictions=preds, references=refs
    )["exact_match"]
    rouge_score = rouge_metric.compute(
        predictions=preds, references=refs, use_stemmer=True
    )
    bertscore = bert_score_metric.compute(predictions=preds, references=refs, lang="en")
    bertscore = {
        "precision": sum(bertscore["precision"]) / len(bertscore["precision"]),
        "recall": sum(bertscore["recall"]) / len(bertscore["recall"]),
        "f1": sum(bertscore["f1"]) / len(bertscore["f1"]),
    }
    meteor_score = meteor_metric.compute(predictions=preds, references=refs)["meteor"]
    # --- end legacy block ---

    new = compute_generation_metrics(preds, refs)
    assert new["bleu_4"] == pytest.approx(bleu_score)
    assert new["exact_match"] == pytest.approx(exact_match_score)
    assert new["rouge"] == pytest.approx(rouge_score)
    assert new["bertscore"]["f1"] == pytest.approx(bertscore["f1"])
    assert new["meteor"] == pytest.approx(meteor_score)
