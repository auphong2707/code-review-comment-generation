"""Single shared implementation of the generation metrics used across the repo.

Semantics are frozen for comparability with produced results (AGENTS.md hard
invariants): BLEU-4 wraps each reference as ``[[ref]]``, ROUGE uses
``use_stemmer=True``, BERTScore is the corpus mean of per-sample
precision/recall/f1 with ``lang="en"``, empty generations are replaced by
``"<EMPTY>"`` before scoring.
"""

EMPTY_PLACEHOLDER = "<EMPTY>"

_METRIC_NAMES = ("bleu", "exact_match", "rouge", "bertscore", "meteor")
_metrics = {}
_metric_resource_kwargs = {}


def configure_metric_resources(*, device=None, batch_size=None):
    """Explicit process-local resource policy; default metric calls stay unchanged."""
    if batch_size is not None and batch_size < 1:
        raise ValueError("Metric batch size must be positive")
    _metric_resource_kwargs.clear()
    if device is not None:
        _metric_resource_kwargs["device"] = device
    if batch_size is not None:
        _metric_resource_kwargs["batch_size"] = batch_size



def _metric(name):
    if name not in _metrics:
        import evaluate

        _metrics[name] = evaluate.load(name)
    return _metrics[name]


def preload_metrics():
    """Load all metric backends up front (fail fast, e.g. in trainer __init__)."""
    for name in _METRIC_NAMES:
        _metric(name)


def guard_empty(text):
    """Placeholder for empty generations so BLEU does not crash."""
    return text if text and text.strip() else EMPTY_PLACEHOLDER


def compute_generation_metrics(predictions, references):
    """Compute the five frozen metrics; returns neutral keys (no prefix)."""
    bleu_score = _metric("bleu").compute(
        predictions=predictions,
        references=[[ref] for ref in references],
    )["bleu"]

    exact_match_score = _metric("exact_match").compute(
        predictions=predictions,
        references=references,
    )["exact_match"]

    rouge_score = _metric("rouge").compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    bertscore = _metric("bertscore").compute(
        predictions=predictions,
        references=references,
        lang="en",
        **_metric_resource_kwargs,
    )
    bertscore = {
        "precision": sum(bertscore["precision"]) / len(bertscore["precision"]),
        "recall": sum(bertscore["recall"]) / len(bertscore["recall"]),
        "f1": sum(bertscore["f1"]) / len(bertscore["f1"]),
    }

    meteor_score = _metric("meteor").compute(
        predictions=predictions,
        references=references,
    )["meteor"]

    return {
        "bleu_4": bleu_score,
        "exact_match": exact_match_score,
        "rouge": rouge_score,
        "bertscore": bertscore,
        "meteor": meteor_score,
    }


def as_test_results(metrics):
    """Legacy schema of complete_results.json / re_evaluate_test_results.json."""
    return {f"test_{key}": value for key, value in metrics.items()}


def as_eval_results(metrics):
    """Legacy in-training schema (metric_for_best_model="eval_bleu_4")."""
    return {f"eval_{key}": value for key, value in metrics.items()}


def as_icl_results(metrics):
    """Legacy schema of the ICL result JSON (note: bertscore has no suffix)."""
    return {
        "bleu_score": metrics["bleu_4"],
        "exact_match_score": metrics["exact_match"],
        "rouge_score": metrics["rouge"],
        "bertscore": metrics["bertscore"],
        "meteor_score": metrics["meteor"],
    }
