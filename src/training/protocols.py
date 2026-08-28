"""Per-dataset training-protocol overrides (D-025, KI-009).

The produced TransReview artifacts were trained with a different epochs/batch
protocol than the HEAD specs (which encode the CodeReviewer protocol). New TR
runs must replicate the archived protocol, so `--protocol archived` (the
default) merges these overrides into the TrainingArguments kwargs; CodeReviewer
runs are never touched. Torch-free by design (used by `--dry-run`).
"""

# (dataset, model_key) -> TrainingArguments overrides measured from the
# archived training_args.txt (verified by the phase-03 harvest test)
ARCHIVED = {
    ("trans_review", "codet5"):      {"num_train_epochs": 15, "per_device_train_batch_size": 16, "per_device_eval_batch_size": 16},
    ("trans_review", "plbart"):      {"num_train_epochs": 15, "per_device_train_batch_size": 16, "per_device_eval_batch_size": 16},
    ("trans_review", "qwen25-0.5b"): {"num_train_epochs": 15, "per_device_train_batch_size": 8,  "per_device_eval_batch_size": 8},
    ("trans_review", "qwen15"):      {"num_train_epochs": 8,  "per_device_train_batch_size": 2,  "per_device_eval_batch_size": 2},
    ("trans_review", "qwen25"):      {"num_train_epochs": 8,  "per_device_train_batch_size": 2,  "per_device_eval_batch_size": 2},
}

# Derived models inherit their parent's row (phase 07 adds an explicit entry
# only if the 14B protocol is measured differently)
PARENT_MODEL = {
    "qwen25-14b": "qwen25",
}


def overrides_for(dataset_name, model_key, protocol="archived"):
    """TrainingArguments overrides for this (dataset, model, protocol)."""
    if protocol == "head":
        return {}
    if protocol != "archived":
        raise ValueError(f"unknown protocol '{protocol}'; expected 'archived' or 'head'")
    model_key = PARENT_MODEL.get(model_key, model_key)
    return dict(ARCHIVED.get((dataset_name, model_key), {}))
