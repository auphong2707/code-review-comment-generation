"""A100 revision runtime: deadline deferral, CodeReviewer ICL subset, judge row pairing."""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from src.training import recovery


@pytest.fixture
def no_stop(monkeypatch):
    monkeypatch.setattr(recovery, "_stop_requested", False)


def test_deferral_needs_a_deadline_and_writes_its_marker(tmp_path, monkeypatch, no_stop):
    marker = tmp_path / "defer" / "7"
    monkeypatch.setenv("RCG_DEFER_FILE", str(marker))
    monkeypatch.delenv("RCG_DEADLINE", raising=False)
    recovery.defer_if_short(10**6, "metrics")          # no deadline: never defers
    assert recovery.seconds_left() is None and not marker.exists()
    monkeypatch.setenv("RCG_DEADLINE", str(time.time() + 3600))
    recovery.defer_if_short(60, "metrics")             # fits
    assert not marker.exists()
    with pytest.raises(SystemExit) as stopped:
        recovery.defer_if_short(7200, "test corpus metrics")
    assert stopped.value.code == recovery.INCOMPLETE
    assert "test corpus metrics" in marker.read_text()
    assert recovery.metric_seconds(2000, "cpu") > recovery.metric_seconds(2000, "cuda") > 0


def test_epoch_evaluation_is_deferred_one_step_early(tmp_path, monkeypatch, no_stop):
    callbacks, saved = [], []
    trainer = SimpleNamespace(
        args=SimpleNamespace(save_only_model=False, world_size=1, output_dir=str(tmp_path)),
        add_callback=callbacks.append, model=None,
        _save_checkpoint=lambda *args, **kwargs: saved.append(True))
    recovery.attach_recovery(trainer, every_steps=1000)
    callback = callbacks[0]
    args = SimpleNamespace(max_steps=-1, eval_strategy="epoch", num_train_epochs=2)
    control = SimpleNamespace(should_evaluate=False)
    marker = tmp_path / "defer"
    monkeypatch.setenv("RCG_DEFER_FILE", str(marker))
    monkeypatch.setenv("RCG_DEADLINE", str(time.time() + 600))
    # 20 steps over 2 epochs: only step 9 -> 10 precedes an epoch evaluation.
    callback.on_step_end(args, SimpleNamespace(global_step=8, max_steps=20, epoch=0.8), control)
    assert not saved and not marker.exists()
    # A measured 100 s evaluation fits in 600 s ...
    (tmp_path / "eval_seconds.json").write_text(json.dumps({"max_seconds": 100}))
    callback.on_step_end(args, SimpleNamespace(global_step=9, max_steps=20, epoch=0.9), control)
    assert not saved and not marker.exists()
    # ... the default one-hour estimate does not: commit the step, then exit 75 deferred.
    (tmp_path / "eval_seconds.json").unlink()
    with pytest.raises(SystemExit) as stopped:
        callback.on_step_end(args, SimpleNamespace(global_step=9, max_steps=20, epoch=0.9), control)
    assert stopped.value.code == recovery.INCOMPLETE
    assert saved == [True] and "epoch evaluation" in marker.read_text()


def test_resumed_deferral_point_is_checked_before_the_first_step(tmp_path, monkeypatch, no_stop):
    callbacks, saved = [], []
    trainer = SimpleNamespace(
        args=SimpleNamespace(save_only_model=False, world_size=1, output_dir=str(tmp_path)),
        add_callback=callbacks.append, model=None,
        _save_checkpoint=lambda *args, **kwargs: saved.append(True))
    recovery.attach_recovery(trainer, every_steps=1000)
    callback = callbacks[0]
    args = SimpleNamespace(max_steps=-1, eval_strategy="epoch", num_train_epochs=2)
    monkeypatch.setenv("RCG_DEFER_FILE", str(tmp_path / "defer"))
    monkeypatch.setenv("RCG_DEADLINE", str(time.time() + 600))
    callback.on_train_begin(args, SimpleNamespace(global_step=8, max_steps=20), None)   # step 9: no eval
    with pytest.raises(SystemExit) as stopped:   # resumed at step 9: step 10 evaluates
        callback.on_train_begin(args, SimpleNamespace(global_step=9, max_steps=20), None)
    assert stopped.value.code == recovery.INCOMPLETE and not saved   # step 9 is already committed


def test_eval_duration_is_recorded(tmp_path, no_stop):
    callbacks = []
    trainer = SimpleNamespace(
        args=SimpleNamespace(save_only_model=False, world_size=1, output_dir=str(tmp_path)),
        add_callback=callbacks.append, model=None, _save_checkpoint=lambda *a, **k: None)
    recovery.attach_recovery(trainer, every_steps=1000)
    callback = callbacks[0]
    callback.on_epoch_end(None, None, None)
    callback.eval_started -= 50
    callback.on_evaluate(None, None, None)
    recorded = json.loads((tmp_path / "eval_seconds.json").read_text())
    assert 50 <= recorded["max_seconds"] < 60
    assert callback.eval_estimate() == pytest.approx(1.25 * recorded["max_seconds"])


def write_subset(path, dataset, indices):
    path.write_text(json.dumps({"dataset": dataset, "indices": indices}))
    return str(path)


def test_sample_indices_are_validated_and_sorted(tmp_path):
    from src.icl_inference import load_sample_indices
    path = write_subset(tmp_path / "subset.json", "code_reviewer", [7, 2, 5])
    assert load_sample_indices(path, "acra_by_ls_pt", 10) == [2, 5, 7]
    with pytest.raises(ValueError, match="not"):
        load_sample_indices(path, "trans_review", 10)
    with pytest.raises(ValueError, match="within"):
        load_sample_indices(path, "acra_by_ls_pt", 7)
    duplicate = write_subset(tmp_path / "dup.json", "code_reviewer", [1, 1])
    with pytest.raises(ValueError, match="unique"):
        load_sample_indices(duplicate, "acra_by_ls_pt", 10)


def test_judge_pairs_subset_predictions_with_original_rows(tmp_path):
    from tools.update_llm_judge_tables import iter_result_rows
    result = tmp_path / "icl.json"
    result.write_text(json.dumps({"predictions": ["a", "b"], "references": ["ra", "rb"],
                                  "sample_indices": [3, 9]}))
    job = {"kind": "icl", "result": str(result)}
    assert list(iter_result_rows(job)) == [(3, "a", "ra"), (9, "b", "rb")]
    result.write_text(json.dumps({"predictions": ["a", "b"], "references": ["ra", "rb"]}))
    assert list(iter_result_rows(job)) == [(0, "a", "ra"), (1, "b", "rb")]
    result.write_text(json.dumps({"predictions": ["a"], "references": ["ra"], "sample_indices": [3, 9]}))
    with pytest.raises(ValueError):
        list(iter_result_rows(job))


def test_gate_c_accepts_a_bound_subset_run(tmp_path, monkeypatch):
    from tools import campaign_status as status
    from src.training.recovery import fingerprint
    data = tmp_path / "data.jsonl"
    data.write_text('{"fixture": 1}\n{"fixture": 2}\n{"fixture": 3}\n')
    monkeypatch.setitem(status.DATASET_CONFIGS, "acra_by_ls_pt",
                        {"data_files": dict(train=str(data), test=str(data))})
    subset = write_subset(tmp_path / "subset.json", "code_reviewer", [0, 2])
    argv = [sys.executable, "-m", "src.icl_inference", "--model_name", "qwen25",
            "--dataset", "acra_by_ls_pt", "--num_examples", "0", "--batch-size", "8",
            "--results_dir", str(tmp_path), "--sample-index-file", subset, "--skip-if-complete"]
    path = tmp_path / "qwen25_icl_0ex_random_v2.json"
    identity = dict(model_name="qwen25", dataset="acra_by_ls_pt", num_examples=0,
                    selection_method="random", demo_seed=None, prompt_version="v2",
                    memory_profile="a100", batch_size=8, max_prompt_tokens=2048,
                    attn_implementation="sdpa", seed=42, allow_exact_match=False,
                    max_samples=None, llm_judge=False, sample_index_file=subset,
                    train_data_sha256=status.digest(str(data)), test_data_sha256=status.digest(str(data)),
                    sample_index_sha256=status.digest(subset))
    stamp = fingerprint(identity)
    Path(str(path) + ".identity.json").write_text(json.dumps(dict(identity=identity, fingerprint=stamp)))
    value = {k: 0 for k in ["bleu_score", "exact_match_score", "rouge_score", "bertscore", "meteor_score"]}
    value.update(run_fingerprint=stamp, num_test_samples=2, predictions=["x", "y"],
                 references=["r", "s"], prompts=["p", "q"], sample_indices=[0, 2])
    path.write_text(json.dumps(value))
    assert status.icl_result(argv) == path
    value.pop("sample_indices")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="coverage"):
        status.icl_result(argv)
