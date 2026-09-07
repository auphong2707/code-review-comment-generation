"""Crash/restart behavior; tiny CPU training, no model/data downloads."""
import json
from pathlib import Path
import pytest
from src.training import recovery


def test_rows_survive_restart_and_reject_changed_inputs(tmp_path):
    journal = recovery.PredictionJournal(tmp_path, {"dataset": "sha1"}, 3)
    journal.put(0, {"prediction": "first"})
    resumed = recovery.PredictionJournal(tmp_path, {"dataset": "sha1"}, 3)
    assert resumed.get(0) == {"prediction": "first"}
    assert resumed.get(1) is None
    with pytest.raises(ValueError, match="incomplete"):
        resumed.values()
    with pytest.raises(ValueError, match="changed"):
        recovery.PredictionJournal(tmp_path, {"dataset": "sha2"}, 3)
    (tmp_path / '.row-00000001.json.interrupted').write_text('{')
    assert resumed.get(1) is None
    resumed.put(1, {"prediction": "second"})
    resumed.put(2, {"prediction": "third"})
    assert len(resumed.values()) == 3


def test_duplicate_writer_rejected(tmp_path):
    with recovery.RunLock(tmp_path / "run.lock"):
        with pytest.raises(RuntimeError, match="Another process"):
            recovery.RunLock(tmp_path / "run.lock")
    with recovery.RunLock(tmp_path / "run.lock"):
        pass


def test_cache_append_after_torn_tail(tmp_path):
    from src.llm_as_judge.cache import JsonlCache
    path = tmp_path / "cache.jsonl"
    cache = JsonlCache(str(path))
    cache.set("first", {"score": 1})
    with path.open("ab") as stream:
        stream.write(b'{"key":"torn\xe2\x82')
    resumed = JsonlCache(str(path))
    resumed.set("second", {"score": 2})
    loaded = JsonlCache(str(path))
    assert loaded.get("first")["score"] == 1
    assert loaded.get("second")["score"] == 2


def test_actual_trainer_interrupted_resume_matches_continuous(tmp_path, monkeypatch):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback, set_seed
    monkeypatch.setenv("WANDB_MODE", "offline")
    torch.set_num_threads(1)
    config = GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=1,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    data = [{"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([1, 2, 3, 4])}
            for _ in range(12)]
    cls = recovery.recovery_trainer_class(Trainer)

    def build(root, callbacks=()):
        set_seed(42)
        args = TrainingArguments(output_dir=str(root), max_steps=6,
                                 per_device_train_batch_size=1, learning_rate=0.001,
                                 save_strategy="no", eval_strategy="no", report_to="none",
                                 disable_tqdm=True, use_cpu=True, save_total_limit=3)
        trainer = cls(model=GPT2LMHeadModel(config), args=args, train_dataset=data,
                      callbacks=list(callbacks))
        recovery.attach_recovery(trainer, every_steps=2)
        return trainer

    monkeypatch.setattr(recovery, "_stop_requested", False)
    continuous = build(tmp_path / "continuous")
    continuous.train()

    class StopAtThree(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 3:
                recovery.request_stop()

    interrupted = build(tmp_path / "resumed", [StopAtThree()])
    with pytest.raises(SystemExit) as exc:
        interrupted.train()
    assert exc.value.code == 75
    checkpoint = recovery.latest_recovery_checkpoint(tmp_path / "resumed")
    assert checkpoint.endswith("checkpoint-3")
    # A torn newer directory must not displace the previous committed snapshot.
    (tmp_path / "resumed" / "checkpoint-4").mkdir()
    assert recovery.latest_recovery_checkpoint(tmp_path / "resumed") == checkpoint
    monkeypatch.setattr(recovery, "_stop_requested", False)
    resumed = build(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=checkpoint)
    assert resumed.state.global_step == 6
    for name, expected in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], expected, rtol=0, atol=0)


def test_epoch_boundary_stop_keeps_validation_and_best_model(tmp_path, monkeypatch):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
    monkeypatch.setattr(recovery, "_stop_requested", False)
    torch.set_num_threads(1)
    data = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])} for _ in range(2)]
    args = TrainingArguments(output_dir=str(tmp_path), num_train_epochs=2,
                             per_device_train_batch_size=1, per_device_eval_batch_size=1,
                             save_strategy="epoch", eval_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="loss", greater_is_better=False,
                             report_to="none", use_cpu=True, disable_tqdm=True)
    class StopAtBoundary(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                recovery.request_stop()
    cls = recovery.recovery_trainer_class(Trainer)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=4, n_embd=8, n_layer=1, n_head=1))
    trainer = cls(model=model, args=args, train_dataset=data, eval_dataset=data, callbacks=[StopAtBoundary()])
    recovery.attach_recovery(trainer, 1)
    with pytest.raises(SystemExit) as exc:
        trainer.train()
    assert exc.value.code == 75
    path = recovery.latest_recovery_checkpoint(tmp_path)
    state = json.loads((Path(path) / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    assert state["best_metric"] is not None
    assert any("eval_loss" in entry for entry in state["log_history"])
    monkeypatch.setattr(recovery, "_stop_requested", False)
