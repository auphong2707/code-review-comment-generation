"""Opt-in durable progress primitives for campaigns spanning Slurm allocations.

No torch/transformers imports until a Trainer callback is constructed. The default
legacy runtime does not use these helpers. State is local to an isolated run root.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import tempfile

INCOMPLETE = 75
_stop_requested = False


def request_stop(signum=None, frame=None):
    global _stop_requested
    _stop_requested = True


def install_stop_handlers():
    global _stop_requested
    _stop_requested = False
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def stop_requested():
    return _stop_requested


def stop_if_requested():
    if _stop_requested:
        raise SystemExit(INCOMPLETE)


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RunLock:
    """Hold the returned object for the lifetime of the run (one writer only)."""
    def __init__(self, path):
        import fcntl
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(path, "a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError(f"Another process owns run lock: {path}") from None

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def bind_identity(path, identity):
    path = Path(path)
    expected = {"fingerprint": fingerprint(identity), "identity": identity}
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"Run configuration/input changed: {path}; use a new output root")
    else:
        atomic_json(path, expected)
    return expected["fingerprint"]


class PredictionJournal:
    """Atomic per-row progress; corpus metrics still run on the full ordered set."""
    def __init__(self, path, identity, count):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.fingerprint = bind_identity(self.path / "identity.json", {"run": identity, "count": count})
        self.count = count

    def get(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        path = self.path / f"row-{index:08d}.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("index") != index or value.get("fingerprint") != self.fingerprint:
            raise ValueError(f"Incompatible prediction row: {path}")
        return value["value"]

    def put(self, index, value):
        if not 0 <= index < self.count:
            raise IndexError(index)
        atomic_json(self.path / f"row-{index:08d}.json",
                    {"index": index, "fingerprint": self.fingerprint, "value": value})

    def values(self):
        values = [self.get(i) for i in range(self.count)]
        if any(value is None for value in values):
            raise ValueError("Prediction journal is incomplete")
        return values


def valid_checkpoint(path):
    path = Path(path)
    try:
        manifest = json.loads((path / "recovery_complete.json").read_text())
        state = json.loads((path / "trainer_state.json").read_text())
        if state["global_step"] != manifest["global_step"]:
            return False
        files = manifest["files"]
        if not {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"} <= files.keys():
            return False
        if not any(name.endswith((".safetensors", ".bin")) for name in files):
            return False
        return all((path / name).is_file() and (path / name).stat().st_size == size
                   for name, size in files.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_recovery_checkpoint(output_dir):
    root = Path(output_dir)
    candidates = [p for p in root.glob("checkpoint-*")
                  if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()]
    candidates.sort(key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for path in candidates:
        if valid_checkpoint(path):
            return str(path)
    if candidates:
        raise RuntimeError(f"No complete recovery checkpoint under {root}; refusing a fresh restart")
    return None


class RecoveryMixin:
    """Commit checkpoint metadata only after Trainer finishes writing full state."""
    def _save_checkpoint(self, model, trial, *args, **kwargs):
        path = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        marker = path / "recovery_complete.json"
        marker.unlink(missing_ok=True)
        limit = self.args.save_total_limit
        # Trainer must not rotate away the preceding valid save before this one commits.
        self.args.save_total_limit = None
        try:
            super()._save_checkpoint(model, trial, *args, **kwargs)
            files = {str(p.relative_to(path)): p.stat().st_size
                     for p in path.rglob("*") if p.is_file()}
            import torch
            peaks = {
                "peak_vram_allocated_gb": max((torch.cuda.max_memory_allocated(i) / 2**30
                                               for i in range(torch.cuda.device_count())), default=0),
                "peak_vram_reserved_gb": max((torch.cuda.max_memory_reserved(i) / 2**30
                                              for i in range(torch.cuda.device_count())), default=0),
            }
            atomic_json(marker, {"global_step": self.state.global_step, "files": files, **peaks})
            if not valid_checkpoint(path):
                marker.unlink(missing_ok=True)
                raise RuntimeError(f"Incomplete full-state checkpoint: {path}")
        finally:
            self.args.save_total_limit = limit
        # Keep at least two recovery points; Trainer separately protects the best model.
        self.args.save_total_limit = max(limit or 3, 3)
        try:
            self._rotate_checkpoints(use_mtime=False, output_dir=str(path.parent))
        finally:
            self.args.save_total_limit = limit
        stop_if_requested()


def recovery_trainer_class(base):
    return type("Recoverable" + base.__name__, (RecoveryMixin, base), {})


def attach_recovery(trainer, every_steps=100, stop_after_step=None):
    from transformers import TrainerCallback
    if every_steps < 1:
        raise ValueError("Recovery interval must be positive")
    if trainer.args.save_only_model or trainer.args.world_size != 1:
        raise ValueError("Recovery requires full-state saves on a single GPU/process")

    class RecoveryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if stop_after_step is not None and state.global_step >= stop_after_step:
                request_stop()
            if control.should_evaluate:
                return control  # step-evaluation probes must commit their evaluated state too
            # Trainer evaluates epochs AFTER this callback. Never publish a recovery
            # point that would make resume skip an as-yet unevaluated epoch.
            at_epoch_end = state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-8
            if str(args.eval_strategy).split(".")[-1].lower() == "epoch" and (
                    at_epoch_end or state.global_step >= state.max_steps):
                return control  # normal epoch evaluation/save commits, then honors stop
            if stop_requested() or state.global_step % every_steps == 0:
                trainer._save_checkpoint(trainer.model, trial=None)
                stop_if_requested()
            return control

    trainer.add_callback(RecoveryCallback())


def validate_training_result(path, expected_fingerprint):
    value = json.loads(Path(path).read_text())
    required = {"test_bleu_4", "test_exact_match", "test_rouge", "test_bertscore", "test_meteor"}
    rows = value.get("raw_data", [])
    if (value.get("run_fingerprint") != expected_fingerprint or not required <= value.keys()
            or not rows or value.get("expected_test_rows") != len(rows)
            or any(not isinstance(r.get("prediction"), str) or not isinstance(r.get("reference"), str)
                   or r["prediction"] == "<ERROR>" for r in rows)):
        raise ValueError(f"Invalid/incompatible complete training result: {path}")
    return value


def validate_icl_result(path, expected_fingerprint):
    value = json.loads(Path(path).read_text())
    required = {"bleu_score", "exact_match_score", "rouge_score", "bertscore", "meteor_score"}
    count = value.get("num_test_samples", 0)
    if (value.get("run_fingerprint") != expected_fingerprint or not required <= value.keys()
            or count < 1 or any(len(value.get(key, [])) != count
                              for key in ("predictions", "references", "prompts"))
            or any(not isinstance(p, str) or p == "<ERROR>" for p in value["predictions"])):
        raise ValueError(f"Invalid/incompatible complete ICL result: {path}")
    return value


def generation_identity(trainer, run_fingerprint):
    """Bind journals to the selected weights, including the resolved PEFT backbone."""
    checkpoint = trainer.state.best_model_checkpoint
    if checkpoint is None:
        checkpoint = latest_recovery_checkpoint(trainer.args.output_dir)
    if checkpoint is None:
        raise RuntimeError("No persisted model is available for resumable test generation")
    weights = {p.name: file_digest(p) for p in Path(checkpoint).iterdir()
               if p.suffix in (".safetensors", ".bin")}
    return {"run": run_fingerprint, "checkpoint_weights": weights,
            "base_revision": getattr(trainer, "recovery_model_revision", None)}


def model_revision(output_dir, model_id, resolved_revision, resumed=False):
    """Retain original hub revision when a full-model checkpoint loads locally."""
    path = Path(output_dir) / "model_identity.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["model_id"] != model_id:
            raise ValueError("Model identity changed while resuming")
        if resolved_revision is not None and saved["revision"] != resolved_revision:
            raise ValueError("Resolved backbone revision changed; use a new run root")
        if resolved_revision is None and not resumed and saved["revision"] is not None:
            raise ValueError("Cannot verify the original backbone revision")
        return saved["revision"]
    atomic_json(path, {"model_id": model_id, "revision": resolved_revision})
    return resolved_revision
