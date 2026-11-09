"""TrainingArguments construction for every spec (phase 02).

The refactor gate was dry-run only, so nothing ever built TrainingArguments
from the specs; this closes that hole (KI-011: qwen15 non-FMFT specs carry
deepspeed=None and used to crash on os.path.exists(None)).
"""

import pytest

from src.training.peft_setup import build_training_args, static_marker_context
from src.training.registry import resolve_run
from src.training.specs import LEGACY_SPECS


def make_ctx(spec, tmp_path):
    run_info = resolve_run(spec, learning_rate=5e-05)
    run_info["experiment_results_dir"] = str(tmp_path / run_info["experiment_name"])
    return static_marker_context(run_info, spec)


@pytest.mark.parametrize("key", sorted(LEGACY_SPECS), ids="_".join)
def test_build_training_args_all_specs(key, tmp_path):
    spec = LEGACY_SPECS[key]
    ctx = make_ctx(spec, tmp_path)
    ds_path = spec["training_args"].get("deepspeed")
    if ds_path:  # FMFT deepspeed specs: KI-003 clear error (config not in repo)
        with pytest.raises(FileNotFoundError, match="KI-003"):
            build_training_args(spec, ctx, smoke=False)
        args = build_training_args(spec, ctx, smoke=True)  # smoke pops deepspeed
    else:  # incl. qwen15 deepspeed=None specs, which must not raise (KI-011)
        args = build_training_args(spec, ctx, smoke=False)
    assert args.metric_for_best_model in ("eval_bleu_4", None)
    assert args.deepspeed is None


def test_overrides_reach_training_args(tmp_path):
    spec = LEGACY_SPECS[("codet5", "lora")]
    ctx = make_ctx(spec, tmp_path)
    args = build_training_args(spec, ctx, overrides={"seed": 7, "data_seed": 7})
    assert args.seed == 7 and args.data_seed == 7
    assert build_training_args(spec, ctx).seed == 42  # default untouched


def test_skip_if_complete(tmp_path):
    from src import train

    run_dir = tmp_path / "codet5_lora_official_5e-05"
    run_dir.mkdir()
    (run_dir / "complete_results.json").write_text("{}")
    # returns before any runner import/training
    train.main(["--model", "codet5", "--method", "lora", "--learning_rate", "5e-05",
                "--results-root", str(tmp_path), "--skip-if-complete"])


def test_no_skip_without_marker(tmp_path, monkeypatch):
    import src.utils.helper as helper
    from src import train

    class Sentinel(Exception):
        pass

    def boom(*args, **kwargs):
        raise Sentinel

    monkeypatch.setattr(helper, "set_seed", boom)
    with pytest.raises(Sentinel):  # empty dir: proceeds past the skip check
        train.main(["--model", "codet5", "--method", "lora", "--learning_rate", "5e-05",
                    "--results-root", str(tmp_path), "--skip-if-complete"])
