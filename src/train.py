"""Unified training entry point replacing the 24 legacy src/models scripts.

Usage (mirrors the legacy per-script CLIs; run_training.sh dispatches here):
    python -m src.train --model qwen25 --method lora --learning_rate 5e-5
    python -m src.train --model codet5 --method fmft --experiment_id 0
    python -m src.train --model qwen25 --method lora --dry-run     # config only
    python -m src.train --model qwen25-0.5b --method lora --smoke  # 2-step run

--dry-run prints the resolved configuration as JSON and exits before any
torch/transformers import or login — it is the equivalence instrument checked
against the legacy scripts (plans/refactor-source/tools/compare_specs.py).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Keep direct-script invocation (python src/train.py) working
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from src.training.registry import get_spec, resolve_run, supported_methods, supported_models


def build_parser():
    parser = argparse.ArgumentParser(description="Unified training entry point")
    parser.add_argument("--model", required=True,
                        choices=supported_models() + ["qwen25-0.5b-instruct", "qwen25_0.5b",
                                                      "qwen25_14b"],
                        help="Model to train")
    parser.add_argument("--method", required=True,
                        choices=supported_methods() + ["ptuning_v2"],
                        help="Adaptation method")
    parser.add_argument("--experiment_id", type=int, default=None,
                        help="Experiment ID for tracking multiple runs (sweep mode)")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override the learning rate (official mode)")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of training data to use (0 to 1)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["trans_review", "acra_by_ls_pt"],
                        help="Dataset to train on (default: constants.DATASET_NAME)")
    parser.add_argument("--protocol", choices=["archived", "head"], default="archived",
                        help="Training protocol for TransReview runs: 'archived' replicates the "
                             "produced TR artifacts (D-025 overrides), 'head' keeps the spec "
                             "values. CodeReviewer runs are never affected.")
    parser.add_argument("--pin-peft-defaults", dest="pin_peft_defaults",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Pin archived PEFT module resolutions for reruns (e.g. codet5-IA3 "
                             "['v','wo','k']/['wo'] on TransReview, KI-015). Default: on for "
                             "--protocol archived, off for head.")
    parser.add_argument("--galore-variant", dest="galore_variant", default=None,
                        choices=["galore_adamw", "galore_adamw_8bit", "galore_adamw_8bit_layerwise"],
                        help="Override the GaLore optimizer variant (galore method only; e.g. "
                             "8bit_layerwise if 80 GB is exceeded — recorded via the overrides)")
    parser.add_argument("--recovery-steps", type=int, default=100,
                        help="Optimizer steps between memory-profile recovery checkpoints (default: 100)")
    parser.add_argument("--upload", action="store_true", help="Explicitly enable HF login/upload")
    parser.add_argument("--memory-profile", choices=["a100"], default=None,
                        help="Opt-in A100 80 GB profile (native protocol, batched generation, "
                             "durable recovery); requires an isolated --results-root and target "
                             "GPU probes")
    parser.add_argument("--gen-batch-size", dest="gen_batch_size", type=int, default=1,
                        help="Batch size for validation/test generation (1 = legacy per-sample "
                             "loops; D-028c)")
    parser.add_argument("--eval-max-samples", dest="eval_max_samples", type=int, default=None,
                        help="Cap the validation set used for per-epoch generation (recorded in "
                             "resources; never for seed replicates). Under --smoke it sizes the "
                             "test set instead (batched-equivalence instrument).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Non-default seed: seeds the run and suffixes the "
                             "experiment name with _seed<N> (default: legacy seed 42, no suffix)")
    parser.add_argument("--results-root", dest="results_root", default=None,
                        help="Root dir for the experiment dir (e.g. per-dataset revision roots); "
                             "default: the legacy per-model results_dir")
    parser.add_argument("--skip-if-complete", dest="skip_if_complete", action="store_true",
                        help="Exit 0 immediately when complete_results.json already exists "
                             "(idempotent campaign re-submission)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Print the resolved configuration as JSON and exit (no torch)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny offline run: max_steps=2, capped datasets, no upload")
    parser.add_argument("--lifecycle-probe", action="store_true",
                        help="Memory profiles only: two optimizer steps, validation, save/reload and test pilot")
    parser.add_argument("--probe-stop-after-step", type=int, default=None,
                        help="Lifecycle probe only: simulate allocation stop after a committed step")
    parser.add_argument("--probe", action="store_true",
                        help="Memory probe: one forward/backward/optimizer step at the real "
                             "batch size on 8 real samples, print peak VRAM + timing, exit "
                             "(no checkpoint; sizes 14B/GaLore runs before a sweep)")
    parser.add_argument("--attn-implementation", dest="attn_implementation", default=None,
                        help="Override attn_implementation (e.g. sdpa when flash-attn is unavailable)")
    parser.add_argument("--data_dir", dest="data_dir", default=None,
                        help="Override directory holding train/val/test jsonl files")
    return parser


def resolve_protocol_overrides(model, args):
    from src.training.protocols import overrides_for
    from src.utils.constants import DATASET_NAME

    return overrides_for(args.dataset or DATASET_NAME, model, args.protocol)


def dry_run_payload(model, method, spec, args):
    lr = args.learning_rate if args.learning_rate is not None else 5e-05
    experiment_id = args.experiment_id if args.experiment_id is not None else 0
    official = resolve_run(spec, learning_rate=lr, seed=args.seed,
                           results_root=args.results_root)
    sweep = resolve_run(spec, experiment_id=experiment_id, seed=args.seed,
                        results_root=args.results_root)
    return {
        "model": model,
        "method": method,
        "experiment_name_official": official["experiment_name"],
        "experiment_name_sweep": sweep["experiment_name"],
        "experiment_results_dir": official["experiment_results_dir"],
        "spec": spec,
        "protocol_overrides": resolve_protocol_overrides(model, args),
    }


def recovery_identity(spec, profile_overrides, args, run_info, digest=None):
    from src.training.recovery import file_digest
    digest = digest or file_digest
    from src.utils.constants import DATASET_CONFIGS, DATASET_NAME
    dataset = args.dataset or DATASET_NAME
    files = DATASET_CONFIGS[dataset]["data_files"]
    if args.data_dir:
        files = {k: Path(args.data_dir) / Path(v).name for k, v in files.items()}
    identity = {
        "spec": spec, "protocol_overrides": profile_overrides,
        "dataset": dataset, "data": {k: digest(v) for k, v in files.items()},
        "seed": args.seed, "learning_rate": run_info["learning_rate"],
        "subset": args.subset, "eval_max_samples": args.eval_max_samples,
        "gen_batch_size": args.gen_batch_size, "memory_profile": args.memory_profile,
        "attention": args.attn_implementation, "pin_peft_defaults": args.pin_peft_defaults,
        "lifecycle_probe": args.lifecycle_probe,
    }
    return identity


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, method, spec = get_spec(args.model, args.method)
    if args.lifecycle_probe and (not args.memory_profile or args.probe or args.smoke):
        raise SystemExit("--lifecycle-probe requires --memory-profile and excludes --probe/--smoke")
    if args.probe_stop_after_step is not None and (
            not args.lifecycle_probe or args.probe_stop_after_step not in (1, 2)):
        raise SystemExit("--probe-stop-after-step requires a lifecycle probe and step 1 or 2")
    if args.recovery_steps < 1:
        raise SystemExit("--recovery-steps must be positive")
    if args.memory_profile:
        if args.seed == 42:
            args.seed = None  # effective default seed shares the canonical output name
        if not args.results_root and not args.probe:
            raise SystemExit("--memory-profile requires an isolated --results-root")
        if args.gen_batch_size < 1:
            raise SystemExit("--gen-batch-size must be positive")
        if args.attn_implementation not in (None, "sdpa"):
            raise SystemExit("The A100 profile requires SDPA")
        args.attn_implementation = "sdpa"
        from src.training.memory_profile import apply_memory_profile
        spec, profile_overrides = apply_memory_profile(
            spec, resolve_protocol_overrides(model, args), args.memory_profile, model, method)
    else:
        profile_overrides = None

    if args.dry_run:
        payload = dry_run_payload(model, method, spec, args)
        if profile_overrides is not None:
            payload["protocol_overrides"] = profile_overrides
            payload["memory_profile"] = args.memory_profile
        print(json.dumps(payload, indent=2))
        return

    if args.learning_rate is None and args.experiment_id is None:
        if args.smoke or args.probe or args.lifecycle_probe:
            args.learning_rate = 5e-05
        else:
            raise SystemExit("error: provide --learning_rate (official) or --experiment_id (sweep)")

    if args.smoke or args.probe:
        os.environ.setdefault("WANDB_MODE", "offline")

    if args.probe and args.results_root is None:
        # never touch (or resume from) the real experiment dir for a probe
        import tempfile

        args.results_root = tempfile.mkdtemp(prefix="train-probe-")

    run_info = resolve_run(spec, experiment_id=args.experiment_id,
                           learning_rate=args.learning_rate,
                           seed=args.seed, results_root=args.results_root)

    from contextlib import nullcontext
    from src.training.recovery import RunLock, bind_identity, file_digest, install_stop_handlers
    recover = bool(args.memory_profile and not (args.probe or args.smoke))
    run_dir = Path(run_info["experiment_results_dir"])
    run_info["lifecycle_probe"] = args.lifecycle_probe
    run_info["probe_stop_after_step"] = args.probe_stop_after_step
    with (RunLock(run_dir / "run.lock") if recover else nullcontext()):
        if recover:
            identity = recovery_identity(spec, profile_overrides, args, run_info)
            if not (run_dir / "run_identity.json").exists() and (
                    (run_dir / "complete_results.json").exists() or list(run_dir.glob("checkpoint-*"))):
                raise SystemExit("Refusing to adopt existing results/checkpoints without a profile identity")
            run_info["fingerprint"] = bind_identity(run_dir / "run_identity.json", identity)
            run_info["recovery"] = True
            install_stop_handlers()
        if args.skip_if_complete and os.path.exists(
                os.path.join(run_info["experiment_results_dir"], "complete_results.json")):
            if run_info.get("recovery"):
                from src.training.recovery import validate_training_result
                validate_training_result(Path(run_info["experiment_results_dir"]) / "complete_results.json",
                                         run_info["fingerprint"])
            print(f"skip-if-complete: {run_info['experiment_results_dir']} already has "
                  "complete_results.json — nothing to do")
            return

        if args.data_dir:
            # Point the dataset registry at an alternate data directory (smoke runs
            # on machines where the canonical paths do not exist)
            from src.utils.constants import DATASET_CONFIGS, DATASET_NAME

            dataset_name = args.dataset or DATASET_NAME
            files = DATASET_CONFIGS[dataset_name]["data_files"]
            DATASET_CONFIGS[dataset_name]["data_files"] = {
                split: os.path.join(args.data_dir, os.path.basename(path))
                for split, path in files.items()
            }

        from src.utils.helper import set_seed

        set_seed() if args.seed is None else set_seed(args.seed)

        if spec["family"] == "causal":
            from src.training.causal_runner import run
        else:
            from src.training.seq2seq_runner import run

        protocol_overrides = (profile_overrides if profile_overrides is not None
                              else resolve_protocol_overrides(model, args))
        if args.galore_variant and method == "galore":
            protocol_overrides["optim"] = args.galore_variant

        from src.training.specs_revision import peft_pin_for
        from src.utils.constants import DATASET_NAME

        pin_enabled = (args.pin_peft_defaults if args.pin_peft_defaults is not None
                       else args.protocol == "archived")
        peft_overrides = (peft_pin_for(args.dataset or DATASET_NAME, model, method)
                          if pin_enabled else {})

        run(model, method, spec, run_info, subset=args.subset, dataset_name=args.dataset,
            smoke=args.smoke, attn_implementation=args.attn_implementation, seed=args.seed,
            protocol=args.protocol, protocol_overrides=protocol_overrides,
            gen_batch_size=args.gen_batch_size, eval_max_samples=args.eval_max_samples,
            peft_overrides=peft_overrides, probe=args.probe, memory_profile=args.memory_profile,
            recovery_steps=args.recovery_steps, upload=args.upload)

if __name__ == "__main__":
    main()
