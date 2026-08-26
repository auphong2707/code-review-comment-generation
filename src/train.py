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
                        choices=supported_models() + ["qwen25-0.5b-instruct", "qwen25_0.5b"],
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
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Print the resolved configuration as JSON and exit (no torch)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny offline run: max_steps=2, capped datasets, no upload")
    parser.add_argument("--attn-implementation", dest="attn_implementation", default=None,
                        help="Override attn_implementation (e.g. sdpa when flash-attn is unavailable)")
    parser.add_argument("--data_dir", dest="data_dir", default=None,
                        help="Override directory holding train/val/test jsonl files")
    return parser


def dry_run_payload(model, method, spec, args):
    lr = args.learning_rate if args.learning_rate is not None else 5e-05
    experiment_id = args.experiment_id if args.experiment_id is not None else 0
    official = resolve_run(spec, learning_rate=lr)
    sweep = resolve_run(spec, experiment_id=experiment_id)
    return {
        "model": model,
        "method": method,
        "experiment_name_official": official["experiment_name"],
        "experiment_name_sweep": sweep["experiment_name"],
        "experiment_results_dir": official["experiment_results_dir"],
        "spec": spec,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, method, spec = get_spec(args.model, args.method)

    if args.dry_run:
        print(json.dumps(dry_run_payload(model, method, spec, args), indent=2))
        return

    if args.learning_rate is None and args.experiment_id is None:
        raise SystemExit("error: provide --learning_rate (official) or --experiment_id (sweep)")

    if args.smoke:
        os.environ.setdefault("WANDB_MODE", "offline")

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

    set_seed()

    run_info = resolve_run(spec, experiment_id=args.experiment_id,
                           learning_rate=args.learning_rate)

    if spec["family"] == "causal":
        from src.training.causal_runner import run
    else:
        from src.training.seq2seq_runner import run

    run(model, method, spec, run_info, subset=args.subset, dataset_name=args.dataset,
        smoke=args.smoke, attn_implementation=args.attn_implementation)


if __name__ == "__main__":
    main()
