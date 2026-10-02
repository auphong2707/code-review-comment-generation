#!/usr/bin/env python3
"""Read-only completion gate for selected A5000 producer campaigns A/B/C/D.

Run lists come from the sbatch DRY JSON preview. Preserve RUN_P*/SEEDS scope when
checking; ONLY/SKIP filters are deliberately ignored by this whole-scope gate.
No model imports, inference, downloads or writes occur here.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.recovery import file_digest, fingerprint, validate_training_result, validate_icl_result
from src.train import build_parser, recovery_identity, resolve_protocol_overrides
from src.training.registry import get_spec, resolve_run
from src.training.memory_profile import apply_memory_profile
from src.icl.naming import icl_result_filename
from src.utils.constants import DATASET_CONFIGS, LR
from tools.select_sweep_lr import best_metric

SCRIPTS = {"A": "12_campaign_a", "B": "13_campaign_b", "C": "14_campaign_c", "D": "15_campaign_d"}
digest = lru_cache(maxsize=None)(file_digest)


@lru_cache(maxsize=None)
def row_count(path):
    with open(path) as stream:
        return sum(bool(line.strip()) for line in stream)


def commands(phase, results_root):
    env = dict(os.environ, DRY="1", RCG_JSON="1", RCG_PYTHON=sys.executable,
               RCG_ROOT=str(ROOT), RCG_RESULTS_ROOT=str(results_root))
    for key in ("ONLY", "SKIP", "RCG_PBS_STEP"):
        env.pop(key, None)
    # J checks several producers in one process. Their optional block meanings
    # differ (e.g. B/C P2 is supported while D P2 is forbidden).
    for key in ("RUN_P0", "RUN_P1", "RUN_P2", "SEEDS"):
        scoped = f"RCG_{phase}_{key}"
        if scoped in env:
            env[key] = env[scoped]
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm" / (SCRIPTS[phase] + ".sbatch"))],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"label":')]


def training_result(argv, phase):
    args = build_parser().parse_args(argv[3:])
    if args.seed == 42:
        args.seed = None
    model, method, spec = get_spec(args.model, args.method)
    spec, overrides = apply_memory_profile(spec, resolve_protocol_overrides(model, args),
                                          args.memory_profile, model, method)
    if phase == "D" and args.learning_rate is not None:
        sweep_root = Path(args.results_root) / "sweeps"
        metrics = [best_metric(sweep_root / f'{spec["name_stem"]}_{i}', True) for i in range(3)]
        if any(metric is None for metric in metrics):
            raise ValueError("14B official requires all three completed sweep results")
        args.learning_rate = LR[max(range(3), key=lambda i: metrics[i])]
    run = resolve_run(spec, args.experiment_id, args.learning_rate, args.seed, args.results_root)
    path = Path(run["experiment_results_dir"]) / "complete_results.json"
    expected = recovery_identity(spec, overrides, args, run, digest=digest)
    saved = json.loads((path.parent / "run_identity.json").read_text())
    if saved != {"identity": expected, "fingerprint": fingerprint(expected)}:
        raise ValueError(f"Configuration/input mismatch: {path.parent}")
    value = validate_training_result(path, saved["fingerprint"])
    expected_rows = int(row_count(DATASET_CONFIGS[args.dataset]["data_files"]["test"]) * args.subset)
    if len(value["raw_data"]) != expected_rows:
        raise ValueError(f"Test coverage incomplete: {path}")
    return path


def icl_result(argv):
    # All campaign CLI values are explicit scalar flags except skip-if-complete.
    flags = {}; values = iter(argv[3:])
    for flag in values:
        if flag == "--skip-if-complete":
            continue
        flags[flag[2:].replace("-", "_")] = next(values)
    expected = dict(model_name=flags["model_name"], dataset=flags["dataset"],
                    num_examples=int(flags["num_examples"]),
                    selection_method=flags.get("selection_method", "random"),
                    demo_seed=int(flags["demo_seed"]) if "demo_seed" in flags else None,
                    prompt_version="v2", memory_profile="a5000", batch_size=1,
                    max_prompt_tokens=2048, attn_implementation="sdpa", seed=42,
                    allow_exact_match=False, max_samples=None, llm_judge=False)
    name = icl_result_filename(expected["model_name"], expected["num_examples"],
                               expected["selection_method"], demo_seed=expected["demo_seed"],
                               prompt_version="v2", non_default=True)
    path = Path(flags["results_dir"]) / name
    saved = json.loads(Path(str(path) + ".identity.json").read_text())
    identity = saved["identity"]
    if saved["fingerprint"] != fingerprint(identity) or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Configuration mismatch: {path}")
    files = DATASET_CONFIGS[expected["dataset"]]["data_files"]
    for split in ("train", "test"):
        if identity.get(split + "_data_sha256") != digest(files[split]):
            raise ValueError(f"Input changed: {path}")
    value = validate_icl_result(path, saved["fingerprint"])
    if value["num_test_samples"] != row_count(files["test"]):
        raise ValueError(f"Test coverage incomplete: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=SCRIPTS)
    parser.add_argument("--results-root", default="assets/results/revision-a5000")
    args = parser.parse_args(argv)
    items = commands(args.phase, Path(args.results_root).resolve())
    failures = []; paths = set()
    for item in items:
        try:
            command = item["argv"]
            paths.add(training_result(command, args.phase) if command[2] == "src.train" else icl_result(command))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(item["label"])
            print(f'INCOMPLETE {item["label"]}: {exc}', file=sys.stderr)
    if not items or failures:
        print(f"Phase {args.phase}: incomplete ({len(failures)}/{len(items)} commands missing/invalid)")
        return 1
    print(f"Phase {args.phase}: selected scope complete; {len(items)} commands, {len(paths)} unique artifacts")
    if args.phase in ("B", "D"):
        print("Excluded hardware-infeasible rows remain unavailable, not completed research evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
