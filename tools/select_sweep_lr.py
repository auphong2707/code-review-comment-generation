#!/usr/bin/env python3
"""Pick the official learning rate from a finished 3-point sweep (D-028d).

The campaign sbatch files run `./run_training.sh <model> <method> sweep 0 2 <subset>`,
which writes `<sweep-root>/<stem>_<id>/` with lr = constants.LR[id]. This reads each
run's best validation BLEU-4 (`trainer_state.json::best_metric`, the checkpoint
selector pinned by `metric_for_best_model="eval_bleu_4"`) and prints the winning LR
in the exact `str(lr)` form the official experiment dir uses.

    python tools/select_sweep_lr.py --sweep-root assets/results/revision/trans_review/sweeps \
        --stem qwen25_galore                  # table on stderr, winning LR on stdout
    python tools/select_sweep_lr.py ... --quiet   # LR only (for $(...) capture)

Exits 1 when no sweep run has a readable metric, so a campaign step can skip the
official run instead of training at an arbitrary LR.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.constants import LR


def best_metric(run_dir: Path, require_complete=False):
    """Best eval_bleu_4 recorded for one sweep run (None if it never evaluated)."""
    if require_complete:
        from src.training.recovery import fingerprint, validate_training_result, valid_checkpoint
        try:
            identity = json.loads((run_dir / "run_identity.json").read_text())
            if identity["fingerprint"] != fingerprint(identity["identity"]):
                return None
            validate_training_result(run_dir / "complete_results.json", identity["fingerprint"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
    if require_complete:
        states = [p for p in states if valid_checkpoint(p.parent)]
    if not require_complete and (run_dir / "trainer_state.json").exists():
        states.append(run_dir / "trainer_state.json")
    values = []
    for state_path in states:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("best_metric") is not None:
            values.append(float(state["best_metric"]))
        values.extend(
            float(entry["eval_bleu_4"])
            for entry in state.get("log_history", [])
            if entry.get("eval_bleu_4") is not None
        )
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-root", required=True,
                        help="Directory holding the <stem>_<id> sweep runs")
    parser.add_argument("--stem", required=True,
                        help="Experiment name stem, e.g. qwen25_galore or qwen25_14b_qlora")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2],
                        help="Sweep experiment ids to compare (default: the 3-point bracket 0 1 2)")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the winning LR (no table)")
    parser.add_argument("--require-complete", action="store_true",
                        help="Require every bracket run to have validated profile final results and committed state")
    args = parser.parse_args(argv)
    if any(i < 0 or i >= len(LR) for i in args.ids):
        parser.error("Sweep ids must index the configured LR grid")

    root = Path(args.sweep_root)
    rows = []
    for experiment_id in args.ids:
        run_dir = root / f"{args.stem}_{experiment_id}"
        rows.append((experiment_id, LR[experiment_id], run_dir,
                     best_metric(run_dir, args.require_complete) if run_dir.is_dir() else None))

    if not args.quiet:
        print(f"sweep {args.stem} under {root}", file=sys.stderr)
        for experiment_id, lr, run_dir, metric in rows:
            state = "missing" if not run_dir.is_dir() else (
                "no eval yet" if metric is None else f"eval_bleu_4={metric:.6f}")
            print(f"  id={experiment_id} lr={lr!s:<8} {state:<24} {run_dir}", file=sys.stderr)

    if args.require_complete and any(row[3] is None for row in rows):
        print("select_sweep_lr: incomplete bracket; official run is blocked", file=sys.stderr)
        return 1

    scored = [row for row in rows if row[3] is not None]
    if not scored:
        print(f"select_sweep_lr: no sweep run with a validation metric under {root} "
              f"for stem '{args.stem}'", file=sys.stderr)
        return 1

    winner = max(scored, key=lambda row: row[3])
    if not args.quiet:
        print(f"  -> best: id={winner[0]} lr={winner[1]} (eval_bleu_4={winner[3]:.6f})",
              file=sys.stderr)
    print(str(winner[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
