#!/usr/bin/env python3
"""Emit a --jobs-file for tools/update_llm_judge_tables.py from the revision tree.

The frozen JOBS table in the driver covers the produced Tables 1-5 only (KI-002).
Campaign runs land under `assets/results/revision/`, so the judge campaign needs a
jobs file listing exactly the artifacts that exist right now — this scans the tree
and writes one (`--only-existing` by construction, so re-running it after every
block is the intended workflow):

    python tools/revision_judge_jobs.py --out assets/models/llm-as-judge/jobs_revision.json

Expected layout (created by scripts/slurm/*.sbatch):

    assets/results/revision/<ds>/<stem>_official_<lr>[_seed<n>]/complete_results.json
    assets/results/revision/<ds>/sweeps/<stem>_<id>/...          (skipped unless --include-sweeps)
    assets/results/revision/icl/<ds>/<model>_icl_<k>ex_<sel>[_s<seed>][_v2].json

Run names are resolved against the training registry (never re-parsed by hand), so
`qwen25_0.5b_lora` and `qwen25_14b_qlora` split correctly. `model` + `method` carry
the LR/seed/shot variant because the driver derives its job id — and therefore the
scored file name — from them.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.registry import SPECS

# dataset directory name -> the driver's dataset key (DATASETS in the driver)
DATASETS = {"trans_review": "trans_review", "acra_by_ls_pt": "code_reviewer"}

MODEL_LABELS = {
    "codet5": "CodeT5",
    "plbart": "PLBART",
    "qwen15": "CodeQwen1.5-7B",
    "qwen25": "Qwen2.5-7B",
    "qwen25-0.5b": "Qwen2.5-0.5B",
    "qwen25-14b": "Qwen2.5-14B",
}
METHOD_LABELS = {
    "fmft": "FMFT", "ia3": "IA3", "lora": "LoRA", "qlora": "QLoRA",
    "ptuning": "P-Tuning", "dora": "DoRA", "galore": "GaLore",
}
# GaLore is full-parameter, FMFT is the frozen full-FT row: both group as "fmft"
FULL_FT_METHODS = {"fmft", "galore"}

# <stem>_official_<lr> or <stem>_<sweep id>, either with an optional _seed<n>
RUN_RE = re.compile(r"^(?P<stem>.+?)_(?:official_(?P<lr>[^_]+)|(?P<sweep>\d+))"
                    r"(?:_seed(?P<seed>\d+))?$")
ICL_RE = re.compile(r"^(?P<model>.+?)_icl_(?P<k>\d+)ex_(?P<sel>[a-z0-9]+)"
                    r"(?:_s(?P<seed>\d+))?(?:_(?P<version>v\d+))?\.json$")

STEMS = {spec["name_stem"]: (model, method) for (model, method), spec in SPECS.items()}


def training_job(run_dir: Path, dataset: str, tables: str):
    """One JOBS entry for a finished training run, or None if unrecognised."""
    match = RUN_RE.match(run_dir.name)
    if not match or match["stem"] not in STEMS:
        return None
    model, method = STEMS[match["stem"]]
    variant = f"lr={match['lr']}" if match["lr"] else f"sweep={match['sweep']}"
    if match["seed"]:
        variant += f" seed={match['seed']}"
    return {
        "tables": tables,
        "source": "fmft" if method in FULL_FT_METHODS else "peft",
        "dataset": DATASETS[dataset],
        "model": MODEL_LABELS.get(model, model),
        "method": f"{METHOD_LABELS.get(method, method)} {variant}",
        "result": str(run_dir / "complete_results.json"),
        "kind": "complete",
    }


def icl_job(result_path: Path, dataset: str, tables: str):
    """One JOBS entry for an ICL result file, or None if unrecognised."""
    match = ICL_RE.match(result_path.name)
    if not match:
        return None
    method = f"{match['k']}-shot {match['sel']}"
    if match["seed"]:
        method += f" s{match['seed']}"
    method += f" {match['version'] or 'v1'}"
    return {
        "tables": tables,
        "source": "icl",
        "dataset": DATASETS[dataset],
        "model": MODEL_LABELS.get(match["model"], match["model"]),
        "method": method,
        "result": str(result_path),
        "kind": "icl",
    }


def collect(root: Path, tables: str, include_sweeps: bool):
    jobs, skipped = [], []
    for dataset in sorted(DATASETS):
        dataset_root = root / dataset
        if dataset_root.is_dir():
            for result in sorted(dataset_root.rglob("complete_results.json")):
                run_dir = result.parent
                if not include_sweeps and "sweeps" in run_dir.relative_to(dataset_root).parts:
                    continue
                job = training_job(run_dir, dataset, tables)
                (jobs if job else skipped).append(job or str(run_dir))

        icl_root = root / "icl" / dataset
        if icl_root.is_dir():
            for result in sorted(icl_root.glob("*.json")):
                job = icl_job(result, dataset, tables)
                (jobs if job else skipped).append(job or str(result))
    return jobs, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="assets/results/revision",
                        help="Root of the revision artifacts (default: %(default)s)")
    parser.add_argument("--out", default="assets/models/llm-as-judge/jobs_revision.json",
                        help="Jobs file to write (default: %(default)s)")
    parser.add_argument("--tables", default="revision",
                        help="Value of the 'tables' field, part of every job id "
                             "(default: %(default)s)")
    parser.add_argument("--include-sweeps", action="store_true",
                        help="Also judge sweep runs (default: officials and ICL only)")
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    if not root.is_dir():
        print(f"revision_judge_jobs: no such results root: {root}", file=sys.stderr)
        return 1

    jobs, skipped = collect(root, args.tables, args.include_sweeps)
    for path in skipped:
        print(f"  ignored (unrecognised name): {path}", file=sys.stderr)
    if not jobs:
        print(f"revision_judge_jobs: no judgeable artifacts under {root}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    kinds = {}
    for job in jobs:
        kinds[job["kind"]] = kinds.get(job["kind"], 0) + 1
    print(f"wrote {len(jobs)} job(s) to {out_path} "
          f"({', '.join(f'{count} {kind}' for kind, count in sorted(kinds.items()))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
