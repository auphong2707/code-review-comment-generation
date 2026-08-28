#!/usr/bin/env python3
"""Emit a --jobs-file for tools/update_llm_judge_tables.py (phase 05 skeleton).

Until the phase-08 run manifest (src/analysis/manifest.py) exists, this
accepts a runs.json — a list of dicts with at least
{dataset, model, method, source, result, kind} (tables optional) — and writes
JOBS-shaped entries the driver can consume via --jobs-file. The frozen JOBS
table in the driver is never touched (D-026).
"""

import argparse
import json
from pathlib import Path

REQUIRED = ("dataset", "model", "method", "source", "result", "kind")
DEFAULT_OUT = "plans/jss-revision/jobs_revision.json"


def to_job(entry):
    missing = [key for key in REQUIRED if key not in entry]
    if missing:
        raise SystemExit(f"runs.json entry missing {missing}: {entry}")
    if entry["kind"] not in ("complete", "icl"):
        raise SystemExit(f"kind must be 'complete' or 'icl': {entry}")
    return {
        "tables": str(entry.get("tables", "revision")),
        "source": entry["source"],
        "dataset": entry["dataset"],
        "model": entry["model"],
        "method": entry["method"],
        "result": entry["result"],
        "kind": entry["kind"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_json", help="Input runs.json (list of run dicts)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    entries = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))
    jobs = [to_job(entry) for entry in entries]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(jobs)} job(s) to {out_path}")


if __name__ == "__main__":
    main()
