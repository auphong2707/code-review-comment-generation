#!/usr/bin/env python3
"""Committed CodeReviewer judge subset (phase 05, D-019).

Draws 2,000 test indices with proportional stratification by `lang`
(largest-remainder rounding) using random.Random(42), so the Llama judge
scores an identical, reproducible subset on every run.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TEST = "assets/data/Code_Review/msg-test.jsonl"
DEFAULT_OUT = "paper/generated/cr_judge_subset.json"


def stratified_indices(langs, n, seed):
    """Largest-remainder proportional allocation per lang, then a seeded
    sample within each lang."""
    total = len(langs)
    by_lang = {}
    for index, lang in enumerate(langs):
        by_lang.setdefault(lang, []).append(index)

    quotas = {lang: n * len(indices) / total for lang, indices in by_lang.items()}
    counts = {lang: int(quota) for lang, quota in quotas.items()}
    remainder = n - sum(counts.values())
    for lang, _ in sorted(quotas.items(), key=lambda item: item[1] - int(item[1]),
                          reverse=True)[:remainder]:
        counts[lang] += 1

    rng = random.Random(seed)
    selected = []
    for lang in sorted(by_lang):
        selected.extend(rng.sample(by_lang[lang], counts[lang]))
    return sorted(selected), {lang: counts[lang] for lang in sorted(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", default=DEFAULT_TEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.test_path, encoding="utf-8") as handle:
        langs = [json.loads(line).get("lang", "unknown") for line in handle if line.strip()]

    indices, per_lang = stratified_indices(langs, args.n, args.seed)
    payload = {
        "dataset": "code_reviewer",
        "seed": args.seed,
        "n": args.n,
        "indices": indices,
        "per_lang": per_lang,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(indices)} indices ({Counter(per_lang).total()} allocated) to {out_path}")


if __name__ == "__main__":
    main()
