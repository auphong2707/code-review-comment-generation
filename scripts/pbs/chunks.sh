#!/usr/bin/env bash
# Canonical default inventory, derived from the existing Bash campaign previews.
# No Python, scheduler calls or artifact writes. IDs are one-based, stable for this source.
set -euo pipefail
cd "${RCG_ROOT:-$(dirname "${BASH_SOURCE[0]}")/../..}"
index=0
emit() { index=$((index + 1)); printf '%s\t%s\t%s\t%s\n' "$index" "$1" "$2" "$3"; }
emit phase 00_07_infrastructure all
for pair in '12_campaign_a A' '13_campaign_b B' '14_campaign_c C' '15_campaign_d D'; do
    read -r phase letter <<< "$pair"
    # Capture first so an invalid preview cannot silently produce a partial inventory.
    preview=$(unset RCG_PBS_STEP; DRY=1 RCG_JSON=0 RCG_PYTHON=/bin/false bash "scripts/slurm/$phase.sbatch")
    while read -r marker label rest; do
        [[ "$marker" == STEP ]] || continue
        emit step "$phase" "${label%:}"
    done <<< "$preview"
    emit gate "$letter" all
done
emit prepare J all
# At the default scope, 100 revision artifacts / 48 = 2–3 artifacts per job;
# 54 legacy artifacts / 18 = 3. Each artifact retains its complete row set.
for backend in qwen-revision llama-legacy llama-revision; do
    count=48; [[ "$backend" != llama-legacy ]] || count=18
    for ((shard=0; shard<count; shard++)); do emit judge "$backend" "$shard"; done
    emit summary "$backend" all
done
