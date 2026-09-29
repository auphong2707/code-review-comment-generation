#!/usr/bin/env bash
# Preview is safe on Ubuntu; submission belongs on the SUTD PBS login node.
set -euo pipefail
mode="${1:---preview}"
[[ $# -le 1 && ( "$mode" == --preview || "$mode" == --submit ) ]] || {
    echo "Usage: bash scripts/pbs/submit_revision.sh [--preview|--submit]" >&2; exit 2;
}
cd "${RCG_ROOT:-$PWD}"
[[ -f scripts/pbs/run_phase.sh ]] || { echo "Run from the repository root" >&2; exit 2; }
export RCG_ROOT="$PWD" RCG_TARGET_GPU=a100_80gb
# Share the existing full-chain scope validation (Bash-only, no submission/writes).
# This prevents a filtered smoke or producer from satisfying the default chain.
bash scripts/slurm/submit_revision.sh --preview >/dev/null
for key in RCG_PARTITION RCG_ACCOUNT RCG_QOS RCG_GRES; do
    [[ -z "${!key:-}" ]] || { echo "Unset Slurm-only $key; use RCG_PBS_* settings" >&2; exit 2; }
done
options=(-V -q "${RCG_PBS_QUEUE:-a100q}")
[[ -z "${RCG_PBS_PROJECT:-}" ]] || options+=(-P "$RCG_PBS_PROJECT")
if [[ -n "${RCG_PBS_WALLTIME:-}" ]]; then
    [[ "$RCG_PBS_WALLTIME" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] || {
        echo "RCG_PBS_WALLTIME must be HH:MM:SS" >&2; exit 2;
    }
    seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
    (( seconds > 600 )) || { echo "Walltime must exceed 600 seconds" >&2; exit 2; }
    options+=(-l "walltime=$RCG_PBS_WALLTIME")
fi
[[ "$mode" != --submit ]] || command -v qsub >/dev/null
previous=""
for phase in 00_07_infrastructure 12_campaign_a 13_campaign_b 14_campaign_c 15_campaign_d 16_campaign_j; do
    command_args=(qsub "${options[@]}")
    [[ -z "$previous" ]] || command_args+=(-W "depend=afterok:$previous")
    command_args+=("scripts/pbs/$phase.pbs")
    if [[ "$mode" == --preview ]]; then
        printf '%q ' "${command_args[@]}"; printf '\n'
        previous="JOB_ID_$phase"
    else
        submitted=$("${command_args[@]}")
        # Keep the server suffix: e.g. 123456.aimc-hn1, not just 123456.
        [[ "$submitted" =~ ^[0-9]+(\.[A-Za-z0-9_.-]+)?$ ]] || {
            echo "Unexpected qsub response: $submitted; inspect myjob before retrying" >&2; exit 2;
        }
        previous="$submitted"
        printf '%s %s\n' "$previous" "$phase"
    fi
done
