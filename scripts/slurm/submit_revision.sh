#!/usr/bin/env bash
# User-run launcher. Preview by default; --submit explicitly contacts Slurm.
set -euo pipefail
mode="${1:---preview}"
[[ "$mode" == --preview || "$mode" == --submit ]] || {
    echo "Usage: bash scripts/slurm/submit_revision.sh [--preview|--submit]" >&2; exit 2;
}
ROOT_DIR="${RCG_ROOT:-$PWD}"
cd "$ROOT_DIR"
[[ -f scripts/slurm/common.sh ]] || { echo "Run from the repository root" >&2; exit 2; }
export RCG_ROOT="$PWD" RCG_TARGET_GPU=a100_80gb
# A single chain serializes shared controls and keeps all campaigns behind smoke.
# Deliberately fixed default scope; advanced/partial runs use individual sbatch files.
for key in RUN_P0 RUN_P1 RUN_P2 SEEDS ONLY SKIP RUN_GALORE RUN_14B_LORA ICL_14B \
           RCG_PROBES J_REQUIRE_PHASES RUN_REVISION RUN_LEGACY RUN_QWEN RUN_LLAMA \
           RCG_A_RUN_P0 RCG_A_RUN_P1 RCG_A_RUN_P2 RCG_A_SEEDS \
           RCG_B_RUN_P0 RCG_B_RUN_P1 RCG_B_RUN_P2 RCG_B_SEEDS \
           RCG_C_RUN_P0 RCG_C_RUN_P1 RCG_C_RUN_P2 RCG_C_SEEDS \
           RCG_D_RUN_P0 RCG_D_RUN_P1 RCG_D_RUN_P2 RCG_D_SEEDS; do
    [[ -z "${!key:-}" ]] || { echo "Unset $key for the default full-chain launcher" >&2; exit 2; }
done
[[ "${RCG_JSON:-0}" == 0 && "${DRY:-0}" == 0 ]] || {
    echo "Unset DRY and RCG_JSON; use --preview for this launcher" >&2; exit 2;
}
options=(--parsable --export=ALL --chdir="$PWD")
[[ -z "${RCG_PARTITION:-}" ]] || options+=(--partition="$RCG_PARTITION")
[[ -z "${RCG_ACCOUNT:-}" ]] || options+=(--account="$RCG_ACCOUNT")
[[ -z "${RCG_QOS:-}" ]] || options+=(--qos="$RCG_QOS")
# GRES names are site-specific. Runtime independently rejects 40GB/MIG/other GPUs.
[[ -z "${RCG_GRES:-}" ]] || options+=(--gres="$RCG_GRES")
if [[ "$mode" == --submit ]]; then
    command -v sbatch >/dev/null
    mkdir -p assets/logs/slurm
fi
previous=""
for script in 00_07_infrastructure 12_campaign_a 13_campaign_b 14_campaign_c 15_campaign_d 16_campaign_j; do
    command_args=(sbatch "${options[@]}")
    [[ -z "$previous" ]] || command_args+=(--dependency="afterok:$previous")
    command_args+=("scripts/slurm/$script.sbatch")
    if [[ "$mode" == --preview ]]; then
        printf '%q ' "${command_args[@]}"; printf '\n'
        previous="JOB_ID_$script"
    else
        submitted=$("${command_args[@]}")
        previous="${submitted%%;*}"
        [[ "$previous" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $submitted" >&2; exit 2; }
        printf '%s %s\n' "$previous" "$script"
    fi
done
