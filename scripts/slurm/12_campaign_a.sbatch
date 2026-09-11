#!/usr/bin/env bash
#SBATCH --job-name=rcg_phase_12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
# Submit from repository root, selecting the site's A5000 partition with sbatch.
# Manual resubmission of this SAME file recovers committed progress. No automatic requeue.
RCG_PHASE=12
source "${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"


WANDB_TAGS="${WANDB_TAGS:-campaignA}"
RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-0}"
SEEDS="${SEEDS:-42 1 2}"

# ===========================================================================
# P0 — seed replicates, TransReview, paper LRs (54 runs)
# ===========================================================================
if [[ "$RUN_P0" == "1" ]]; then
    banner "campaign A / P0 — TransReview seed replicates (order: seq2seq, 0.5B, 7B)"

    # "<model> <method> <paper LR>" — the LRs the paper reports for TransReview
    REPLICATES=(
        "codet5      ia3   0.0001"
        "codet5      lora  0.001"
        "codet5      qlora 5e-05"
        "codet5      fmft  0.0005"
        "plbart      ia3   5e-05"
        "plbart      lora  1e-05"
        "plbart      qlora 1e-05"
        "plbart      fmft  0.0001"
        "qwen25-0.5b ia3   0.0005"
        "qwen25-0.5b lora  0.0005"
        "qwen25-0.5b qlora 0.001"
        "qwen25-0.5b fmft  5e-05"
        "qwen25      ia3   0.001"
        "qwen25      lora  5e-05"
        "qwen25      qlora 5e-05"
        "qwen15      ia3   1e-05"
        "qwen15      lora  5e-05"
        "qwen15      qlora 5e-05"
    )
    # seed-major would leave a half-finished replicate set if the wall clock hits;
    # config-major keeps every finished config complete for all three seeds.
    for row in "${REPLICATES[@]}"; do
        read -r model method lr <<<"$row"
        for seed in $SEEDS; do
            train_official "a_p0_${model}_${method}_seed${seed}" trans_review \
                "$model" "$method" "$lr" --seed "$seed"
        done
    done
fi

# ===========================================================================
# P1 — HP transfer: does the subset-sweep LR win on full data? (10 commands; 8 additional outputs)
# ===========================================================================
if [[ "$RUN_P1" == "1" ]]; then
    banner "campaign A / P1 — full-data LR grid for two representative configs"

    # Ten grid commands; two reuse matching A5000 seed-42 baseline outputs.
    for lr in 1e-05 5e-05 0.0001 0.0005 0.001; do
        train_official "a_p1_hp_codet5_lora_$lr" trans_review codet5 lora "$lr"
    done
    for lr in 1e-05 5e-05 0.0001 0.0005 0.001; do
        train_official "a_p1_hp_qwen05b_lora_$lr" trans_review qwen25-0.5b lora "$lr"
    done
fi


finish_report
