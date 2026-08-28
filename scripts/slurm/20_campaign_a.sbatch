#!/usr/bin/env bash
#SBATCH --job-name=rcg_campaign_a
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1            # EDIT: plain gpu:1 if the cluster has no typed gres
#SBATCH --time=72:00:00              # P0 is ~110 h: expect 2 submissions (runs resume/skip)
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
#
# Campaign A (phase 12) — QUEUE POSITION 2. TransReview variance + HP transfer.
#   P0  seed replicates: 18 configs x seeds {1,2} = 36 runs   (~110 h)  R1.1
#   P1  HP transfer: 2 configs x 4 missing full-data LRs = 8 runs (~10 h)  R2.7
#
#     sbatch scripts/slurm/20_campaign_a.sbatch                    # P0
#     RUN_P0=0 RUN_P1=1 sbatch scripts/slurm/20_campaign_a.sbatch  # P1
#     ONLY='seed1' sbatch scripts/slurm/20_campaign_a.sbatch       # one seed only
#     DRY=1 bash scripts/slurm/20_campaign_a.sbatch                # print the queue only
#
# Every run uses --protocol archived (D-025: the produced TransReview artifacts were
# trained at epochs/batch 15/16, 15/8, 8/2 — NOT the HEAD CodeReviewer protocol) and
# lands in assets/results/revision/trans_review/<stem>_official_<lr>_seed<n>/, so the
# frozen seed-42 artifacts are never touched. CodeReviewer gets no replicates by
# design (paired bootstrap CIs instead, phase 10).
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WANDB_TAGS="${WANDB_TAGS:-campaignA}"
RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-0}"
SEEDS="${SEEDS:-1 2}"

# ===========================================================================
# P0 — seed replicates, TransReview, paper LRs (36 runs)
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
    # config-major keeps every finished config complete for both seeds.
    for row in "${REPLICATES[@]}"; do
        read -r model method lr <<<"$row"
        for seed in $SEEDS; do
            train_official "a_p0_${model}_${method}_seed${seed}" trans_review \
                "$model" "$method" "$lr" --seed "$seed"
        done
    done
fi

# ===========================================================================
# P1 — HP transfer: does the subset-sweep LR win on full data? (8 runs)
# ===========================================================================
if [[ "$RUN_P1" == "1" ]]; then
    banner "campaign A / P1 — full-data LR grid for two representative configs"

    # Seed 42 (default, no suffix). The full LR grid = constants.LR. The phase-12
    # plan lists only the 4 *missing* LRs per config; the paper LR is re-run here
    # too (+2 runs, ~2.5 h) because the frozen artifact was generated unbatched and
    # KI-016 makes batched vs unbatched generations non-identical — a grid whose
    # rows come from two generation paths is not a like-for-like comparison.
    # Drop the paper LR from the loops to follow the plan's 8-run version exactly.
    for lr in 1e-05 5e-05 0.0001 0.0005 0.001; do
        train_official "a_p1_hp_codet5_lora_$lr" trans_review codet5 lora "$lr"
    done
    for lr in 1e-05 5e-05 0.0001 0.0005 0.001; do
        train_official "a_p1_hp_qwen05b_lora_$lr" trans_review qwen25-0.5b lora "$lr"
    done
fi

finish_report
