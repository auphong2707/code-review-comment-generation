#!/usr/bin/env bash
#SBATCH --job-name=rcg_phase_15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
# Submit from repository root, selecting the site's A100 80GB partition/GRES.
# Manual resubmission of this SAME file recovers committed progress. No automatic requeue.
RCG_PHASE=15
source "${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
# Retain qualified-method scope: 14B QLoRA only; ordinary LoRA needs a new profile.
if [[ "${RUN_14B_LORA:-0}" == 1 || "${RUN_P2:-0}" == 1 ]]; then
    echo "Ordinary 14B LoRA is not enabled in this conservative profile; qualify it separately on A100." >&2
    exit 2
fi
# 14B trains on TransReview only: three LR sweeps on 20% of the data, then the official
# run at the best sweep LR on the full set.
dataset=trans_review
train_sweep "d_${dataset}_qlora_sweep" "$dataset" qwen25-14b qlora 0 2 0.2
lr=5e-05
if selected "d_${dataset}_qlora_official"; then lr=$(best_sweep_lr "$REVISION_ROOT/$dataset/sweeps" qwen25_14b_qlora); fi
train_official "d_${dataset}_qlora_official" "$dataset" qwen25-14b qlora "$lr"
echo "UNAVAILABLE: ordinary 14B LoRA and all 14B CodeReviewer runs; report the missing evidence explicitly."

finish_report
