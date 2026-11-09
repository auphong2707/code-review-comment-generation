#!/usr/bin/env bash
#SBATCH --job-name=rcg_phase_13
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
RCG_PHASE=13
source "${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
# GaLore is unavailable until a separately qualified full-state/offload path exists.
# DoRA runs on TransReview only, at the LR of the paper's LoRA run for the same model;
# that archived LoRA run is its control (paper/generated/seed42_paper_reuse.json).
RUN_P0="${RUN_P0:-0}"; RUN_P1="${RUN_P1:-1}"
if [[ "$RUN_P0" == 1 || "${RUN_GALORE:-0}" == 1 ]]; then
    echo "GaLore is outside this profile: optimizer accumulation/resume requires separate validation." >&2
    exit 2
fi
if [[ "$RUN_P1" == 1 ]]; then
    train_official b_tr_codet5_dora trans_review codet5 dora 0.001
    train_official b_tr_plbart_dora trans_review plbart dora 1e-05
    train_official b_tr_qwen05b_dora trans_review qwen25-0.5b dora 0.0005
    train_official b_tr_qwen25_dora trans_review qwen25 dora 5e-05
    train_official b_tr_qwen15_dora trans_review qwen15 dora 5e-05
fi
echo "UNAVAILABLE: GaLore sweeps/officials/seeds, including optional qwen15; no full-FT result is claimed."

finish_report
