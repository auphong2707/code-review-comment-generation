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
# Submit from repository root, selecting the site's A5000 partition with sbatch.
# Manual resubmission of this SAME file recovers committed progress. No automatic requeue.
RCG_PHASE=15
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# Only 14B QLoRA is hardware-admissible; never relabel it as ordinary LoRA.
if [[ "${RUN_14B_LORA:-0}" == 1 || "${RUN_P2:-0}" == 1 ]]; then
    echo "Ordinary 14B BF16 LoRA cannot reside on 24 GB; no CPU-offload training path is qualified." >&2
    exit 2
fi
for dataset in trans_review acra_by_ls_pt; do
    subset=0.2; [[ "$dataset" == trans_review ]] || subset=0.05
    train_sweep "d_${dataset}_qlora_sweep" "$dataset" qwen25-14b qlora 0 2 "$subset"
    lr=5e-05
    if selected "d_${dataset}_qlora_official"; then lr=$(best_sweep_lr "$REVISION_ROOT/$dataset/sweeps" qwen25_14b_qlora); fi
    extra=(); [[ "$dataset" == trans_review ]] || extra=(--eval-max-samples "${CR_EVAL_MAX:-2000}")
    train_official "d_${dataset}_qlora_official" "$dataset" qwen25-14b qlora "$lr" "${extra[@]}"
done
echo "UNAVAILABLE: ordinary 14B LoRA TR/CR sweeps and officials; report the missing evidence explicitly."

finish_report
