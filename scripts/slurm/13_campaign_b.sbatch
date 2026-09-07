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
# Submit from repository root, selecting the site's A5000 partition with sbatch.
# Manual resubmission of this SAME file recovers committed progress. No automatic requeue.
RCG_PHASE=13
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# GaLore is unavailable until a separately qualified full-state/offload path exists.
# DoRA always gets a matching A5000 LoRA control at the identical LR and protocol.
RUN_P0="${RUN_P0:-0}"; RUN_P1="${RUN_P1:-1}"; RUN_P2="${RUN_P2:-0}"
if [[ "$RUN_P0" == 1 || "${RUN_GALORE:-0}" == 1 ]]; then
    echo "GaLore is not admitted on 24 GB: layerwise optimizer requires separate accumulation/resume validation." >&2
    exit 2
fi
paired_run() {
    local label="$1" dataset="$2" model="$3" lr="$4"
    train_official "${label}_lora_control" "$dataset" "$model" lora "$lr" --seed 42
    train_official "${label}_dora" "$dataset" "$model" dora "$lr" --seed 42
}
if [[ "$RUN_P1" == 1 ]]; then
    paired_run b_tr_codet5 trans_review codet5 0.001
    paired_run b_tr_plbart trans_review plbart 1e-05
    paired_run b_tr_qwen05b trans_review qwen25-0.5b 0.0005
    paired_run b_cr_codet5 acra_by_ls_pt codet5 0.001
    paired_run b_cr_plbart acra_by_ls_pt plbart 0.001
    paired_run b_cr_qwen05b acra_by_ls_pt qwen25-0.5b 5e-05
    paired_run b_tr_qwen25 trans_review qwen25 5e-05
    paired_run b_tr_qwen15 trans_review qwen15 5e-05
fi
if [[ "$RUN_P2" == 1 ]]; then
    paired_run b_cr_qwen25 acra_by_ls_pt qwen25 1e-05
    paired_run b_cr_qwen15 acra_by_ls_pt qwen15 0.0005
fi
echo "UNAVAILABLE: GaLore sweeps/officials/seeds, including optional qwen15; no full-FT result is claimed."

finish_report
