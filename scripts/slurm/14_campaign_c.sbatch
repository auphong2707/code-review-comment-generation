#!/usr/bin/env bash
#SBATCH --job-name=rcg_phase_14
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
RCG_PHASE=14
source "${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
RUN_P0="${RUN_P0:-1}"; RUN_P2="${RUN_P2:-0}"
if [[ "${ICL_14B:-0}" == 1 ]]; then
    echo "14B ICL is not admitted by this profile; requires a separate quantized-inference qualification." >&2
    exit 2
fi
for pair in "qwen25-0.5b trans_review" "qwen25-0.5b acra_by_ls_pt" \
            "qwen25 trans_review" "qwen15 trans_review" "qwen25 acra_by_ls_pt" "qwen15 acra_by_ls_pt"; do
    read -r model dataset <<<"$pair"
    tag="${model}_${dataset}"
    if [[ "$RUN_P0" == 1 ]]; then
        icl_run "c_${tag}_0shot" "$dataset" "$model" --num_examples 0
        for k in 3 5; do
            for seed in 42 1 2; do
                icl_run "c_${tag}_${k}shot_s${seed}" "$dataset" "$model" \
                    --num_examples "$k" --selection_method random --demo-seed "$seed"
            done
        done
        icl_run "c_${tag}_10shot_s42" "$dataset" "$model" \
            --num_examples 10 --selection_method random --demo-seed 42
        for k in 3 5 10; do
            icl_run "c_${tag}_${k}shot_bm25" "$dataset" "$model" --num_examples "$k" --selection_method bm25
        done
    fi
    if [[ "$RUN_P2" == 1 ]]; then
        for seed in 1 2; do
            icl_run "c_${tag}_10shot_s${seed}" "$dataset" "$model" \
                --num_examples 10 --selection_method random --demo-seed "$seed"
        done
    fi
done

finish_report
