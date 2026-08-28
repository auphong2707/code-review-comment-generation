#!/usr/bin/env bash
#SBATCH --job-name=rcg_campaign_c
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1            # EDIT: plain gpu:1 if the cluster has no typed gres
#SBATCH --time=48:00:00              # P0 ~60 h batched: expect 2 submissions
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
#
# Campaign C (phase 14) — QUEUE POSITION 3. ICL regeneration under prompt v2 (R2.5).
#   P0  66 runs: zero-shot + repeated random few-shot + BM25, 3 models x 2 datasets
#   P2  12 runs: 10-shot at demo seeds 1 and 2   (+4 optional 14B runs, ICL_14B=1)
#
#     sbatch scripts/slurm/30_campaign_c.sbatch                    # P0
#     RUN_P0=0 RUN_P2=1 sbatch scripts/slurm/30_campaign_c.sbatch  # P2
#     DRY=1 bash scripts/slurm/30_campaign_c.sbatch                # print the queue only
#
# Every row of Tables 4/5 is regenerated: KI-014 (the v1 CodeReviewer test turn said
# "Code:" while the demos said "Code changes:") means the published ICL numbers came
# from a prompt the paper never shows. --prompt-version v2 fixes it and forces a new
# filename (..._v2.json), so no legacy ICL artifact can be overwritten (D-023).
# Outputs: assets/results/revision/icl/<dataset>/<model>_icl_<k>ex_<sel>[_s<seed>]_v2.json
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_P0="${RUN_P0:-1}"
RUN_P2="${RUN_P2:-0}"
ICL_14B="${ICL_14B:-0}"
# Legacy default on CUDA is flash_attention_2. If 00_preflight reports flash_attn
# MISSING, set ICL_ATTN=sdpa — it does not change the v2 output filename.
ATTN="${ICL_ATTN:+--attn-implementation $ICL_ATTN}"

# icl_block <model> <dataset> — the 11 P0 runs for one (model, dataset) pair
icl_block() {
    local model="$1" dataset="$2" tag="$3"
    banner "ICL v2 — $model x $dataset (11 runs, batch ${GEN_BS[$model]:-8})"

    # zero-shot: the missing baseline (R2.5) — no demos, so no demo seed
    icl_run "c_p0_${tag}_0shot" "$dataset" "$model" --num_examples 0 $ATTN

    # repeated random few-shot: 3 demo seeds per k -> variance of the ICL rows
    for k in 3 5; do
        for seed in 42 1 2; do
            icl_run "c_p0_${tag}_${k}shot_s${seed}" "$dataset" "$model" \
                --num_examples "$k" --selection_method random --demo-seed "$seed" $ATTN
        done
    done
    icl_run "c_p0_${tag}_10shot_s42" "$dataset" "$model" \
        --num_examples 10 --selection_method random --demo-seed 42 $ATTN

    # BM25 retrieval; exact test-code matches are excluded by default (leakage)
    for k in 3 5 10; do
        icl_run "c_p0_${tag}_${k}shot_bm25" "$dataset" "$model" \
            --num_examples "$k" --selection_method bm25 $ATTN
    done
}

# ===========================================================================
# P0 — 66 runs. Order: 0.5B validates the pipeline cheaply, then 7B TR, then 7B CR
# ===========================================================================
if [[ "$RUN_P0" == "1" ]]; then
    icl_block qwen25-0.5b trans_review  qwen05b_tr
    icl_block qwen25-0.5b acra_by_ls_pt qwen05b_cr
    icl_block qwen25      trans_review  qwen25_tr
    icl_block qwen15      trans_review  qwen15_tr
    icl_block qwen25      acra_by_ls_pt qwen25_cr
    icl_block qwen15      acra_by_ls_pt qwen15_cr
fi

# ===========================================================================
# P2 — 10-shot demo-seed replicates (12 runs) + optional 14B ICL (4 runs)
# ===========================================================================
if [[ "$RUN_P2" == "1" ]]; then
    banner "campaign C / P2 — 10-shot demo seeds 1 and 2"
    for row in "qwen25-0.5b trans_review qwen05b_tr" "qwen25-0.5b acra_by_ls_pt qwen05b_cr" \
               "qwen25 trans_review qwen25_tr" "qwen15 trans_review qwen15_tr" \
               "qwen25 acra_by_ls_pt qwen25_cr" "qwen15 acra_by_ls_pt qwen15_cr"; do
        read -r model dataset tag <<<"$row"
        for seed in 1 2; do
            icl_run "c_p2_${tag}_10shot_s${seed}" "$dataset" "$model" \
                --num_examples 10 --selection_method random --demo-seed "$seed" $ATTN
        done
    done

    # Optional 14B ICL context for T-14b (needs phase 07 + the 14B weights cached)
    if [[ "$ICL_14B" == "1" ]]; then
        banner "campaign C / P2 — qwen25-14b zero-shot + BM25 3-shot"
        for dataset in trans_review acra_by_ls_pt; do
            icl_run "c_p2_14b_${dataset}_0shot" "$dataset" qwen25-14b --num_examples 0 $ATTN
            icl_run "c_p2_14b_${dataset}_3shot_bm25" "$dataset" qwen25-14b \
                --num_examples 3 --selection_method bm25 $ATTN
        done
    fi
fi

finish_report
