#!/usr/bin/env bash
#SBATCH --job-name=rcg_campaign_d
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                   # 14B: higher host RAM for checkpoint save/load
#SBATCH --gres=gpu:h100:1            # EDIT: plain gpu:1 if the cluster has no typed gres
#SBATCH --time=72:00:00              # P1 ~80 h: expect 2 submissions (runs resume/skip)
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
#
# Campaign D (phase 15) — QUEUE POSITION 5. Does PEFT still hold at 14B? (R2.3)
#   P1  QLoRA both datasets + LoRA TransReview: 3-point brackets then officials (~80 h)
#   P2  LoRA CodeReviewer: bracket + official                                   (~40 h)
#
#     sbatch scripts/slurm/50_campaign_d.sbatch                    # P1
#     RUN_P1=0 RUN_P2=1 sbatch scripts/slurm/50_campaign_d.sbatch  # P2
#     RUN_PROBES=1 sbatch scripts/slurm/50_campaign_d.sbatch       # re-run the memory probes
#     DRY=1 bash scripts/slurm/50_campaign_d.sbatch                # print the queue only
#
# Official LRs come from the brackets automatically (tools/select_sweep_lr.py);
# override with LR_14B_QLORA_TR / LR_14B_QLORA_CR / LR_14B_LORA_TR / LR_14B_LORA_CR.
# CodeReviewer officials cap per-epoch validation generation at 2,000 samples
# (recorded in resources.eval_max_samples and footnoted in T-14b) — the only place
# in the whole queue where that knob is allowed.
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WANDB_TAGS="${WANDB_TAGS:-campaignD}"
RUN_P1="${RUN_P1:-1}"
RUN_P2="${RUN_P2:-0}"
RUN_PROBES="${RUN_PROBES:-0}"
CR_EVAL_CAP="--eval-max-samples ${CR_EVAL_MAX:-2000}"

TR_SWEEPS="$REVISION_ROOT/trans_review/sweeps"
CR_SWEEPS="$REVISION_ROOT/acra_by_ls_pt/sweeps"

# --- memory probes (normally already done by 00_preflight.sbatch) ----------
if [[ "$RUN_PROBES" == "1" ]]; then
    banner "campaign D — 14B memory probes"
    for method in qlora lora; do
        for dataset in trans_review acra_by_ls_pt; do
            run_step "d_probe_${method}_${dataset}" python -m src.train \
                --model qwen25-14b --method "$method" --probe \
                --protocol archived --dataset "$dataset" --learning_rate 5e-05
        done
    done
fi

# ===========================================================================
# P1 — QLoRA both datasets + LoRA TransReview
# ===========================================================================
if [[ "$RUN_P1" == "1" ]]; then
    banner "campaign D / P1 — 3-point LR brackets {1e-05, 5e-05, 1e-04}"
    train_sweep d_p1_qlora_sweep_tr trans_review  qwen25-14b qlora 0 2 0.2
    train_sweep d_p1_qlora_sweep_cr acra_by_ls_pt qwen25-14b qlora 0 2 0.05
    train_sweep d_p1_lora_sweep_tr  trans_review  qwen25-14b lora  0 2 0.2

    LR_14B_QLORA_TR="${LR_14B_QLORA_TR:-$(best_sweep_lr "$TR_SWEEPS" qwen25_14b_qlora)}"
    LR_14B_QLORA_CR="${LR_14B_QLORA_CR:-$(best_sweep_lr "$CR_SWEEPS" qwen25_14b_qlora)}"
    LR_14B_LORA_TR="${LR_14B_LORA_TR:-$(best_sweep_lr "$TR_SWEEPS" qwen25_14b_lora)}"
    echo "selected 14B LRs:  qlora TR=${LR_14B_QLORA_TR:-<none>}" \
         " qlora CR=${LR_14B_QLORA_CR:-<none>}  lora TR=${LR_14B_LORA_TR:-<none>}"

    banner "campaign D / P1 — officials"
    if require_var LR_14B_QLORA_TR "Finish the TR QLoRA bracket first."; then
        train_official d_p1_qlora_official_tr trans_review qwen25-14b qlora "$LR_14B_QLORA_TR"
    fi
    if require_var LR_14B_QLORA_CR "Finish the CR QLoRA bracket first."; then
        train_official d_p1_qlora_official_cr acra_by_ls_pt qwen25-14b qlora \
            "$LR_14B_QLORA_CR" $CR_EVAL_CAP
    fi
    if require_var LR_14B_LORA_TR "Finish the TR LoRA bracket first."; then
        train_official d_p1_lora_official_tr trans_review qwen25-14b lora "$LR_14B_LORA_TR"
    fi
fi

# ===========================================================================
# P2 — LoRA CodeReviewer (completeness)
# ===========================================================================
if [[ "$RUN_P2" == "1" ]]; then
    banner "campaign D / P2 — 14B LoRA x CodeReviewer"
    train_sweep d_p2_lora_sweep_cr acra_by_ls_pt qwen25-14b lora 0 2 0.05

    LR_14B_LORA_CR="${LR_14B_LORA_CR:-$(best_sweep_lr "$CR_SWEEPS" qwen25_14b_lora)}"
    if require_var LR_14B_LORA_CR "Finish the CR LoRA bracket first."; then
        train_official d_p2_lora_official_cr acra_by_ls_pt qwen25-14b lora \
            "$LR_14B_LORA_CR" $CR_EVAL_CAP
    fi
fi

finish_report
