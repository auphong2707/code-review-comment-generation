#!/usr/bin/env bash
#SBATCH --job-name=rcg_campaign_b
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1            # EDIT: plain gpu:1 if the cluster has no typed gres
#SBATCH --time=72:00:00              # P0 ~60 h; resubmit to continue (every run is idempotent)
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
#
# Campaign B (phase 13) — QUEUE POSITION 1, claim-critical.
#   P0  GaLore full-parameter Qwen2.5-Coder-7B: 3-point LR bracket + officials  (~60 h)
#   P1  DoRA at the LoRA-selected LR: small models both datasets + 7B TR         (~30 h)
#   P2  DoRA 7B x CodeReviewer, GaLore TR seed replicates, qwen15 GaLore TR      (~65 h)
#
#     sbatch scripts/slurm/10_campaign_b.sbatch                    # P0
#     RUN_P0=0 RUN_P1=1 sbatch scripts/slurm/10_campaign_b.sbatch  # P1
#     RUN_P0=0 RUN_P2=1 sbatch scripts/slurm/10_campaign_b.sbatch  # P2
#     DRY=1 bash scripts/slurm/10_campaign_b.sbatch                # print the queue only
#
# The officials need the LR the sweep selected. It is read automatically from the
# sweep runs' best validation BLEU-4 (tools/select_sweep_lr.py); override with
#     GALORE_LR_TR=5e-05 GALORE_LR_CR=1e-05 sbatch ...
# If the preflight probe reported > ~75 GB peak, also pass
#     GALORE_VARIANT=galore_adamw_8bit_layerwise
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WANDB_TAGS="${WANDB_TAGS:-campaignB}"
RUN_P0="${RUN_P0:-1}"
RUN_P1="${RUN_P1:-0}"
RUN_P2="${RUN_P2:-0}"
GALORE_EXTRA="${GALORE_VARIANT:+--galore-variant $GALORE_VARIANT}"

# ===========================================================================
# P0 — GaLore: the scale-matched 7B full-parameter baseline (R2.1, R2.4)
# ===========================================================================
if [[ "$RUN_P0" == "1" ]]; then
    banner "campaign B / P0 — GaLore qwen25 (3-point bracket, then officials)"

    # 3-point LR bracket {1e-05, 5e-05, 1e-04} on the published prefix subsets (D-028d)
    train_sweep b_p0_galore_sweep_tr trans_review  qwen25 galore 0 2 0.2  $GALORE_EXTRA
    train_sweep b_p0_galore_sweep_cr acra_by_ls_pt qwen25 galore 0 2 0.05 $GALORE_EXTRA

    GALORE_LR_TR="${GALORE_LR_TR:-$(best_sweep_lr "$REVISION_ROOT/trans_review/sweeps"  qwen25_galore)}"
    GALORE_LR_CR="${GALORE_LR_CR:-$(best_sweep_lr "$REVISION_ROOT/acra_by_ls_pt/sweeps" qwen25_galore)}"
    echo "selected GaLore LR:  TR=${GALORE_LR_TR:-<none>}  CR=${GALORE_LR_CR:-<none>}"

    if require_var GALORE_LR_TR "Finish the TR bracket, or pass GALORE_LR_TR=<lr>."; then
        train_official b_p0_galore_official_tr trans_review qwen25 galore "$GALORE_LR_TR" $GALORE_EXTRA
    fi
    if require_var GALORE_LR_CR "Finish the CR bracket, or pass GALORE_LR_CR=<lr>."; then
        train_official b_p0_galore_official_cr acra_by_ls_pt qwen25 galore "$GALORE_LR_CR" $GALORE_EXTRA
    fi
fi

# ===========================================================================
# P1 — DoRA at the LoRA-selected LR (8 runs, D-024; no sweep of its own)
# ===========================================================================
if [[ "$RUN_P1" == "1" ]]; then
    banner "campaign B / P1 — DoRA (small models both datasets, then 7B TransReview)"

    # "<label suffix> <dataset> <model> <lr>"; order = cheap validation first
    DORA_RUNS=(
        "tr_codet5      trans_review  codet5       0.001"
        "tr_plbart      trans_review  plbart       1e-05"
        "tr_qwen05b     trans_review  qwen25-0.5b  0.0005"
        "cr_codet5      acra_by_ls_pt codet5       0.001"
        "cr_plbart      acra_by_ls_pt plbart       0.001"
        "cr_qwen05b     acra_by_ls_pt qwen25-0.5b  5e-05"
        "tr_qwen25      trans_review  qwen25       5e-05"
        "tr_qwen15      trans_review  qwen15       5e-05"
    )
    for row in "${DORA_RUNS[@]}"; do
        read -r suffix dataset model lr <<<"$row"
        train_official "b_p1_dora_$suffix" "$dataset" "$model" dora "$lr"
    done
fi

# ===========================================================================
# P2 — completeness: DoRA 7B x CR, GaLore seed replicates, qwen15 GaLore
# ===========================================================================
if [[ "$RUN_P2" == "1" ]]; then
    banner "campaign B / P2 — DoRA 7B x CodeReviewer, GaLore seeds, qwen15 GaLore"

    train_official b_p2_dora_cr_qwen25 acra_by_ls_pt qwen25 dora 1e-05
    train_official b_p2_dora_cr_qwen15 acra_by_ls_pt qwen15 dora 0.0005

    # GaLore TransReview seed replicates (R1.1 for the full-FT baseline)
    GALORE_LR_TR="${GALORE_LR_TR:-$(best_sweep_lr "$REVISION_ROOT/trans_review/sweeps" qwen25_galore)}"
    if require_var GALORE_LR_TR "Run P0 first, or pass GALORE_LR_TR=<lr>."; then
        for seed in 1 2; do
            train_official "b_p2_galore_tr_seed$seed" trans_review qwen25 galore \
                "$GALORE_LR_TR" --seed "$seed" $GALORE_EXTRA
        done
    fi

    # qwen15 GaLore TransReview: own bracket + official
    train_sweep b_p2_galore_qwen15_sweep_tr trans_review qwen15 galore 0 2 0.2 $GALORE_EXTRA
    GALORE_LR_TR_QWEN15="${GALORE_LR_TR_QWEN15:-$(best_sweep_lr "$REVISION_ROOT/trans_review/sweeps" qwen15_galore)}"
    if require_var GALORE_LR_TR_QWEN15 "Finish the qwen15 bracket first."; then
        train_official b_p2_galore_qwen15_official_tr trans_review qwen15 galore \
            "$GALORE_LR_TR_QWEN15" $GALORE_EXTRA
    fi
fi

finish_report
