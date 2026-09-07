#!/usr/bin/env bash
#SBATCH --job-name=rcg_phase_00_07
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
RCG_PHASE=00_07
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# One combined verification job for historical phases 00–07; no destructive cleanup.
run_step infra_pytest "$PYTHON" -m pytest -q
run_step infra_train_help "$PYTHON" -m src.train --help
run_step infra_icl_help "$PYTHON" -m src.icl_inference --help
run_step infra_judge_mock "$PYTHON" tools/update_llm_judge_tables.py --mock \
    --limit-jobs 1 --limit-samples 3 --out-root "$PREFLIGHT_ROOT/mock-tables"
if [[ "${RCG_PROBES:-1}" == 1 ]]; then
    for model in codet5 plbart qwen25-0.5b qwen25 qwen15; do
        methods="ia3 lora qlora dora"
        [[ "$model" == qwen25 || "$model" == qwen15 ]] || methods+=" fmft"
        for method in $methods; do
            run_step "probe_tr_${model}_${method}" "$PYTHON" tools/a5000_preflight.py \
                --model "$model" --method "$method" --dataset trans_review --out-root "$PREFLIGHT_ROOT"
        done
        # Matched A5000 LoRA controls accompany DoRA on CodeReviewer.
        for method in lora dora; do
            run_step "probe_cr_${model}_${method}" "$PYTHON" tools/a5000_preflight.py \
                --model "$model" --method "$method" --dataset acra_by_ls_pt --out-root "$PREFLIGHT_ROOT"
        done
    done
    for dataset in trans_review acra_by_ls_pt; do
        run_step "probe_14b_qlora_$dataset" "$PYTHON" tools/a5000_preflight.py \
            --model qwen25-14b --method qlora --dataset "$dataset" --out-root "$PREFLIGHT_ROOT"
    done
    for model in qwen25-0.5b qwen25 qwen15; do
        run_step "probe_icl_$model" "$PYTHON" tools/a5000_inference_probe.py \
            --kind icl --model "$model" --out-root "$PREFLIGHT_ROOT"
    done
    for model in qwen-judge llama-judge; do
        run_step "probe_$model" "$PYTHON" tools/a5000_inference_probe.py --kind judge \
            --model "$model" --max-input-tokens "${JUDGE_MAX_INPUT_TOKENS:-4096}" --out-root "$PREFLIGHT_ROOT"
    done
else
    echo "Code checks only. GPU campaigns still require individual admission reports."
fi

finish_report
