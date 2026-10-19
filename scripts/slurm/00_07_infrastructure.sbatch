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
# Submit from repository root, selecting the site's A100 80GB partition/GRES.
# Manual resubmission of this SAME file recovers committed progress. No automatic requeue.
RCG_PHASE=00_07
source "${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
# First job: software gate, recovery/memory pilots, then real inference integration.
run_step infra_pip_check "$PYTHON" -m pip check
run_step infra_pytest "$PYTHON" -m pytest -q
run_step infra_train_help "$PYTHON" -m src.train --help
run_step infra_icl_help "$PYTHON" -m src.icl_inference --help
run_step infra_judge_mock "$PYTHON" tools/update_llm_judge_tables.py --mock \
    --limit-jobs 1 --limit-samples 3 --out-root "$PREFLIGHT_ROOT/mock-tables"
if [[ "${RCG_PROBES:-1}" == 1 ]]; then
    # Every training configuration the campaigns run (all TransReview): A seeds and B DoRA
    # for the small models, LoRA seeds and DoRA for the 7B models, D 14B QLoRA.
    for model in codet5 plbart qwen25-0.5b qwen25 qwen15; do
        methods="lora dora"
        [[ "$model" == qwen25 || "$model" == qwen15 ]] || methods="ia3 lora qlora fmft dora"
        for method in $methods; do
            run_step "probe_tr_${model}_${method}" "$PYTHON" tools/a100_preflight.py \
                --model "$model" --method "$method" --dataset trans_review \
                --gen-batch-size "${GEN_BS[$model]}" --out-root "$PREFLIGHT_ROOT"
        done
    done
    run_step probe_14b_qlora_trans_review "$PYTHON" tools/a100_preflight.py \
        --model qwen25-14b --method qlora --dataset trans_review \
        --gen-batch-size "${GEN_BS[qwen25-14b]}" --out-root "$PREFLIGHT_ROOT"
    for model in qwen25-0.5b qwen25 qwen15; do
        run_step "probe_icl_$model" "$PYTHON" tools/a100_inference_probe.py \
            --kind icl --model "$model" --batch-size "${ICL_BS[$model]}" --out-root "$PREFLIGHT_ROOT"
    done
    for model in qwen-judge llama-judge; do
        run_step "probe_$model" "$PYTHON" tools/a100_inference_probe.py --kind judge \
            --model "$model" --max-input-tokens "${JUDGE_MAX_INPUT_TOKENS:-4096}" \
            --batch-size "$JUDGE_BS" --out-root "$PREFLIGHT_ROOT"
    done
    # Synthetic pilots alone cannot check dataset parsing, retrieval, metrics or
    # judge input/cache/output plumbing. Exercise those paths on bounded rows.
    for dataset in trans_review acra_by_ls_pt; do
        subset=(); [[ "$dataset" != acra_by_ls_pt ]] || subset=(--sample-index-file "$CR_SUBSET")
        for model in qwen25-0.5b qwen25 qwen15; do
            run_step "smoke_icl_${dataset}_${model}" "$PYTHON" -m src.icl_inference \
                --model_name "$model" --dataset "$dataset" --prompt-version v2 \
                --memory-profile a100 --batch-size "${ICL_BS[$model]}" --max-prompt-tokens 2048 \
                --attn-implementation sdpa --num_examples 3 --selection_method bm25 "${subset[@]}" \
                --max_samples "${ICL_BS[$model]}" --results_dir "$PREFLIGHT_ROOT/icl-smoke/$dataset" --skip-if-complete
        done
    done
    run_step smoke_judge_subset test -r paper/generated/cr_judge_subset.json
    run_step smoke_qwen_pipeline "$PYTHON" tools/update_llm_judge_tables.py \
        --limit-jobs 1 --limit-samples 3 --out-root "$PREFLIGHT_ROOT/qwen-smoke" \
        --judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct \
        --judge-local-files-only --judge-device cuda --judge-k 1 --judge-batch-size "$JUDGE_BS" \
        --judge-input-token-limit "${JUDGE_MAX_INPUT_TOKENS:-4096}"
    run_step smoke_llama_pipeline "$PYTHON" tools/update_llm_judge_tables.py \
        --limit-jobs 1 --limit-samples 3 --out-root "$PREFLIGHT_ROOT/llama-smoke" \
        --reuse-inputs assets/models/llm-as-judge/table-results/inputs \
        --sample-index-file paper/generated/cr_judge_subset.json \
        --backend hf-chat --column-prefix llama_judge_ \
        --judge-model-path assets/models/llm-judge/Llama-3.1-8B-Instruct \
        --judge-local-files-only --judge-device cuda --judge-k 1 --judge-batch-size "$JUDGE_BS" \
        --judge-top-k 20 --judge-repetition-penalty 1.05 --judge-seed 42 \
        --judge-input-token-limit "${JUDGE_MAX_INPUT_TOKENS:-4096}"
else
    echo "Code checks only. GPU campaigns still require individual admission reports."
    [[ "$DRY" == 1 ]] || exit 75
fi

finish_report
