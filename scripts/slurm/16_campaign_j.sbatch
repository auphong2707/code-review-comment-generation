#!/usr/bin/env bash
#SBATCH --job-name=rcg_phase_16
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
RCG_PHASE=16
source "${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/slurm/common.sh"
RUN_REVISION="${RUN_REVISION:-1}"; RUN_LEGACY="${RUN_LEGACY:-1}"
RUN_QWEN="${RUN_QWEN:-1}"; RUN_LLAMA="${RUN_LLAMA:-1}"
JUDGE_ROOT="${RCG_JUDGE_ROOT:-$ROOT_DIR/assets/results/a100-judges}"
JOBS_FILE="$JUDGE_ROOT/jobs_revision.json"
SUBSET=paper/generated/cr_judge_subset.json
TOKEN_LIMIT="${JUDGE_MAX_INPUT_TOKENS:-4096}"
QWEN_ARGS=(--judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
            --judge-local-files-only --judge-device cuda --judge-k 3 --judge-batch-size 1
            --judge-input-token-limit "$TOKEN_LIMIT")
LLAMA_ARGS=(--backend hf-chat --judge-model-path assets/models/llm-judge/Llama-3.1-8B-Instruct
             --judge-local-files-only --judge-device cuda --column-prefix llama_judge_
             --judge-k 3 --judge-batch-size 1 --judge-input-token-limit "$TOKEN_LIMIT"
             --judge-top-k 20 --judge-repetition-penalty 1.05 --judge-seed 42)
if [[ "$RUN_REVISION" == 1 ]]; then
    # Explicitly selecting fewer prerequisites means a partial judge pass.
    for phase in ${J_REQUIRE_PHASES:-A B C D}; do
        run_step "j_require_$phase" "$PYTHON" tools/campaign_status.py "$phase" --results-root "$REVISION_ROOT"
    done
    run_step j_build_jobs "$PYTHON" tools/revision_judge_jobs.py --results-root "$REVISION_ROOT" --out "$JOBS_FILE"
fi
if [[ "$RUN_QWEN" == 1 && "$RUN_REVISION" == 1 ]]; then
    run_step j_qualify_qwen "$PYTHON" tools/a5000_inference_probe.py --kind judge --model qwen-judge \
        --max-input-tokens "$TOKEN_LIMIT" --out-root "$PREFLIGHT_ROOT" --check
    run_step j_qwen_revision "$PYTHON" tools/update_llm_judge_tables.py \
        --jobs-file "$JOBS_FILE" --out-root "$JUDGE_ROOT/qwen-revision" "${QWEN_ARGS[@]}"
fi
if [[ "$RUN_LLAMA" == 1 ]]; then
    run_step j_qualify_llama "$PYTHON" tools/a5000_inference_probe.py --kind judge --model llama-judge \
        --max-input-tokens "$TOKEN_LIMIT" --out-root "$PREFLIGHT_ROOT" --check
    if [[ "$RUN_LEGACY" == 1 ]]; then
        run_step j_llama_legacy "$PYTHON" tools/update_llm_judge_tables.py \
            --reuse-inputs assets/models/llm-as-judge/table-results/inputs --sample-index-file "$SUBSET" \
            --out-root "$JUDGE_ROOT/llama-legacy" "${LLAMA_ARGS[@]}"
    fi
    if [[ "$RUN_REVISION" == 1 ]]; then
        run_step j_llama_revision "$PYTHON" tools/update_llm_judge_tables.py \
            --jobs-file "$JOBS_FILE" --sample-index-file "$SUBSET" \
            --out-root "$JUDGE_ROOT/llama-revision" "${LLAMA_ARGS[@]}"
    fi
fi

finish_report
