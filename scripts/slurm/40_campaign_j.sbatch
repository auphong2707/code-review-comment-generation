#!/usr/bin/env bash
#SBATCH --job-name=rcg_campaign_j
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1            # EDIT: plain gpu:1 if the cluster has no typed gres
#SBATCH --time=48:00:00              # P0 ~40 h; cached keys make re-submission cheap
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
#
# Campaign J (phase 16) — QUEUE POSITION 4, re-run after EVERY campaign block.
#   1. rebuild the jobs file from whatever revision artifacts exist right now
#   2. Qwen2.5-Coder-3B judge (frozen protocol, k=3) over those artifacts   R1.2
#   3. Llama-3.1-8B judge over the legacy 54 jobs and over the same artifacts  R1.3
#
#     sbatch scripts/slurm/40_campaign_j.sbatch                       # everything
#     RUN_LEGACY=0 sbatch scripts/slurm/40_campaign_j.sbatch          # revision rows only
#     RUN_LLAMA=0  sbatch scripts/slurm/40_campaign_j.sbatch          # Qwen judge only
#     DRY=1 bash scripts/slurm/40_campaign_j.sbatch                   # print the queue only
#
# Re-running is cheap and safe: every (prediction, reference, config) pair is cached
# (make_cache_key), so already-judged rows are replayed, not regenerated. The Qwen
# cache is seeded from the frozen legacy cache by 00_preflight.sbatch.
#
# CodeReviewer is judged on the committed 2,000-sample stratified subset
# (paper/generated/cr_judge_subset.json); TransReview is judged in full.
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN_QWEN="${RUN_QWEN:-1}"
RUN_LLAMA="${RUN_LLAMA:-1}"
RUN_LEGACY="${RUN_LEGACY:-1}"        # the Llama pass over the 54 frozen-table jobs

JUDGE_ROOT=assets/models/llm-as-judge
JOBS_FILE="$JUDGE_ROOT/jobs_revision.json"
SUBSET=paper/generated/cr_judge_subset.json
# One out-root per (judge, artifact generation): each keeps its own manifest,
# summary and cache. Phase 16 names a single Llama root; splitting legacy from
# revision stops the second pass from overwriting the first pass's manifest.
QWEN_OUT="$JUDGE_ROOT/table-results-revision"
LLAMA_LEGACY_OUT="$JUDGE_ROOT/table-results-llama31-legacy"
LLAMA_REVISION_OUT="$JUDGE_ROOT/table-results-llama31-revision"

QWEN_ARGS=(--judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
           --judge-local-files-only --judge-device cuda
           --judge-k 3 --judge-batch-size 32 --log-level INFO)
LLAMA_ARGS=(--backend hf-chat
            --judge-model-path assets/models/llm-judge/Llama-3.1-8B-Instruct
            --judge-local-files-only --judge-device cuda
            --column-prefix llama_judge_
            --judge-k 3 --judge-top-k 20 --judge-repetition-penalty 1.05 --judge-seed 42
            --judge-batch-size 16 --log-level INFO)

# ===========================================================================
# 1. jobs file — every revision artifact that exists, nothing that does not
# ===========================================================================
banner "campaign J / 1 — build the jobs file from $REVISION_ROOT"
run_step "j_build_jobs" python tools/revision_judge_jobs.py \
    --results-root "$REVISION_ROOT" --out "$JOBS_FILE"

HAVE_JOBS=0
[[ -s "$JOBS_FILE" || "${DRY:-0}" == "1" ]] && HAVE_JOBS=1
[[ "$HAVE_JOBS" == "1" ]] || echo "no revision artifacts yet — only the legacy Llama pass can run"

# ===========================================================================
# 2. Qwen judge (frozen protocol) on the revision artifacts
# ===========================================================================
if [[ "$RUN_QWEN" == "1" && "$HAVE_JOBS" == "1" ]]; then
    # NO --sample-index-file here: the frozen Qwen tables were scored on every row,
    # so the revision rows must be too or the two are not comparable. The subset
    # exists only to keep the *second* judge affordable.
    banner "campaign J / 2 — Qwen2.5-Coder-3B judge, k=3, all rows"
    run_step "j_qwen_revision" python tools/update_llm_judge_tables.py \
        --jobs-file "$JOBS_FILE" --out-root "$QWEN_OUT" "${QWEN_ARGS[@]}"
    run_step "j_qwen_summary" python tools/update_llm_judge_tables.py \
        --jobs-file "$JOBS_FILE" --out-root "$QWEN_OUT" --summary-only
fi

# ===========================================================================
# 3. Llama-3.1-8B judge — a second, non-Qwen judge over identical inputs (D-019)
# ===========================================================================
if [[ "$RUN_LLAMA" == "1" ]]; then
    if [[ ! -d assets/models/llm-judge/Llama-3.1-8B-Instruct && "${DRY:-0}" != "1" ]]; then
        echo "Llama-3.1-8B-Instruct missing — run RCG_DOWNLOAD=1 sbatch scripts/slurm/00_preflight.sbatch"
        STEPS_SKIPPED+=("j_llama_* (model missing)")
    else
        # 3a. the 54 frozen-table jobs, scored from the inputs the Qwen judge used
        #     verbatim — same rows, same order, different judge.
        if [[ "$RUN_LEGACY" == "1" ]]; then
            banner "campaign J / 3a — Llama judge on the 54 legacy jobs"
            run_step "j_llama_legacy" python tools/update_llm_judge_tables.py \
                --reuse-inputs "$JUDGE_ROOT/table-results/inputs" \
                --sample-index-file "$SUBSET" \
                --out-root "$LLAMA_LEGACY_OUT" "${LLAMA_ARGS[@]}"
            run_step "j_llama_legacy_summary" python tools/update_llm_judge_tables.py \
                --reuse-inputs "$JUDGE_ROOT/table-results/inputs" \
                --sample-index-file "$SUBSET" \
                --out-root "$LLAMA_LEGACY_OUT" --column-prefix llama_judge_ --summary-only
        fi

        # 3b. the revision artifacts, same subset rule
        if [[ "$HAVE_JOBS" == "1" ]]; then
            banner "campaign J / 3b — Llama judge on the revision artifacts"
            run_step "j_llama_revision" python tools/update_llm_judge_tables.py \
                --jobs-file "$JOBS_FILE" --sample-index-file "$SUBSET" \
                --out-root "$LLAMA_REVISION_OUT" "${LLAMA_ARGS[@]}"
            run_step "j_llama_revision_summary" python tools/update_llm_judge_tables.py \
                --jobs-file "$JOBS_FILE" --sample-index-file "$SUBSET" \
                --out-root "$LLAMA_REVISION_OUT" --column-prefix llama_judge_ --summary-only
        fi
    fi
fi

finish_report
