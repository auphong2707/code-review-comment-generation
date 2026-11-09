#!/usr/bin/env bash
# Producer gates and bounded judge shards, called only through run_chunk.sh.
RCG_PHASE="chunk_${RCG_PBS_CHUNK:?}"
source "$RCG_ROOT/scripts/slurm/common.sh"
JUDGE_ROOT="${RCG_JUDGE_ROOT:-$ROOT_DIR/assets/results/a100-native-judges}"
JOBS_FILE="$JUDGE_ROOT/jobs_revision.json"
TOKEN_LIMIT="${JUDGE_MAX_INPUT_TOKENS:-4096}"
case "$RCG_CHUNK_KIND" in
    gate)
        run_step "gate_$RCG_CHUNK_PHASE" "$PYTHON" tools/campaign_status.py "$RCG_CHUNK_PHASE" --results-root "$REVISION_ROOT"
        ;;
    prepare)
        for phase in A B C D; do
            run_step "j_require_$phase" "$PYTHON" tools/campaign_status.py "$phase" --results-root "$REVISION_ROOT"
        done
        run_step j_build_jobs "$PYTHON" tools/revision_judge_jobs.py --results-root "$REVISION_ROOT" --out "$JOBS_FILE"
        for model in qwen-judge llama-judge; do
            run_step "j_qualify_$model" "$PYTHON" tools/a100_inference_probe.py --kind judge --model "$model" \
                --max-input-tokens "$TOKEN_LIMIT" --batch-size "$JUDGE_BS" --out-root "$PREFLIGHT_ROOT" --check
        done
        ;;
    judge|summary)
        # Every pass scores each row once (k=1, including the legacy Llama pass) in batches
        # of JUDGE_BS rows. Neither k nor the batch size is part of the cache key.
        args=(--out-root "$JUDGE_ROOT/$RCG_CHUNK_PHASE" --judge-local-files-only --judge-device cuda
              --judge-k 1 --judge-batch-size "$JUDGE_BS" --judge-input-token-limit "$TOKEN_LIMIT")
        count=48
        case "$RCG_CHUNK_PHASE" in
            qwen-revision)
                args+=(--jobs-file "$JOBS_FILE" --judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct) ;;
            llama-legacy|llama-revision)
                args+=(--backend hf-chat --column-prefix llama_judge_
                       --judge-model-path assets/models/llm-judge/Llama-3.1-8B-Instruct
                       --sample-index-file paper/generated/cr_judge_subset.json
                       --judge-top-k 20 --judge-repetition-penalty 1.05 --judge-seed 42)
                if [[ "$RCG_CHUNK_PHASE" == llama-legacy ]]; then
                    count=18
                    args+=(--reuse-inputs assets/models/llm-as-judge/table-results/inputs)
                else args+=(--jobs-file "$JOBS_FILE"); fi ;;
            *) echo "Unknown judge pass" >&2; exit 2 ;;
        esac
        if [[ "$RCG_CHUNK_KIND" == summary ]]; then
            args+=(--summary-only --require-complete)
        else
            # Shards may run concurrently; a shared cache JSONL would interleave appends
            # on NFS. Shards cover whole artifacts, so keys never need to cross shards.
            printf -v shard '%03d' "$RCG_CHUNK_ITEM"
            args+=(--job-shard-index "$RCG_CHUNK_ITEM" --job-shard-count "$count"
                   --judge-cache-path "$JUDGE_ROOT/$RCG_CHUNK_PHASE/cache/shard_$shard.jsonl")
        fi
        run_step "${RCG_CHUNK_KIND}_${RCG_CHUNK_PHASE}_${RCG_CHUNK_ITEM}" "$PYTHON" tools/update_llm_judge_tables.py "${args[@]}"
        ;;
    *) echo "Unknown chunk kind" >&2; exit 2 ;;
esac
finish_report
