#!/usr/bin/env bash
# Shared server runtime. Plain DRY=1 previews need only Bash (RCG_JSON=0).
set -euo pipefail
ROOT_DIR="${RCG_ROOT:-${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$ROOT_DIR"
[[ -f src/train.py ]] || { echo "Submit from repository root or set RCG_ROOT" >&2; exit 2; }
DRY="${DRY:-0}"
PYTHON="${RCG_PYTHON:-python}"
export RCG_TARGET_GPU="${RCG_TARGET_GPU:-a100_80gb}"
REVISION_ROOT="${RCG_RESULTS_ROOT:-$ROOT_DIR/assets/results/revision-a100-native}"
PREFLIGHT_ROOT="${RCG_PREFLIGHT_ROOT:-$ROOT_DIR/assets/results/a100-native-preflight}"
# A100 profile generation batches: training validation/test generation (GEN_BS), ICL
# (ICL_BS, 2,048-token prompts) and judges (JUDGE_BS rows per call), sized from the
# measured pilot peaks. Smaller for CodeQwen1.5 (full multi-head KV cache). A run's
# batch is part of its identity: never change it for a model whose runs have started.
declare -A GEN_BS=([codet5]=32 [plbart]=32 [qwen25-0.5b]=64 [qwen25]=16 [qwen15]=8 [qwen25-14b]=16)
declare -A ICL_BS=([qwen25-0.5b]=64 [qwen25]=16 [qwen15]=8)
JUDGE_BS=16
CR_SUBSET="paper/generated/cr_judge_subset.json"
JOB_TAG="${RCG_PHASE:-${PBS_JOBNAME:-${SLURM_JOB_NAME:-local}}}"
LOG_ROOT="${RCG_LOG_ROOT:-$ROOT_DIR/assets/logs/a100}/$JOB_TAG"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-offline}" TOKENIZERS_PARALLELISM=false
STEPS_OK=(); STEPS_FAILED=(); STEPS_SKIPPED=()
STOP_REQUESTED=0
ACTIVE_PID=""
CHUNK_STEPS=0
if [[ -n "${RCG_PBS_STEP:-}" ]]; then
    [[ -z "${ONLY:-}${SKIP:-}" ]] || { echo "Do not combine a PBS chunk with ONLY/SKIP" >&2; exit 2; }
fi
if [[ "$DRY" != 1 ]]; then
    [[ -n "${PBS_JOBID:-}${SLURM_JOB_ID:-}" ]] || { echo "Run real jobs via PBS or Slurm on the server; use DRY=1 locally." >&2; exit 2; }
    if [[ -n "${RCG_CONDA_ENV:-}" ]]; then
        source "${CONDA_SH:?Set CONDA_SH when selecting RCG_CONDA_ENV}"
        set +u; conda activate "$RCG_CONDA_ENV"; set -u
    fi
    "$PYTHON" -c 'from tools.a100_preflight import require_gpu; require_gpu()'
    export WANDB_DIR="${WANDB_DIR:-$LOG_ROOT/wandb}"
    export HF_HOME="${HF_HOME:-$ROOT_DIR/assets/cache/huggingface}"
    mkdir -p "$LOG_ROOT" "$WANDB_DIR"
fi

banner() { echo "$*"; }
request_stop() {
    STOP_REQUESTED=1
    if [[ -n "$ACTIVE_PID" ]]; then
        kill -USR1 -- "-$ACTIVE_PID" 2>/dev/null || true
    fi
}
trap request_stop USR1 TERM

selected() {
    local label="$1"
    [[ -z "${RCG_PBS_STEP:-}" || "$label" == "$RCG_PBS_STEP" ]] || return 1
    [[ -z "${ONLY:-}" || "$label" =~ ${ONLY} ]] && [[ -z "${SKIP:-}" || ! "$label" =~ ${SKIP} ]]
}

run_step() {
    local label="$1"; shift
    if [[ -n "${RCG_PBS_STEP:-}" && "$label" != "$RCG_PBS_STEP" ]]; then
        return 0
    fi
    CHUNK_STEPS=$((CHUNK_STEPS + 1))
    if [[ -n "${ONLY:-}" && ! "$label" =~ ${ONLY} ]] || [[ -n "${SKIP:-}" && "$label" =~ ${SKIP} ]]; then
        STEPS_SKIPPED+=("$label"); echo "SKIP $label"; return 0
    fi
    [[ "$STOP_REQUESTED" == 0 ]] || return 75
    printf 'STEP %s:' "$label"; printf ' %q' "$@"; printf '\n'
    if [[ "$DRY" == 1 ]]; then
        if [[ "${RCG_JSON:-0}" == 1 ]]; then
            "$PYTHON" -c 'import json,sys; print(json.dumps({"label":sys.argv[1],"argv":sys.argv[2:]}))' "$label" "$@"
        fi
        return 0
    fi
    local rc=0
    setsid "$@" > >(tee -a "$LOG_ROOT/$label.log") 2>&1 &
    ACTIVE_PID=$!
    # A trap interrupts wait; keep observing the same child until it is terminal.
    while true; do
        if wait "$ACTIVE_PID"; then rc=0; break; else rc=$?; fi
        kill -0 "$ACTIVE_PID" 2>/dev/null || break
    done
    ACTIVE_PID=""
    if [[ "$STOP_REQUESTED" == 1 || "$rc" == 75 ]]; then
        echo "INCOMPLETE $label; resubmit this same phase to resume" >&2
        exit 75
    fi
    if [[ "$rc" != 0 ]]; then
        STEPS_FAILED+=("$label ($rc)")
        echo "FAILED $label (exit $rc); dependent steps were not started" >&2
        exit "$rc"
    fi
    STEPS_OK+=("$label")
}

require_training_admission() {
    local dataset="$1" model="$2" method="$3"
    if [[ "$DRY" != 1 ]]; then
        "$PYTHON" tools/a100_preflight.py --model "$model" --method "$method" \
            --dataset "$dataset" --gen-batch-size "${GEN_BS[$model]}" --out-root "$PREFLIGHT_ROOT" --check
    fi
}

train_official() {
    local label="$1" dataset="$2" model="$3" method="$4" lr="$5"; shift 5
    if selected "$label"; then require_training_admission "$dataset" "$model" "$method"; fi
    run_step "$label" "$PYTHON" -m src.train --model "$model" --method "$method" \
        --learning_rate "$lr" --dataset "$dataset" --results-root "$REVISION_ROOT/$dataset" \
        --protocol archived --memory-profile a100 --gen-batch-size "${GEN_BS[$model]}" \
        --attn-implementation sdpa --recovery-steps "${RCG_RECOVERY_STEPS:-100}" --skip-if-complete "$@"
}

train_sweep() {
    local label="$1" dataset="$2" model="$3" method="$4" first="$5" last="$6" subset="$7"; shift 7
    local id
    for ((id=first;id<=last;id++)); do
        if selected "${label}_lr${id}"; then require_training_admission "$dataset" "$model" "$method"; fi
        run_step "${label}_lr${id}" "$PYTHON" -m src.train --model "$model" --method "$method" \
            --experiment_id "$id" --subset "$subset" --dataset "$dataset" \
            --results-root "$REVISION_ROOT/$dataset/sweeps" --protocol archived \
            --memory-profile a100 --gen-batch-size "${GEN_BS[$model]}" --attn-implementation sdpa \
            --recovery-steps "${RCG_RECOVERY_STEPS:-100}" --skip-if-complete "$@"
    done
}

best_sweep_lr() {
    if [[ "$DRY" == 1 ]]; then echo 5e-05; return; fi
    "$PYTHON" tools/select_sweep_lr.py --sweep-root "$1" --stem "$2" --quiet --require-complete
}
require_var() {
    [[ -n "${!1:-}" ]] || { echo "Missing required $1: $2" >&2; exit 2; }
}

icl_run() {
    local label="$1" dataset="$2" model="$3"; shift 3
    if [[ "$DRY" != 1 ]] && selected "$label"; then
        "$PYTHON" tools/a100_inference_probe.py --kind icl --model "$model" \
            --batch-size "${ICL_BS[$model]}" --out-root "$PREFLIGHT_ROOT" --check
    fi
    # CodeReviewer ICL runs on the fixed 2,000-row subset the Llama judge also uses.
    local subset=(); [[ "$dataset" != acra_by_ls_pt ]] || subset=(--sample-index-file "$CR_SUBSET")
    run_step "$label" "$PYTHON" -m src.icl_inference --model_name "$model" --dataset "$dataset" \
        --prompt-version v2 --memory-profile a100 --batch-size "${ICL_BS[$model]}" --max-prompt-tokens 2048 \
        --attn-implementation sdpa --results_dir "$REVISION_ROOT/icl/$dataset" "${subset[@]}" \
        --skip-if-complete "$@"
}

finish_report() {
    [[ "$STOP_REQUESTED" == 0 ]] || exit 75
    if [[ -n "${RCG_PBS_STEP:-}" ]]; then
        [[ "$CHUNK_STEPS" == 1 ]] || { echo "PBS chunk must select exactly one step: $RCG_PBS_STEP" >&2; return 2; }
        echo "Chunk complete: $RCG_PBS_STEP (whole-campaign gate is separate)"
    fi
    printf 'Phase %s: %s successful, %s filtered, %s failed\n' "$JOB_TAG" "${#STEPS_OK[@]}" "${#STEPS_SKIPPED[@]}" "${#STEPS_FAILED[@]}"
    if [[ ${#STEPS_SKIPPED[@]} -gt 0 ]]; then
        echo "Filtered execution is not a complete campaign; resubmit without ONLY/SKIP for its gate."
        [[ "$DRY" == 1 ]] || return 75
    fi
}
