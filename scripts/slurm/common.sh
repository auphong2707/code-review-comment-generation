#!/usr/bin/env bash
# Shared A5000 phase runtime. DRY=1 is read-only and needs only Bash/Python.
set -euo pipefail
ROOT_DIR="${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"
[[ -f src/train.py ]] || { echo "Submit from repository root or set RCG_ROOT" >&2; exit 2; }
DRY="${DRY:-0}"
PYTHON="${RCG_PYTHON:-python}"
REVISION_ROOT="${RCG_RESULTS_ROOT:-$ROOT_DIR/assets/results/revision-a5000}"
PREFLIGHT_ROOT="${RCG_PREFLIGHT_ROOT:-$ROOT_DIR/assets/results/a5000-preflight}"
JOB_TAG="${RCG_PHASE:-${SLURM_JOB_NAME:-local}}"
LOG_ROOT="${RCG_LOG_ROOT:-$ROOT_DIR/assets/logs/a5000}/$JOB_TAG"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-offline}" TOKENIZERS_PARALLELISM=false
STEPS_OK=(); STEPS_FAILED=(); STEPS_SKIPPED=()
STOP_REQUESTED=0
ACTIVE_PID=""
if [[ "$DRY" != 1 ]]; then
    if [[ -n "${RCG_CONDA_ENV:-}" ]]; then
        source "${CONDA_SH:?Set CONDA_SH when selecting RCG_CONDA_ENV}"
        set +u; conda activate "$RCG_CONDA_ENV"; set -u
    fi
    export WANDB_DIR="${WANDB_DIR:-$LOG_ROOT/wandb}"
    export HF_HOME="${HF_HOME:-$ROOT_DIR/assets/cache/huggingface}"
    mkdir -p "$LOG_ROOT" "$WANDB_DIR"
    "$PYTHON" -c 'from tools.a5000_preflight import require_gpu; require_gpu()'
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
    [[ -z "${ONLY:-}" || "$label" =~ ${ONLY} ]] && [[ -z "${SKIP:-}" || ! "$label" =~ ${SKIP} ]]
}

run_step() {
    local label="$1"; shift
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
        "$PYTHON" tools/a5000_preflight.py --model "$model" --method "$method" \
            --dataset "$dataset" --out-root "$PREFLIGHT_ROOT" --check
    fi
}

train_official() {
    local label="$1" dataset="$2" model="$3" method="$4" lr="$5"; shift 5
    if selected "$label"; then require_training_admission "$dataset" "$model" "$method"; fi
    run_step "$label" "$PYTHON" -m src.train --model "$model" --method "$method" \
        --learning_rate "$lr" --dataset "$dataset" --results-root "$REVISION_ROOT/$dataset" \
        --protocol archived --memory-profile a5000 --gen-batch-size 1 --attn-implementation sdpa \
        --recovery-steps "${RCG_RECOVERY_STEPS:-100}" --skip-if-complete "$@"
}

train_sweep() {
    local label="$1" dataset="$2" model="$3" method="$4" first="$5" last="$6" subset="$7"; shift 7
    local id
    for ((id=first;id<=last;id++)); do
        if selected "${label}_lr${id}"; then require_training_admission "$dataset" "$model" "$method"; fi
        run_step "${label}_lr${id}" "$PYTHON" -m src.train --model "$model" --method "$method" \
            --experiment_id "$id" --subset "$subset" --dataset "$dataset" \
            --results-root "$REVISION_ROOT/$dataset/sweeps" --protocol archived \
            --memory-profile a5000 --gen-batch-size 1 --attn-implementation sdpa \
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
        "$PYTHON" tools/a5000_inference_probe.py --kind icl --model "$model" \
            --out-root "$PREFLIGHT_ROOT" --check
    fi
    run_step "$label" "$PYTHON" -m src.icl_inference --model_name "$model" --dataset "$dataset" \
        --prompt-version v2 --memory-profile a5000 --batch-size 1 --max-prompt-tokens 2048 \
        --attn-implementation sdpa --results_dir "$REVISION_ROOT/icl/$dataset" --skip-if-complete "$@"
}

declare -A GEN_BS=([codet5]=1 [plbart]=1 [qwen25-0.5b]=1 [qwen25]=1 [qwen15]=1 [qwen25-14b]=1)
finish_report() {
    [[ "$STOP_REQUESTED" == 0 ]] || exit 75
    printf 'Phase %s: %s successful, %s filtered, %s failed\n' "$JOB_TAG" "${#STEPS_OK[@]}" "${#STEPS_SKIPPED[@]}" "${#STEPS_FAILED[@]}"
    if [[ ${#STEPS_SKIPPED[@]} -gt 0 ]]; then
        echo "Filtered execution is not a complete campaign; resubmit without ONLY/SKIP for its gate."
        [[ "$DRY" == 1 ]] || return 75
    fi
}
