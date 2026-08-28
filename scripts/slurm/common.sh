#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/slurm/common.sh — shared preamble for every H100 campaign job.
#
# Sourced (never executed) by the *.sbatch files in this directory. Provides:
#   * repo root + conda + environment exports (identical for every campaign)
#   * run_step  — run one campaign command, log it, keep going on failure
#   * finish_report — final PASS/FAIL summary; non-zero exit if anything failed
#   * best_sweep_lr — pick the official LR from a finished 3-point sweep
#
# Knobs (all optional, all env vars):
#   RCG_CONDA_ENV   conda env to activate            (default: kse)
#   RCG_ROOT        repo root                        (default: $SLURM_SUBMIT_DIR / $PWD)
#   RCG_LOG_ROOT    per-step log root                (default: assets/logs/campaign)
#   WANDB_MODE      offline|online                   (default: offline — see README)
#   DRY=1           print every command, run nothing (offline preview of a queue)
#   ONLY=<regex>    run only steps whose label matches
#   SKIP=<regex>    skip steps whose label matches
#   RUN_P0/P1/P2    per-campaign priority blocks (defaults set by each sbatch)
# ---------------------------------------------------------------------------
set -euo pipefail

# --- repo root -------------------------------------------------------------
ROOT_DIR="${RCG_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT_DIR"
if [[ ! -f run_training.sh || ! -d src ]]; then
    echo "common.sh: '$ROOT_DIR' is not the repo root (no run_training.sh/src)." >&2
    echo "Submit from the repo root or export RCG_ROOT=/path/to/repo." >&2
    exit 1
fi

# --- cluster modules -------------------------------------------------------
# module load cuda/12.4 gcc/12        # EDIT to your cluster's module names

# --- conda -----------------------------------------------------------------
CONDA_ENV="${RCG_CONDA_ENV:-kse}"
CONDA_PROFILE="${CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_PROFILE" ]]; then
    echo "Conda profile not found: $CONDA_PROFILE — set CONDA_SH to conda.sh." >&2
    exit 1
fi
# conda's activate scripts are not -u clean on every install
set +u
# shellcheck disable=SC1090
source "$CONDA_PROFILE"
conda activate "$CONDA_ENV"
set -u

# --- environment (mirrors run_training.sh; all artifacts stay under assets/) --
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/assets/wandb}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/assets/cache/huggingface}"
# offline by default: a compute node without egress must never stall a 20 h run.
# Sync afterwards with `wandb sync assets/wandb/offline-run-*`.
export WANDB_MODE="${WANDB_MODE:-offline}"

# HUGGINGFACE_TOKEN for gated repos (Llama-3.1) lives in the gitignored .env only.
if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
fi

JOB_TAG="${SLURM_JOB_NAME:-local}"
LOG_ROOT="${RCG_LOG_ROOT:-assets/logs/campaign}/$JOB_TAG"
mkdir -p "$LOG_ROOT" "$WANDB_DIR" "$HF_HOME" assets/logs/slurm

# --- bookkeeping -----------------------------------------------------------
STEP_INDEX=0
STEPS_OK=()
STEPS_FAILED=()
STEPS_SKIPPED=()

banner() {
    echo
    echo "==================================================================="
    echo "$*"
    echo "==================================================================="
}

# run_step <label> <cmd...> — one campaign command. Never aborts the job:
# a failed run is recorded and the queue moves on (a 3-day job must not die
# on run 4 of 36). Env-prefixed commands go through `env VAR=val cmd ...`.
run_step() {
    local label="$1"
    shift
    STEP_INDEX=$((STEP_INDEX + 1))

    if [[ -n "${ONLY:-}" && ! "$label" =~ ${ONLY} ]]; then
        STEPS_SKIPPED+=("$label"); echo "--- skip (ONLY)  $label"; return 0
    fi
    if [[ -n "${SKIP:-}" && "$label" =~ ${SKIP} ]]; then
        STEPS_SKIPPED+=("$label"); echo "--- skip (SKIP)  $label"; return 0
    fi

    echo
    echo "-------------------------------------------------------------------"
    echo ">>> [$STEP_INDEX] $label   ($(date '+%F %T'))"
    printf '    cmd:'; printf ' %q' "$@"; printf '\n'

    if [[ "${DRY:-0}" == "1" ]]; then
        STEPS_SKIPPED+=("$label (dry)"); echo "    (DRY=1 — not executed)"; return 0
    fi

    local log="$LOG_ROOT/${label}.log" start=$SECONDS rc=0
    echo "    log: $log"
    set +e
    "$@" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    set -e
    local mins=$(( (SECONDS - start) / 60 ))

    if [[ $rc -eq 0 ]]; then
        STEPS_OK+=("$label"); echo "<<< OK      $label   (${mins} min)"
    else
        STEPS_FAILED+=("$label (exit $rc)")
        echo "<<< FAILED  $label   (exit $rc, ${mins} min) — continuing" >&2
    fi
    return 0
}

# require_var <NAME> <hint> — guard for steps that need a human-selected value
# (e.g. the official LR chosen from a sweep). Returns 1 (step skipped) if unset.
require_var() {
    local name="$1" hint="$2"
    if [[ -z "${!name:-}" ]]; then
        echo "--- skip: $name is unset. $hint"
        STEPS_SKIPPED+=("needs \$$name")
        return 1
    fi
    return 0
}

# best_sweep_lr <sweep_root> <stem> [ids...] — echo the LR whose sweep run has
# the best validation BLEU-4 (trainer_state.json best_metric). Empty on failure.
best_sweep_lr() {
    python tools/select_sweep_lr.py --sweep-root "$1" --stem "$2" --quiet 2>/dev/null || true
}

# --- campaign run builders -------------------------------------------------
# Generation batch size per model (budget.md: the single largest lever; bf16
# beam ties make it batch-size sensitive, so it is pinned per model — KI-016).
declare -A GEN_BS=(
    [codet5]=32 [plbart]=32 [qwen25-0.5b]=16 [qwen25]=8 [qwen15]=8 [qwen25-14b]=8
)
REVISION_ROOT="assets/results/revision"
WANDB_TAGS_DEFAULT=""

# train_official <label> <dataset> <model> <method> <lr> [extra src.train args...]
#   -> assets/results/revision/<dataset>/<stem>_official_<lr>[_seed<n>]/
train_official() {
    local label="$1" dataset="$2" model="$3" method="$4" lr="$5"
    shift 5
    local extra="--dataset $dataset --results-root $REVISION_ROOT/$dataset/"
    extra+=" --protocol archived --gen-batch-size ${GEN_BS[$model]:-8} --skip-if-complete"
    [[ $# -gt 0 ]] && extra+=" $*"
    run_step "$label" env WANDB_GROUP="$dataset" WANDB_TAGS="${WANDB_TAGS:-$WANDB_TAGS_DEFAULT}" \
        TRAIN_EXTRA_ARGS="$extra" ./run_training.sh "$model" "$method" fixed "$lr"
}

# train_sweep <label> <dataset> <model> <method> <start_id> <end_id> <subset> [extra...]
#   -> assets/results/revision/<dataset>/sweeps/<stem>_<id>/   (lr = constants.LR[id])
train_sweep() {
    local label="$1" dataset="$2" model="$3" method="$4" start="$5" end="$6" subset="$7"
    shift 7
    local extra="--dataset $dataset --results-root $REVISION_ROOT/$dataset/sweeps/"
    extra+=" --protocol archived --gen-batch-size ${GEN_BS[$model]:-8} --skip-if-complete"
    [[ $# -gt 0 ]] && extra+=" $*"
    run_step "$label" env WANDB_GROUP="$dataset" WANDB_TAGS="${WANDB_TAGS:-$WANDB_TAGS_DEFAULT}" \
        TRAIN_EXTRA_ARGS="$extra" ./run_training.sh "$model" "$method" sweep "$start" "$end" "$subset"
}

# icl_run <label> <dataset> <model> [extra src.icl_inference args...]
#   -> assets/results/revision/icl/<dataset>/<model>_icl_<k>ex_<sel>[_s<seed>]_v2.json
icl_run() {
    local label="$1" dataset="$2" model="$3"
    shift 3
    run_step "$label" python -m src.icl_inference \
        --model_name "$model" --dataset "$dataset" --prompt-version v2 \
        --batch-size "${GEN_BS[$model]:-8}" \
        --results_dir "$REVISION_ROOT/icl/$dataset" --skip-if-complete "$@"
}

finish_report() {
    banner "Finished at $(date '+%F %T')  —  ${#STEPS_OK[@]} ok, ${#STEPS_FAILED[@]} failed, ${#STEPS_SKIPPED[@]} skipped"
    [[ ${#STEPS_OK[@]}      -gt 0 ]] && printf '  ok      %s\n' "${STEPS_OK[@]}"
    [[ ${#STEPS_SKIPPED[@]} -gt 0 ]] && printf '  skipped %s\n' "${STEPS_SKIPPED[@]}"
    if [[ ${#STEPS_FAILED[@]} -gt 0 ]]; then
        printf '  FAILED  %s\n' "${STEPS_FAILED[@]}" >&2
        echo "Re-submit the same script after fixing: every run line is idempotent" >&2
        echo "(--skip-if-complete), so finished runs are skipped instantly." >&2
        return 1
    fi
    echo "All requested steps completed."
    return 0
}

# --- start banner ----------------------------------------------------------
banner "$(date '+%F %T')  job=${SLURM_JOB_ID:-local}  host=$(hostname)"
echo "repo:   $ROOT_DIR ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "conda:  $CONDA_ENV  ($(python -c 'import sys;print(sys.version.split()[0])'))"
echo "wandb:  $WANDB_MODE   HF_HOME: $HF_HOME"
echo "logs:   $LOG_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU: none visible"
