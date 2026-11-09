#!/usr/bin/env bash
# =====================================================================================
# Revision campaign on Spartan HPC (University of Melbourne, Slurm): the single job file.
# Setup (environment, inputs, .env): scripts/spartan/README.md. From the repository root:
#
#   bash scripts/spartan/revision.slurm                     submit with 2 x A100 80GB (default)
#   bash scripts/spartan/revision.slurm --gpus 1|2|3|4      submit with that many A100s
#        [--mem-gb N] [--time D-HH:MM:SS] [--preview]       host RAM (default 32 GB/GPU), walltime
#                                                           (default 7-00:00:00); --preview prints only
#   bash scripts/spartan/revision.slurm --hf-sync-only      section 2 on the login node (no GPU)
#   bash scripts/spartan/revision.slurm --status            per-section progress (read-only)
#   sbatch scripts/spartan/revision.slurm                   also works (2 GPUs; the output
#                                                           directory assets/logs/spartan must exist)
#
# Required: RCG_PYTHON = absolute python of the rcg-a100 environment. Optional:
# RCG_SPARTAN_ACCOUNT (project, e.g. punim1234 -> --account), RCG_MODULES (modules to load
# in the job), HF_TOKEN in .env (auto-push), RCG_HF_SYNC=0 (no download: compute node offline).
# Stop the campaign: touch assets/results/revision-a100-native/.pool/stop (drains, no resubmit).
#
# The job runs six sections. Each one prints a start/end banner (time, exit code,
# duration) here in assets/logs/spartan/rcg_revision-<jobid>.out and writes its own log to
# assets/logs/spartan/<jobid>/NN-<section>.log:
#   01 environment  modules, interpreter, torch, allocated GPUs (1-4 x A100 80GB)
#   02 hf-status    download finished results missing here from the public HF dataset,
#                   then report done/missing per section (campaigns A-D, judges, pool)
#   03 inputs       datasets, judge weights, model cache
#   04 campaign     the pool (scripts/pool/pool.sh, log 04-campaign-pool.log) over the 226 tasks;
#                   a per-section progress table every 30 minutes; finished work is skipped
#   05 hf-push      upload new results to the same HF dataset while the campaign runs
#                   (after tasks complete, at most every 20 minutes) and once at the end
#   06 summary      final table; if the walltime ends with work left, resubmit this file
# =====================================================================================
#SBATCH --job-name=rcg_revision
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=assets/logs/spartan/%x-%j.out
set -uo pipefail

JOB_NAME=rcg_revision
CPUS_PER_GPU=8                   # gpu-a100 node: 32 cores, 4 GPUs
MAX_MEM_GB=480                   # gpu-a100 node: 495,000 MB
if [[ -n "${RCG_ROOT:-}" ]]; then ROOT="$RCG_ROOT"
elif [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" == "$JOB_NAME" ]]; then ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$ROOT" 2>/dev/null && [[ -f src/train.py ]] || { echo "Not the repository root: $ROOT (set RCG_ROOT)" >&2; exit 2; }
SELF="$ROOT/scripts/spartan/revision.slurm"
RESULTS="${RCG_RESULTS_ROOT:-$ROOT/assets/results/revision-a100-native}"
JUDGES="${RCG_JUDGE_ROOT:-$ROOT/assets/results/a100-native-judges}"
STATE="${RCG_POOL_STATE:-$RESULTS/.pool}"
LOG_ROOT="${RCG_SPARTAN_LOG_ROOT:-$ROOT/assets/logs/spartan}"   # this file's .out and section logs
TASK_LOGS="${RCG_LOG_ROOT:-$ROOT/assets/logs/a100}"           # task and pool logs (common.sh, pool.sh)
HF_REPO="${RCG_HF_REPO:-PhucInAI/code-review-comment-generation}"
SBATCH="${RCG_SBATCH:-sbatch}" SQUEUE="${RCG_SQUEUE:-squeue}" NVSMI="${RCG_NVIDIA_SMI:-nvidia-smi}"
PROGRESS_EVERY="${RCG_SPARTAN_PROGRESS_EVERY:-1800}"
PUSH_EVERY="${RCG_SPARTAN_PUSH_EVERY:-1200}"
POLL="${RCG_SPARTAN_POLL:-60}"

# HF scope, as repository-relative paths (the dataset mirrors the repository layout).
relative() { [[ "$1" == "$ROOT"/* ]] || { echo "$1 is outside $ROOT; the HF mirror needs repository paths" >&2; return 1; }
             printf '%s' "${1#"$ROOT"/}"; }
REL_RESULTS=$(relative "$RESULTS") && REL_JUDGES=$(relative "$JUDGES") &&
    REL_LOGS=$(relative "$LOG_ROOT") && REL_TASK_LOGS=$(relative "$TASK_LOGS") || exit 2
# Resume state, locks and checkpoints never leave the cluster (same scope as the
# 2026-11-03 push). Weights are not downloaded: no campaign step reads trained weights.
EXCLUDE=('*/checkpoint-*' '*/.pool/*' '*.lock' '*/test_progress/*' '*.progress/*' '.cache/*')
DOWNLOAD_INCLUDE=("$REL_RESULTS/*" "$REL_JUDGES/*")
DOWNLOAD_EXCLUDE=("${EXCLUDE[@]}")
[[ "${RCG_HF_WEIGHTS:-0}" == 1 ]] || DOWNLOAD_EXCLUDE+=('*.safetensors' '*.bin' '*.pt' '*.pth')
PUSH_INCLUDE=("$REL_RESULTS/*" "$REL_JUDGES/*" "$REL_TASK_LOGS/*" "$REL_LOGS/*")

stamp() { date '+%F %T'; }
duration() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# ---- launcher ------------------------------------------------------------------------
usage() { sed -n '5,13p' "$SELF" | sed 's/^# \{0,1\}//' >&2; exit 2; }
submit() {    # submit <preview|submit|continue> ; uses GPUS MEM_GB WALLTIME
    local mode="$1" queued
    [[ "$GPUS" =~ ^[1-4]$ ]] || { echo "--gpus must be 1, 2, 3 or 4 (one gpu-a100 node has 4)" >&2; return 2; }
    MEM_GB="${MEM_GB:-$((32 * GPUS))}"
    [[ "$MEM_GB" =~ ^[1-9][0-9]{0,2}$ ]] && (( MEM_GB <= MAX_MEM_GB )) ||
        { echo "--mem-gb must be 1..$MAX_MEM_GB" >&2; return 2; }
    [[ "$WALLTIME" =~ ^([0-7]-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$ ]] ||
        { echo "--time must be [D-]HH:MM:SS, at most 7 days on gpu-a100" >&2; return 2; }
    [[ -n "${RCG_PYTHON:-}" && "$RCG_PYTHON" == /* ]] || { echo "Set RCG_PYTHON to the absolute rcg-a100 python" >&2; return 2; }
    [[ "$ROOT$RCG_PYTHON" != *,* ]] || { echo "Paths passed with --export cannot contain commas" >&2; return 2; }
    local args=("$SBATCH" --job-name="$JOB_NAME" --partition="${RCG_SPARTAN_PARTITION:-gpu-a100}" --nodes=1 --ntasks=1
        --gres="gpu:$GPUS" --cpus-per-task=$((CPUS_PER_GPU * GPUS)) --mem="${MEM_GB}G" --time="$WALLTIME"
        --chdir="$ROOT" --output="$LOG_ROOT/%x-%j.out"
        --export="ALL,RCG_ROOT=$ROOT,RCG_PYTHON=$RCG_PYTHON,RCG_SPARTAN_MEM_GB=$MEM_GB,RCG_SPARTAN_TIME=$WALLTIME")
    if [[ -n "${RCG_SPARTAN_ACCOUNT:-}" ]]; then args+=(--account="$RCG_SPARTAN_ACCOUNT")
    elif [[ "$mode" != continue ]]; then echo "note: RCG_SPARTAN_ACCOUNT unset; Slurm uses your default project" >&2; fi
    [[ "$mode" != continue ]] || args+=(--dependency="afterany:$SLURM_JOB_ID")
    args+=("$SELF")
    if [[ "$mode" == preview ]]; then printf '%q ' "${args[@]}"; printf '\n'; return 0; fi
    [[ -x "$RCG_PYTHON" ]] || { echo "RCG_PYTHON is not executable: $RCG_PYTHON" >&2; return 2; }
    if [[ -e "$STATE/stop" ]]; then
        echo "Stop file present ($STATE/stop): remove it to resume the campaign" >&2; return 2; fi
    source scripts/pool/check_judge_inputs.sh
    check_judge_inputs || return 2
    # One campaign job at a time (a continuation ignores itself).
    queued=$("$SQUEUE" -h -u "${USER:-$(id -un)}" -n "$JOB_NAME" -o %i 2>&1) ||
        { echo "squeue failed ($queued); not submitting without knowing the queue" >&2; return 3; }
    queued=$(grep -vx "${SLURM_JOB_ID:-none}" <<< "$queued" | grep . | tr '\n' ' ')
    [[ -z "$queued" ]] || { echo "A $JOB_NAME job is already queued/running: $queued- not submitting another" >&2; return 0; }
    mkdir -p "$LOG_ROOT"
    # A job's own Slurm variables must not leak into the job it submits.
    env $(compgen -e | grep -E '^(SLURM_|SBATCH_)' | grep -vx SLURM_CONF | sed 's/^/-u /') -u CUDA_VISIBLE_DEVICES "${args[@]}"
}

# ---- sections --------------------------------------------------------------------------
# section NN name function: banner, output to this .out and to NN-name.log, end banner.
section() {
    local n="$1" name="$2" fn="$3" log="$LOGDIR/$1-$2.log" t0=$SECONDS rc=0 tee_pid
    printf '\n==== [%s] %s | start %s | log %s\n' "$n" "$name" "$(stamp)" "$log" | tee -a "$log"
    exec 3>&1 4>&2
    exec > >(tee -a "$log") 2>&1
    tee_pid=$!
    "$fn" || rc=$?
    exec 1>&3 2>&4 3>&- 4>&-
    wait "$tee_pid" 2>/dev/null || true
    printf '==== [%s] %s | end %s | exit %s | %s\n' "$n" "$name" "$(stamp)" "$rc" "$(duration $((SECONDS - t0)))" | tee -a "$log"
    return "$rc"
}

sec_environment() {
    echo "job ${SLURM_JOB_ID:-} on ${SLURMD_NODENAME:-$(hostname)} partition ${SLURM_JOB_PARTITION:-?}" \
         "cpus ${SLURM_CPUS_PER_TASK:-?} mem ${RCG_SPARTAN_MEM_GB:-?}G time ${RCG_SPARTAN_TIME:-?}"
    echo "repository $ROOT ($(cat SOURCE_COMMIT 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo 'no commit info'))"
    if [[ -n "${RCG_MODULES:-}" ]]; then
        type module >/dev/null 2>&1 || { echo "RCG_MODULES set but the module command is unavailable" >&2; return 2; }
        # shellcheck disable=SC2086
        module load $RCG_MODULES || { echo "module load $RCG_MODULES failed" >&2; return 2; }
    fi
    [[ -n "${RCG_PYTHON:-}" && -x "$RCG_PYTHON" ]] || { echo "Set RCG_PYTHON to the rcg-a100 interpreter (absolute path)" >&2; return 2; }
    export PATH="$(dirname "$RCG_PYTHON"):$PATH"
    export RCG_ROOT="$ROOT" RCG_TARGET_GPU="${RCG_TARGET_GPU:-a100_80gb}" WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"     # 30 s default failed task 35 (2026-10-29)
    export RCG_POOL_LOG_ROOT="${RCG_POOL_LOG_ROOT:-$TASK_LOGS/pool}"
    export HF_HOME="${HF_HOME:-$ROOT/assets/cache/huggingface}" NLTK_DATA="${NLTK_DATA:-$ROOT/assets/cache/nltk_data}"
    echo "python $RCG_PYTHON"
    "$RCG_PYTHON" -c 'import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__,
        "cuda", torch.version.cuda, "devices", torch.cuda.device_count())' || { echo "torch import failed" >&2; return 2; }
    local gpus="${SLURM_GPUS_ON_NODE:-}" listed count small
    [[ "$gpus" =~ ^[1-4]$ ]] || { echo "Allocation must hold 1-4 GPUs (SLURM_GPUS_ON_NODE=${gpus:-unset})" >&2; return 2; }
    listed=$("$NVSMI" --query-gpu=index,name,memory.total --format=csv,noheader,nounits) || { echo "nvidia-smi failed" >&2; return 2; }
    echo "$listed"
    count=$(grep -c . <<< "$listed")
    (( count == gpus )) || { echo "nvidia-smi shows $count GPUs, the allocation $gpus" >&2; return 2; }
    small=$(awk -F', *' '$3 + 0 < 79000' <<< "$listed")
    [[ -z "$small" ]] || { echo "Every GPU must be an A100 80GB; smaller: $small" >&2; return 2; }
    echo "OK: $gpus x A100 80GB, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
}

# Download every in-scope file of the HF dataset that is missing here. Existing local files
# are never overwritten (they are at least as new as the mirror).
hf_sync() {
    "$RCG_PYTHON" - "$HF_REPO" "$ROOT" "${DOWNLOAD_INCLUDE[*]}" "${DOWNLOAD_EXCLUDE[*]}" <<'PY'
import fnmatch, glob, os, sys
from huggingface_hub import HfApi, snapshot_download
repo, root, include, exclude = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
files = HfApi().list_repo_files(repo, repo_type="dataset")
scope = [f for f in files if any(fnmatch.fnmatch(f, p) for p in include)
         and not any(fnmatch.fnmatch(f, p) for p in exclude)]
missing = [f for f in scope if not os.path.exists(os.path.join(root, f))]
print(f"HF {repo}: {len(files)} files, {len(scope)} in scope, "
      f"{len(scope) - len(missing)} already here, {len(missing)} to download")
if missing:
    # Exact paths as patterns: escape [ ] * ? so every name matches only itself.
    snapshot_download(repo, repo_type="dataset", local_dir=root, allow_patterns=[glob.escape(f) for f in missing],
                      max_workers=int(os.environ.get("RCG_HF_WORKERS", "8")))
    still = [f for f in missing if not os.path.exists(os.path.join(root, f))]
    if still:
        sys.exit(f"{len(still)} files still missing after the download, e.g. {still[:3]}")
print("HF sync complete")
PY
}
sec_hf_status() {
    if [[ "${RCG_HF_SYNC:-1}" == 1 ]]; then
        echo "-- download: https://huggingface.co/datasets/$HF_REPO (${DOWNLOAD_INCLUDE[*]})"
        hf_sync || {
            echo "HF check failed (no internet on this node?). Run it on the login node:" >&2
            echo "  bash scripts/spartan/revision.slurm --hf-sync-only" >&2
            echo "then resubmit with RCG_HF_SYNC=0 exported (status from local files only)." >&2
            return 3; }
    else
        echo "-- download skipped (RCG_HF_SYNC=0): status from local files only"
    fi
    echo "-- producer campaigns (tools/campaign_status.py: validated complete_results / ICL outputs)"
    local phase out missing
    for phase in A B C D; do
        out=$("$RCG_PYTHON" tools/campaign_status.py "$phase" --results-root "$RESULTS" 2>&1)
        missing=$(grep -c '^INCOMPLETE ' <<< "$out")
        printf 'section %s: %s\n' "$phase" "$(grep -E '^Phase ' <<< "$out" | tail -1 || tail -1 <<< "$out")"
        (( missing == 0 )) || grep '^INCOMPLETE ' <<< "$out" | sed 's/^/    /'
    done
    echo "-- judges ($JUDGES)"
    local backend
    for backend in qwen-revision llama-legacy llama-revision; do
        printf 'judge %-15s scored files: %s\n' "$backend" \
            "$(find "$JUDGES/$backend/scored" -type f 2>/dev/null | wc -l)"
    done
    echo "-- pool state ($STATE)"
    bash scripts/pool/pool_status.sh
    echo "Finished work is skipped: training/ICL steps run with --skip-if-complete; judges reuse their cache."
}

sec_inputs() {
    local rc=0 dir
    for dir in assets/data/Trans_Review assets/data/Code_Review; do
        [[ -n "$(ls -A "$dir" 2>/dev/null)" ]] && echo "OK $dir" || { echo "MISSING $dir" >&2; rc=2; }
    done
    [[ -s paper/generated/cr_judge_subset.json ]] && echo "OK paper/generated/cr_judge_subset.json" ||
        { echo "MISSING paper/generated/cr_judge_subset.json" >&2; rc=2; }
    source scripts/pool/check_judge_inputs.sh
    check_judge_inputs && echo "OK judge weights (Qwen2.5-Coder-3B, Llama-3.1-8B)" || rc=2
    local cached; cached=$(find "$HF_HOME/hub" -maxdepth 1 -name 'models--*' 2>/dev/null | wc -l)
    echo "model cache $HF_HOME: $cached models"
    (( cached )) || echo "WARNING: empty model cache; base models download at first use (needs internet)"
    return "$rc"
}

# ---- auto-push -----------------------------------------------------------------------
# .env values (KEY=value, optional quotes, trailing " # comment" removed).
env_value() { grep -E "^(export +)?$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- | sed -E 's/[[:space:]]+#.*$//' | tr -d "\"' "; }
hf_token() { [[ -n "${HF_TOKEN:-}" ]] && { printf '%s' "$HF_TOKEN"; return; }; env_value HF_TOKEN; }
# Every secret-looking value of .env (and the token in use), one per line, for the leak scan.
secret_values() {
    { hf_token; echo
      grep -oE '^(export +)?[A-Za-z_][A-Za-z0-9_]*=' .env 2>/dev/null | sed -E 's/^export +//; s/=$//' |
          while read -r key; do env_value "$key"; echo; done; } | awk 'length($0) >= 12' | sort -u
}
hf_push() {
    local token hf dirs=() leaks
    token=$(hf_token)
    [[ -n "$token" ]] || { echo "HF_TOKEN not set (.env or environment): auto-push skipped"; return 0; }
    hf="${RCG_HF:-$(dirname "$RCG_PYTHON")/hf}"
    for dir in "$RESULTS" "$JUDGES" "$TASK_LOGS" "$LOG_ROOT"; do [[ -e "$dir" ]] && dirs+=("$dir"); done
    # The dataset is public: refuse to push anything that contains a token pattern (text
    # files) or any .env value (all files, binary wandb records included; weights skipped).
    local skip=(--exclude-dir='checkpoint-*' --exclude-dir=.pool --exclude-dir=test_progress
                --exclude-dir='*.progress' --exclude-dir=.cache)
    local secrets; secrets=$(umask 077; mktemp) || return 4; secret_values > "$secrets"
    leaks=$( { grep -rIlE 'hf_[A-Za-z0-9]{30,}' "${dirs[@]}" "${skip[@]}" 2>/dev/null
               grep -rlF -f "$secrets" "${dirs[@]}" "${skip[@]}" --exclude='*.safetensors' --exclude='*.bin' \
                   --exclude='*.pt' --exclude='*.pth' 2>/dev/null; } | sort -u)
    rm -f "$secrets"
    [[ -z "$leaks" ]] || { echo "Possible secret in files to push; push skipped:"; sed 's/^/    /' <<< "$leaks"; return 4; }
    echo "push $(stamp): ${PUSH_INCLUDE[*]} -> https://huggingface.co/datasets/$HF_REPO (public)"
    HF_TOKEN="$token" HF_HUB_DISABLE_TELEMETRY=1 "$hf" upload-large-folder "$HF_REPO" "$ROOT" --repo-type dataset \
        --include "${PUSH_INCLUDE[@]}" --exclude "${EXCLUDE[@]}" --num-workers "${RCG_HF_UPLOAD_WORKERS:-4}"
}
push_log() { local rc=0; { echo "== push $(stamp) (done tasks: $1)"; hf_push; } >> "$LOGDIR/05-hf-push.log" 2>&1 || rc=$?
             echo "== push exit $rc $(stamp)" >> "$LOGDIR/05-hf-push.log"; return "$rc"; }
done_count() { find "$STATE/done" -type f 2>/dev/null | wc -l; }

# ---- campaign ------------------------------------------------------------------------
progress() {
    echo "-- progress $(stamp)"
    bash scripts/pool/pool_status.sh
    [[ ! -s "$LOGDIR/04-campaign-pool.log" ]] || { echo "pool log (last lines):"; tail -3 "$LOGDIR/04-campaign-pool.log" | sed 's/^/    /'; }
}
POOL_RC="" TERMINATED=0 CONTINUATION="not needed"
# Work left at the walltime: queue the continuation as soon as the pool exits, before any
# push (the job may have only minutes left). It starts after this job ends.
continue_campaign() {
    if (( TERMINATED )); then CONTINUATION="not submitted: the job was terminated (scancel)"
    elif [[ -e "$STATE/stop" ]]; then CONTINUATION="not submitted: stop file present"
    elif [[ "${RCG_SPARTAN_RESUBMIT:-1}" != 1 ]]; then CONTINUATION="not submitted: RCG_SPARTAN_RESUBMIT=0"
    elif out=$(GPUS="$SLURM_GPUS_ON_NODE" MEM_GB="${RCG_SPARTAN_MEM_GB:-}" WALLTIME="${RCG_SPARTAN_TIME:-7-00:00:00}" \
                  submit continue 2>&1); then CONTINUATION="submitted: $out"
    else CONTINUATION="FAILED ($out); run: bash scripts/spartan/revision.slurm --gpus $SLURM_GPUS_ON_NODE"; fi
    echo "continuation $CONTINUATION"
}
sec_campaign() {
    local pool_pid push_pid="" last_push=$((SECONDS - PUSH_EVERY)) pushed_done last_report=$SECONDS done_now
    local pool_cmd=(bash "$ROOT/scripts/pool/pool.sh")
    [[ -z "${RCG_SPARTAN_POOL_CMD:-}" ]] || read -ra pool_cmd <<< "$RCG_SPARTAN_POOL_CMD"     # tests
    pushed_done=$(done_count)
    # The pool logs to 04-campaign-pool.log; the section log and .out get the progress tables.
    pool_pid=""     # set before the trap so a TERM during the launch still checkpoints
    trap 'TERMINATED=1; echo "termination requested $(stamp); waiting for the pool to checkpoint"; [[ -z "$pool_pid" ]] || kill -TERM "$pool_pid" 2>/dev/null' TERM INT
    "${pool_cmd[@]}" >> "$LOGDIR/04-campaign-pool.log" 2>&1 &
    pool_pid=$!
    echo "pool started (pid $pool_pid); its log: $LOGDIR/04-campaign-pool.log; task logs: $RCG_POOL_LOG_ROOT/$SLURM_JOB_ID/"
    while kill -0 "$pool_pid" 2>/dev/null; do
        sleep "$POLL" & wait $! 2>/dev/null
        done_now=$(done_count)
        if [[ -n "$push_pid" ]] && ! kill -0 "$push_pid" 2>/dev/null; then wait "$push_pid"; push_pid=""; fi
        if [[ -z "$push_pid" ]] && (( ! TERMINATED && done_now != pushed_done && SECONDS - last_push >= PUSH_EVERY )); then
            echo "auto-push started $(stamp): $done_now tasks done (log $LOGDIR/05-hf-push.log)"
            push_log "$done_now" & push_pid=$!
            pushed_done="$done_now"; last_push=$SECONDS
        fi
        (( SECONDS - last_report < PROGRESS_EVERY )) || { progress; last_report=$SECONDS; }
    done
    POOL_RC=0; wait "$pool_pid" || POOL_RC=$?
    echo "pool exit $POOL_RC (0 all done, 75 work left, 1 only failed/blocked tasks left)"
    [[ "$POOL_RC" != 75 ]] || continue_campaign
    trap 'TERMINATED=1' TERM INT
    [[ -z "$push_pid" ]] || wait "$push_pid" || true
    progress
    case "$POOL_RC" in 0|1|75) return 0 ;; *) return "$POOL_RC" ;; esac
}
sec_push() {     # the final push, printed here (pushes during the campaign append to this log)
    (( ! TERMINATED )) || { echo "job terminated: final push skipped (the next job pushes)"; return 0; }
    echo "final push (done tasks: $(done_count))"
    hf_push
}
sec_summary() {
    bash scripts/pool/pool_status.sh
    case "$POOL_RC" in
        0) echo "Campaign complete." ;;
        75) echo "Work left; continuation $CONTINUATION" ;;
        1) echo "Only failed or blocked tasks remain: fix the cause, delete $STATE/failed/<id>, resubmit."
           echo "If you deleted outputs to force a rerun, submit with RCG_HF_SYNC=0 so the HF check does not restore them." ;;
        *) echo "Pool error (exit ${POOL_RC:-not run}); see $LOGDIR/04-campaign-pool.log" ;;
    esac
}

run_job() {
    LOGDIR="$LOG_ROOT/$SLURM_JOB_ID"
    mkdir -p "$LOGDIR"
    echo "rcg_revision job $SLURM_JOB_ID started $(stamp); section logs in $LOGDIR"
    section 01 environment sec_environment || exit 2
    section 02 hf-status sec_hf_status || exit 3
    section 03 inputs sec_inputs || exit 2
    local rc=0
    section 04 campaign sec_campaign || rc=$?
    section 05 hf-push sec_push || true          # a failed push never fails the campaign
    section 06 summary sec_summary || true
    (( rc == 0 )) && rc="$POOL_RC"
    echo "rcg_revision job $SLURM_JOB_ID finished $(stamp), exit $rc"
    exit "$rc"
}

# ---- entry -----------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" == "$JOB_NAME" ]]; then
    (( $# == 0 )) || { echo "The job takes no arguments; use the launcher options when submitting" >&2; exit 2; }
    run_job
fi
mode=submit GPUS=2 MEM_GB="" WALLTIME=7-00:00:00
while (( $# )); do
    case "$1" in
        --gpus) [[ $# -ge 2 ]] || usage; GPUS="$2"; shift 2 ;;
        --mem-gb) [[ $# -ge 2 ]] || usage; MEM_GB="$2"; shift 2 ;;
        --time) [[ $# -ge 2 ]] || usage; WALLTIME="$2"; shift 2 ;;
        --preview) mode=preview; shift ;;
        --hf-sync-only) mode=sync; shift ;;
        --status) mode=status; shift ;;
        *) usage ;;
    esac
done
case "$mode" in
    status) bash scripts/pool/pool_status.sh
            "$SQUEUE" -u "${USER:-$(id -un)}" -n "$JOB_NAME" 2>/dev/null || true ;;
    sync) [[ -n "${RCG_PYTHON:-}" && -x "$RCG_PYTHON" ]] || { echo "Set RCG_PYTHON to the rcg-a100 python" >&2; exit 2; }
          LOGDIR="$LOG_ROOT/login-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$LOGDIR"
          RCG_HF_SYNC=1 section 02 hf-status sec_hf_status ;;
    *) submit "$mode" ;;
esac
