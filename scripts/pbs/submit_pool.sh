#!/usr/bin/env bash
# Submit one parallel pool job (scripts/pbs/pool.pbs). Used by the user (first start),
# by a finishing pool and by pool_keeper.sh (both --continue: keep the error count, retry).
#   bash scripts/pbs/submit_pool.sh [--gpus 1..4] [--ncpus N] [--mem-gb N] [--jobs 1..4] [--preview] [--continue]
# The environment is passed explicitly with -v and recorded in the pool state so the
# keeper (cron, minimal environment) and continuations reuse it; -V is never used.
# Stop the whole campaign with `touch <state>/stop` (the running pool drains, nothing is
# resubmitted); remove that file and run this script again to resume.
set -euo pipefail
mode=submit; gpus=""; ncpus=""; mem=""; jobs=""; continuing=0
while (( $# )); do
    case "$1" in
        --gpus) [[ $# -ge 2 ]] || { echo "Missing value for --gpus" >&2; exit 2; }; gpus="$2"; shift 2 ;;
        --ncpus) [[ $# -ge 2 ]] || { echo "Missing value for --ncpus" >&2; exit 2; }; ncpus="$2"; shift 2 ;;
        --mem-gb) [[ $# -ge 2 ]] || { echo "Missing value for --mem-gb" >&2; exit 2; }; mem="$2"; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || { echo "Missing value for --jobs" >&2; exit 2; }; jobs="$2"; shift 2 ;;
        --preview) mode=preview; shift ;;
        --continue) continuing=1; shift ;;
        *) echo "Usage: $0 [--gpus 1..4] [--ncpus N] [--mem-gb N] [--jobs 1..4] [--preview] [--continue]" >&2; exit 2 ;;
    esac
done
cd "${RCG_ROOT:-$PWD}"
[[ -f scripts/pbs/pool.pbs ]] || { echo "Run from the repository root or set RCG_ROOT" >&2; exit 2; }
ROOT="$PWD"
export PATH="$PATH:/opt/pbs/bin"
STATE="${RCG_POOL_STATE:-${RCG_RESULTS_ROOT:-$ROOT/assets/results/revision-a100-native}/.pool}"
RECORD="$STATE/submit.env"
# Settings that must survive into continuations/keeper submissions. Root overrides are
# also passed to the job; the keeper's cron line must then export RCG_POOL_STATE too.
RECORDED=(RCG_PYTHON HF_HOME NLTK_DATA POOL_GPUS POOL_NCPUS POOL_MEM_GB POOL_JOBS RCG_PBS_WALLTIME RCG_PBS_QUEUE RCG_PBS_PROJECT
          RCG_RESULTS_ROOT RCG_PREFLIGHT_ROOT RCG_POOL_STATE RCG_LOG_ROOT RCG_JUDGE_ROOT)
PASSED_ROOTS=(RCG_RESULTS_ROOT RCG_PREFLIGHT_ROOT RCG_POOL_STATE RCG_LOG_ROOT RCG_JUDGE_ROOT)

if [[ -e "$STATE/stop" ]]; then
    echo "Stop file present ($STATE/stop): remove it to resume the campaign" >&2
    (( continuing )) && exit 0 || exit 2
fi
# Previous submission settings fill anything not given (keeper/continuation path).
if [[ -f "$RECORD" ]]; then
    while IFS='=' read -r key value; do
        [[ " ${RECORDED[*]} " == *" $key "* ]] || continue
        [[ -n "${!key:-}" ]] || printf -v "$key" '%s' "$value"
    done < "$RECORD"
fi
gpus="${gpus:-${POOL_GPUS:-2}}"
[[ "$gpus" =~ ^[1-4]$ ]] || { echo "--gpus must be 1..4 (account cap: 4 running GPUs)" >&2; exit 2; }
# CPUs default to 8 per GPU; fewer lets the job start on a busy node (results unchanged:
# the pool sizes each task's thread count from the allocation).
ncpus="${ncpus:-${POOL_NCPUS:-$((8 * gpus))}}"
[[ "$ncpus" =~ ^[1-9][0-9]?$ ]] || { echo "--ncpus must be 1..99" >&2; exit 2; }
POOL_NCPUS="$ncpus"
# Host RAM defaults to 16 GB per GPU: whole 2-GPU pools peaked at 14-17 GB (PBS
# resources_used.mem of 20400/20401, which ran 7B LoRA and ICL; 21018's cgroup held
# 6.5 GB anonymous, the rest reclaimable file cache), and a smaller request
# lets the job start on a node whose memory other jobs have mostly reserved. Only an
# explicit --mem-gb is recorded, so the default follows a later --gpus change.
POOL_MEM_GB="${mem:-${POOL_MEM_GB:-}}"
mem="${POOL_MEM_GB:-$((16 * gpus))}"
[[ "$mem" =~ ^[1-9][0-9]{0,3}$ ]] || { echo "--mem-gb must be 1..9999" >&2; exit 2; }
# Concurrent pool jobs (leases make them safe). Each queues its own continuation and the
# keeper refills up to this many; queued+running GPUs count against the account quota.
POOL_JOBS="${jobs:-${POOL_JOBS:-1}}"
[[ "$POOL_JOBS" =~ ^[1-4]$ ]] || { echo "--jobs must be 1..4" >&2; exit 2; }
RCG_PYTHON="${RCG_PYTHON:-}"
[[ -n "$RCG_PYTHON" && "$RCG_PYTHON" == /* ]] || { echo "Set RCG_PYTHON to the absolute rcg-a100 python" >&2; exit 2; }
[[ "$mode" == preview || -x "$RCG_PYTHON" ]] || { echo "RCG_PYTHON is not executable: $RCG_PYTHON" >&2; exit 2; }
HF_HOME="${HF_HOME:-$ROOT/assets/cache/huggingface}"
NLTK_DATA="${NLTK_DATA:-$ROOT/assets/cache/nltk_data}"
RCG_PBS_WALLTIME="${RCG_PBS_WALLTIME:-100:00:00}"
RCG_PBS_QUEUE="${RCG_PBS_QUEUE:-a100q}"
RCG_PBS_PROJECT="${RCG_PBS_PROJECT:-20260203}"
POOL_GPUS="$gpus"
[[ "$RCG_PBS_WALLTIME" =~ ^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "Walltime must be HH:MM:SS" >&2; exit 2; }
passed="RCG_ROOT=$ROOT,RCG_PYTHON=$RCG_PYTHON,HF_HOME=$HF_HOME,NLTK_DATA=$NLTK_DATA,WANDB_MODE=offline,RCG_TARGET_GPU=a100_80gb"
for key in "${PASSED_ROOTS[@]}"; do [[ -z "${!key:-}" ]] || passed+=",$key=${!key}"; done
for value in "$ROOT" "$RCG_PYTHON" "$HF_HOME" "$NLTK_DATA" $(for key in "${PASSED_ROOTS[@]}"; do echo "${!key:-}"; done); do
    [[ "$value" != *,* ]] || { echo "Paths passed with qsub -v cannot contain commas: $value" >&2; exit 2; }
done

source scripts/pbs/check_judge_inputs.sh
[[ "$mode" == preview ]] || check_judge_inputs || exit 2

# Submitted as a command (qsub -- bash pool.pbs), not as a spooled script: on aimc-gna1
# the MoM spool directory lost its search bit (2026-10-09 reboot) and every spooled
# script exits 126 "Permission denied". #PBS lines are not read in this form, so every
# directive is given here.
command_args=(qsub -N rcg_pool -q "$RCG_PBS_QUEUE" -P "$RCG_PBS_PROJECT"
    -l "select=1:ncpus=$ncpus:ngpus=$gpus:mem=${mem}gb" -l "walltime=$RCG_PBS_WALLTIME"
    -j oe -v "$passed" -- /bin/bash "$ROOT/scripts/pbs/pool.pbs")
if [[ "$mode" == preview ]]; then printf '%q ' "${command_args[@]}"; printf '\n'; exit 0; fi

# At most POOL_JOBS pool jobs. Plain qselect lists every unfinished job (Q/H/W/R/S/E/T).
# A continuation ignores only itself; IDs are compared by number so a server suffix
# difference cannot matter.
if ! listed=$(qselect -u "${USER:-$(id -un)}" -N rcg_pool 2>&1); then
    echo "qselect failed ($listed); not submitting without knowing the queue" >&2; exit 3
fi
self="${PBS_JOBID:-none}"
others=$(awk -F. -v self="${self%%.*}" 'NF && $1 != self' <<< "$listed")
if (( $(grep -c . <<< "$others") >= POOL_JOBS )); then
    echo "$POOL_JOBS pool job(s) already queued/running: $(tr '\n' ' ' <<< "$others")- not submitting another" >&2
    exit 0
fi
mkdir -p "$STATE" "${RCG_POOL_LOG_ROOT:-$ROOT/assets/logs/a100/pool}"   # keeper's cron line logs there
for key in "${RECORDED[@]}"; do [[ -z "${!key:-}" ]] || printf '%s=%s\n' "$key" "${!key}"; done > "$RECORD"
(( continuing )) || rm -f "$STATE/errors"      # a user submission re-arms after a fix
tries=1; (( continuing )) && tries=3
for ((i = 1; i <= tries; i++)); do
    if submitted=$("${command_args[@]}" 2>&1); then
        rm -f "$STATE/last_exit"      # a queued pool re-arms the keeper (e.g. after a fix)
        # One mark per job: pool.pbs removes its own when it starts; the keeper counts a job
        # that left the queue with its mark still present as a setup error (it died before
        # its script ran).
        id=$(grep -oE '^[0-9]+\.[^[:space:]]+' <<< "$submitted" | tail -1 || true)
        mkdir -p "$STATE/unstarted.d"; [[ -z "$id" ]] || touch "$STATE/unstarted.d/$id"
        echo "$submitted rcg_pool gpus=$gpus"
        exit 0
    fi
    echo "qsub failed (try $i/$tries): $submitted" >&2
    (( i < tries )) && sleep "${RCG_SUBMIT_RETRY_SLEEP:-60}"
done
echo "Could not submit a pool job; the cron keeper (pool_keeper.sh) retries every 30 minutes" >&2
exit 3
