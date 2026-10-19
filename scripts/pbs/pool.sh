#!/usr/bin/env bash
# Parallel revision pool: runs the chunks.sh inventory in dependency order inside one
# PBS allocation, packing tasks onto its GPUs by measured pilot memory. Each task is the
# unchanged `run_chunk.sh --body` command with exactly one visible GPU. Before the
# walltime it checkpoints every task (USR1, exit 75) and leaves the rest for the next
# pool job; state lives on the shared filesystem so any later pool continues.
# Exit: 0 all done, 75 work remains, 1 only failed/blocked work remains.
set -uo pipefail
ROOT="${RCG_ROOT:-${PBS_O_WORKDIR:?Submit from the repository root}}"
cd "$ROOT" || exit 2
[[ -f scripts/pbs/run_chunk.sh ]] || { echo "Not a repository root: $ROOT" >&2; exit 2; }
export RCG_ROOT="$ROOT"
if [[ "${DRY:-0}" == 1 ]]; then    # preview only: never touch pool state or start tasks
    echo "Pool preview: $(bash scripts/pbs/pool_deps.sh | wc -l) tasks in dependency order (scripts/pbs/pool_deps.sh)"
    exit 0
fi
[[ -n "${PBS_JOBID:-}" || "${RCG_POOL_ALLOW_LOCAL:-0}" == 1 ]] || {
    echo "Run the pool through PBS (scripts/pbs/submit_pool.sh); DRY=1 previews" >&2; exit 2; }

STATE="${RCG_POOL_STATE:-${RCG_RESULTS_ROOT:-$ROOT/assets/results/revision-a100-native}/.pool}"
ADMISSIONS="${RCG_PREFLIGHT_ROOT:-$ROOT/assets/results/a100-native-preflight}/admissions"
JOB="${PBS_JOBID:-local-$$}"
LOGS="${RCG_POOL_LOG_ROOT:-$ROOT/assets/logs/a100/pool}/${JOB%%.*}"
BUDGET="${RCG_POOL_BUDGET_GIB:-70}"          # of 80 GiB; admission policy caps a task at 76
CONTEXT="${RCG_POOL_CONTEXT_GIB:-2}"         # CUDA context + allocator slack per process
MAX_PER_GPU="${RCG_POOL_MAX_PER_GPU:-4}"
LAUNCH_GAP="${RCG_POOL_LAUNCH_GAP:-30}"      # stagger model loads (host RAM, NFS)
POLL="${RCG_POOL_POLL:-15}"
MARGIN="${RCG_POOL_MARGIN:-1800}"            # stop launching/checkpoint this long before walltime
GRACE="${RCG_POOL_GRACE:-1200}"              # wait this long for tasks to commit checkpoints
TASK_CMD="${RCG_POOL_TASK_CMD:-bash scripts/pbs/run_chunk.sh --body}"
QSTAT="${RCG_QSTAT:-$(command -v qstat || echo /opt/pbs/bin/qstat)}"
mkdir -p "$STATE"/{done,failed,lease,exclusive,defer} "$LOGS"
log() { printf '%s pool %s: %s\n' "$(date '+%F %T')" "${JOB%%.*}" "$*"; }

# ---- task table -----------------------------------------------------------
declare -A DEPS=() ADMIT=() LABEL=()
IDS=()
# Split by hand: `read` with IFS=tab would merge the empty deps field of task 1.
while IFS= read -r line; do
    id="${line%%$'\t'*}"; rest="${line#*$'\t'}"
    IDS+=("$id"); DEPS["$id"]="${rest%%$'\t'*}"; ADMIT["$id"]="${rest#*$'\t'}"
done < <(bash scripts/pbs/pool_deps.sh)
while IFS=$'\t' read -r id kind phase item; do
    LABEL["$id"]="$kind $phase $item"
done < <(bash scripts/pbs/chunks.sh)
(( ${#IDS[@]} )) || { echo "Empty inventory" >&2; exit 2; }

# GPU memory a task needs: pilot peak reserved + context, clamped to the budget. A missing
# report (e.g. a probe not yet run) means a whole GPU. Reports are read once per pool;
# the result is returned in COST (no subshell, so the cache persists).
declare -A PEAK_COST=()
cost_of() {
    local id="$1" name="${ADMIT[$1]}" peak
    if [[ -e "$STATE/exclusive/$id" || "$name" == exclusive ]]; then COST="$BUDGET"; return; fi
    if [[ "$name" == light ]]; then COST="$CONTEXT"; return; fi
    if [[ -z "${PEAK_COST[$name]:-}" ]]; then
        peak=$(grep -oE '"peak_reserved_gib": *[0-9.]+' "$ADMISSIONS/$name.json" 2>/dev/null | grep -oE '[0-9.]+$')
        if [[ -n "$peak" ]]; then
            PEAK_COST["$name"]=$(awk -v p="$peak" -v c="$CONTEXT" -v b="$BUDGET" \
                'BEGIN {n = int(p + c + 0.999); print (n > b ? b : n)}')
        else PEAK_COST["$name"]="$BUDGET"; fi
    fi
    COST="${PEAK_COST[$name]}"
}

# ---- GPUs ------------------------------------------------------------------
if [[ -n "${RCG_POOL_GPUS:-}" ]]; then IFS=, read -ra GPUS <<< "$RCG_POOL_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
(( ${#GPUS[@]} )) || { echo "No GPU visible to the pool" >&2; exit 2; }
if [[ -n "${PBS_JOBID:-}" && -z "${RCG_POOL_GPUS:-}" ]]; then
    # Never pack onto GPUs this job was not allocated (the node is shared).
    granted=$("$QSTAT" -f "$PBS_JOBID" 2>/dev/null | awk '$1 == "Resource_List.ngpus" {print $3}')
    [[ "$granted" =~ ^[0-9]+$ && "$granted" == "${#GPUS[@]}" ]] || {
        echo "Visible GPUs (${GPUS[*]}) do not match the allocation (ngpus=${granted:-unknown})" >&2; exit 2; }
fi
declare -A GPU_FREE=() GPU_TASKS=()
for g in "${GPUS[@]}"; do GPU_FREE["$g"]="$BUDGET"; GPU_TASKS["$g"]=0; done
THREADS="${RCG_POOL_THREADS:-$(( ${NCPUS:-8} / (${#GPUS[@]} * MAX_PER_GPU) ))}"
(( THREADS >= 1 )) || THREADS=1

# ---- deadline ----------------------------------------------------------------
to_seconds() { [[ "$1" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] || return 1
               echo $((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]})); }
if [[ -n "${RCG_POOL_SECONDS:-}" ]]; then remaining="$RCG_POOL_SECONDS"
elif [[ -n "${PBS_JOBID:-}" ]]; then
    info=$("$QSTAT" -f "$PBS_JOBID" 2>/dev/null)
    limit=$(to_seconds "$(awk '$1 == "Resource_List.walltime" {print $3}' <<< "$info")") || {
        echo "Cannot read Resource_List.walltime for $PBS_JOBID" >&2; exit 2; }
    used=$(to_seconds "$(awk '$1 == "resources_used.walltime" {print $3}' <<< "$info")") || used=0
    remaining=$((limit - used))
else remaining=$((100 * 3600)); fi
(( remaining > MARGIN )) || { echo "Walltime must exceed the ${MARGIN}s checkpoint margin" >&2; exit 2; }
DEADLINE=$(( $(date +%s) + remaining - MARGIN ))

# ---- leases (atomic mkdir on the shared filesystem) --------------------------
job_alive() {   # dead only on positive evidence; a failing qstat must never free a lease
    local owner="$1" out state
    [[ "$owner" == "$JOB" ]] && return 0
    [[ "$owner" == local-* ]] && return 1
    out=$("$QSTAT" -f "$owner" 2>&1)
    state=$(awk '$1 == "job_state" {print $3}' <<< "$out")
    [[ "$state" == F || "$state" == X ]] && return 1
    [[ -z "$state" && "$out" =~ Unknown\ Job\ Id|[Jj]ob\ has\ finished ]] && return 1
    return 0
}
reclaim_stale() {
    local lease owner
    for lease in "$STATE"/lease/*; do
        [[ -d "$lease" ]] || continue
        owner=$(cut -d' ' -f1 "$lease/owner" 2>/dev/null)
        if [[ -z "$owner" ]]; then
            # Another pool may be between mkdir and writing its owner file.
            [[ -n "$(find "$lease" -maxdepth 0 -mmin +10 2>/dev/null)" ]] || continue
        elif job_alive "$owner"; then continue; fi
        log "reclaim lease $(basename "$lease") from ${owner:-unknown}"; rm -rf "$lease"
    done
}
release_own() {  # at start nothing runs yet: own leases are from a requeued run (same job ID)
    local lease
    for lease in "$STATE"/lease/*; do
        [[ -d "$lease" && "$(cut -d' ' -f1 "$lease/owner" 2>/dev/null)" == "$JOB" ]] || continue
        log "release lease $(basename "$lease") left by an earlier run of this job"; rm -rf "$lease"
    done
}
foreign_leases() {
    local lease
    for lease in "$STATE"/lease/*; do
        [[ -d "$lease" ]] || continue
        [[ "$(cut -d' ' -f1 "$lease/owner" 2>/dev/null)" == "$JOB" ]] || return 0
    done
    return 1
}

# ---- scheduling ----------------------------------------------------------------
declare -A PID_OF=() GPU_OF=() COST_OF=() ATTEMPT_LOG=() BLOCKED=() INCOMPLETE=() RESERVED=() PARKED=()
STOP=0
STOPPING=0     # set once shutdown starts: a killed task is incomplete, not failed
trap 'STOP=1' USR1 TERM INT
deps_done() {
    local d list; IFS=, read -ra list <<< "${DEPS[$1]}"
    for d in "${list[@]}"; do [[ -e "$STATE/done/$d" ]] || return 1; done
}
blocked() {    # a prerequisite failed, directly or transitively (deps have lower IDs)
    local d list; IFS=, read -ra list <<< "${DEPS[$1]}"
    for d in "${list[@]}"; do
        [[ -e "$STATE/failed/$d" || -n "${BLOCKED[$d]:-}" ]] && return 0
    done
    return 1
}
HOLD_ID=""; HOLD_GPU=""   # a GPU drained for a task too big for any current gap
pick_gpu() {    # worst fit: most free memory, under the per-GPU task cap
    local cost="$1" id="$2" best="" g
    for g in "${GPUS[@]}"; do
        [[ "$g" != "$HOLD_GPU" || "$id" == "$HOLD_ID" ]] || continue
        (( GPU_TASKS[$g] < MAX_PER_GPU && GPU_FREE[$g] >= cost )) || continue
        [[ -n "$best" ]] && (( GPU_FREE[$g] <= GPU_FREE[$best] )) && continue
        best="$g"
    done
    [[ -n "$best" ]] && echo "$best"
}
launch() {
    local id="$1" gpu="$2" cost="$3" logf
    logf="$LOGS/chunk_${id}_$(date +%s).log"
    printf '%s %s %s %s\n' "$JOB" "$(hostname)" "$$" "$gpu" > "$STATE/lease/$id/owner"
    log "start $id (${LABEL[$id]}) gpu=$gpu cost=${cost}GiB log=$logf"
    # Background launch keeps the task in this job; only its top shell gets USR1,
    # and common.sh forwards it to the Python worker (a group signal would kill tee).
    # RCG_DEADLINE lets the task defer an uninterruptible stage (epoch evaluation, corpus
    # metrics) it cannot finish before this job checkpoints; it then writes RCG_DEFER_FILE.
    rm -f "$STATE/defer/$id"
    env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
        RCG_DEADLINE="$DEADLINE" RCG_DEFER_FILE="$STATE/defer/$id" \
        RCG_PBS_CHUNK="$id" $TASK_CMD > "$logf" 2>&1 &
    PID_OF["$id"]=$!; GPU_OF["$id"]="$gpu"; COST_OF["$id"]="$cost"; ATTEMPT_LOG["$id"]="$logf"
    GPU_FREE["$gpu"]=$(( GPU_FREE[$gpu] - cost )); GPU_TASKS["$gpu"]=$(( GPU_TASKS[$gpu] + 1 ))
}
finish() {
    local id="$1" rc="$2" gpu="${GPU_OF[$1]}" logf="${ATTEMPT_LOG[$1]}"
    GPU_FREE["$gpu"]=$(( GPU_FREE[$gpu] + COST_OF[$id] )); GPU_TASKS["$gpu"]=$(( GPU_TASKS[$gpu] - 1 ))
    unset "PID_OF[$id]" "GPU_OF[$id]" "COST_OF[$id]"
    if (( rc == 0 )); then
        printf '%s %s %s\n' "$JOB" "$(date +%s)" "$logf" > "$STATE/done/$id"; log "done $id"
    elif (( STOPPING )); then
        log "incomplete $id (exit $rc; resumes from its checkpoint in a later pool)"
    elif (( rc == 75 )) && [[ -e "$STATE/defer/$id" ]]; then
        # Not a retry: the task committed its state and needs more time than this job has.
        PARKED["$id"]=1
        log "deferred $id to the next pool job: $(head -c 300 "$STATE/defer/$id")"
    elif (( rc == 75 )) && (( ++INCOMPLETE[$id] < 3 )); then
        log "incomplete $id outside shutdown (exit 75, ${INCOMPLETE[$id]}x); relaunching"
    elif (( rc != 75 )) && [[ ! -e "$STATE/exclusive/$id" ]] &&
         { (( rc > 128 )) || grep -qE 'CUDA out of memory|OutOfMemoryError' "$logf"; }; then
        # GPU OOM, or a signal the pool did not send (e.g. host-memory cgroup kill, 137).
        touch "$STATE/exclusive/$id"; log "out of memory or killed ($rc) $id; retry alone on a GPU"
    else
        printf '%s %s %s\n' "$JOB" "$rc" "$logf" > "$STATE/failed/$id"
        log "FAILED $id (exit $rc); dependents blocked; see $logf"
    fi
    rm -f "$STATE/defer/$id"
    rm -rf "$STATE/lease/$id"
}
reap() {
    local id rc
    for id in "${!PID_OF[@]}"; do
        kill -0 "${PID_OF[$id]}" 2>/dev/null && continue
        rc=0; wait "${PID_OF[$id]}" || rc=$?
        finish "$id" "$rc"
    done
}
nap() { sleep "$1" & wait $! 2>/dev/null || true; }
descendants() { local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$c"; descendants "$c"; done; }
kill_tree() {   # collect first: killing a parent reparents its children
    local sig="$1" pids; pids=("$2" $(descendants "$2"))
    kill -"$sig" "${pids[@]}" 2>/dev/null || true
}
mark_blocked() {
    local id; BLOCKED=()
    for id in "${IDS[@]}"; do
        [[ -e "$STATE/done/$id" || -e "$STATE/failed/$id" ]] && continue
        blocked "$id" && BLOCKED["$id"]=1
    done
}

release_own
reclaim_stale
log "gpus=${GPUS[*]} budget=${BUDGET}GiB max/gpu=$MAX_PER_GPU threads=$THREADS stop_at=$(date -d "@$DEADLINE" '+%F %T' 2>/dev/null || echo "$DEADLINE")"
last_launch=0
while true; do
    # A signal (qdel/qsig) may already have hit the tasks too: treat their exits as
    # shutdown, not as crashes to retry exclusively.
    (( STOP )) && STOPPING=1
    reap
    [[ ! -e "$STATE/stop" ]] || { (( STOP )) || log "stop file present: draining"; STOP=1; }
    (( $(date +%s) < DEADLINE )) || { (( STOP )) || log "walltime margin reached"; STOP=1; }
    (( STOP )) && break
    mark_blocked
    # Drop a hold whose task can no longer start here (finished, failed, blocked, leased).
    if [[ -n "$HOLD_ID" ]] && [[ -e "$STATE/done/$HOLD_ID" || -e "$STATE/failed/$HOLD_ID" ||
            -n "${BLOCKED[$HOLD_ID]:-}" || -n "${PARKED[$HOLD_ID]:-}" || -d "$STATE/lease/$HOLD_ID" ]]; then
        HOLD_ID=""; HOLD_GPU=""
    fi
    pending=0 ready_waiting=0
    for id in "${IDS[@]}"; do
        [[ -e "$STATE/done/$id" || -e "$STATE/failed/$id" || -n "${PID_OF[$id]:-}" ]] && continue
        [[ -n "${BLOCKED[$id]:-}" || -n "${PARKED[$id]:-}" ]] && continue
        pending=$((pending + 1))
        [[ -d "$STATE/lease/$id" ]] && continue
        deps_done "$id" || continue
        cost_of "$id"; cost="$COST"
        if ! gpu=$(pick_gpu "$cost" "$id"); then
            ready_waiting=1
            if [[ -z "$HOLD_ID" ]]; then     # too big for every gap: drain the emptiest GPU
                emptiest=""
                for g in "${GPUS[@]}"; do
                    [[ -n "$emptiest" ]] && (( GPU_FREE[$g] <= GPU_FREE[$emptiest] )) && continue
                    emptiest="$g"
                done
                if (( cost > GPU_FREE[$emptiest] )); then
                    HOLD_ID="$id"; HOLD_GPU="$emptiest"; log "hold gpu $HOLD_GPU for $id (${cost}GiB)"
                fi
            fi
            continue
        fi
        (( $(date +%s) - last_launch >= LAUNCH_GAP )) || { ready_waiting=1; break; }
        mkdir "$STATE/lease/$id" 2>/dev/null || continue     # another pool took it
        # Listing (not stat) defeats NFS negative-lookup caching of another pool's marker.
        if ls "$STATE/done" | grep -qx "$id"; then rmdir "$STATE/lease/$id"; continue; fi
        launch "$id" "$gpu" "$cost"; last_launch=$(date +%s)
        [[ "$id" != "$HOLD_ID" ]] || { HOLD_ID=""; HOLD_GPU=""; }
    done
    if (( ${#PID_OF[@]} == 0 && ready_waiting == 0 )); then
        (( pending == 0 )) && break                                   # all done or blocked
        foreign_leases || { reclaim_stale; foreign_leases || break; }  # nothing can progress here
    fi
    nap "$POLL"
done

# ---- stop: checkpoint running tasks, release their leases ---------------------------
if (( ${#PID_OF[@]} )); then
    STOPPING=1
    log "stopping ${#PID_OF[@]} task(s): ${!PID_OF[*]}"
    for id in "${!PID_OF[@]}"; do kill -USR1 "${PID_OF[$id]}" 2>/dev/null || true; done
    stop_by=$(( $(date +%s) + GRACE ))
    while (( ${#PID_OF[@]} )) && (( $(date +%s) < stop_by )); do reap; nap 5; done
    # Stragglers: signal the whole tree. run_step starts Python with setsid, so it would
    # outlive its shell and keep the GPU while another pool re-ran the same task.
    for id in "${!PID_OF[@]}"; do kill_tree TERM "${PID_OF[$id]}"; done
    (( ${#PID_OF[@]} == 0 )) || nap 30
    reap      # tasks that exited on TERM are finished before KILL walks the rest
    for id in "${!PID_OF[@]}"; do
        kill_tree KILL "${PID_OF[$id]}"
        rc=0; wait "${PID_OF[$id]}" || rc=$?
        finish "$id" "$rc"
    done
fi

mark_blocked
done_n=$(find "$STATE/done" -type f | wc -l); failed_n=$(find "$STATE/failed" -type f | wc -l)
log "summary: ${done_n}/${#IDS[@]} done, ${failed_n} failed, ${#BLOCKED[@]} blocked"
# last_exit tells the cron keeper whether to continue (75) or leave it to the user (0/1).
leave() { printf '%s %s %s\n' "$1" "$JOB" "$(date +%s)" > "$STATE/last_exit"; exit "$1"; }
(( done_n == ${#IDS[@]} )) && leave 0
for id in "${IDS[@]}"; do      # runnable work left for a later pool?
    [[ -e "$STATE/done/$id" || -e "$STATE/failed/$id" || -n "${BLOCKED[$id]:-}" ]] && continue
    leave 75
done
log "only failed or blocked tasks remain; fix them, remove $STATE/failed/<id>, then submit_pool.sh"
leave 1
