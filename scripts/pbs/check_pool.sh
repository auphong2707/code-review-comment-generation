#!/usr/bin/env bash
# Bash-only checks of the parallel pool over the real 226-task inventory with fake tasks,
# fake GPUs and fake qstat: dependency order, memory/count caps, stop -> resume, failure
# isolation, OOM retry, stale/live leases and killing a stuck task's process tree.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
scratch=$(mktemp -d)
trap 'pkill -f "$scratch/" 2>/dev/null || true; rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/pre/admissions"
export RCG_ROOT="$ROOT" RCG_PREFLIGHT_ROOT="$scratch/pre" RCG_QSTAT="$scratch/bin/qstat" RCG_POOL_ALLOW_LOCAL=1
export RCG_POOL_GPUS=0,1 RCG_POOL_POLL=0.05 RCG_POOL_LAUNCH_GAP=0 RCG_POOL_MARGIN=1 RCG_POOL_GRACE=2
export RCG_POOL_TASK_CMD="bash $scratch/task.sh" EVENTS="$scratch/events" FAKE="$scratch"
unset PBS_JOBID CUDA_VISIBLE_DEVICES

# Pilot peaks: training 30 GiB (cost 32, two per GPU), ICL 10, judges 5.
bash scripts/pbs/pool_deps.sh > "$scratch/deps"
for name in $(cut -f3 "$scratch/deps" | sort -u); do
    case "$name" in
        train_*) peak=30 ;; icl_*) peak=10 ;; judge_*) peak=5 ;; *) continue ;;
    esac
    printf '{"admitted": true, "peak_reserved_gib": %s}\n' "$peak" > "$scratch/pre/admissions/$name.json"
done
cat > "$scratch/bin/qstat" <<'EOF'
#!/usr/bin/env bash
state=$(awk -v j="$2" '$1 == j {print $2}' "$FAKE/jobstates" 2>/dev/null)
if [[ -z "$state" ]]; then
    if grep -qx "$2" "$FAKE/unknown" 2>/dev/null; then echo "qstat: Unknown Job Id $2" >&2; exit 153; fi
    echo "qstat: cannot connect to server aimc-hn2 (errno=111)" >&2; exit 1
fi
printf 'Job Id: %s\n    job_state = %s\n' "$2" "$state"
cat "$FAKE/qstat-extra" 2>/dev/null || true
EOF
chmod +x "$scratch/bin/qstat"
# Fake task: logs start/end with its GPU; behaviour per ID from $FAKE/mode/<id>.
cat > "$scratch/task.sh" <<'EOF'
#!/usr/bin/env bash
id="$RCG_PBS_CHUNK"; mode=$(cat "$FAKE/mode/$id" 2>/dev/null || cat "$FAKE/mode/default" 2>/dev/null || echo ok)
echo "start $id $CUDA_VISIBLE_DEVICES $(date +%s%N)" >> "$EVENTS"
end() { echo "end $id $CUDA_VISIBLE_DEVICES $(date +%s%N) $1" >> "$EVENTS"; exit "$1"; }
case "$mode" in
    ok) sleep 0.3; end 0 ;;
    ok2) sleep 2; end 0 ;;
    fail) end 3 ;;
    oom-once) if [[ ! -e "$FAKE/oom-$id" ]]; then touch "$FAKE/oom-$id"; echo "torch.OutOfMemoryError: CUDA out of memory"; end 1; fi; end 0 ;;
    slow) trap 'end 75' USR1; sleep 30 & wait $!; end 0 ;;
    stuck) trap '' USR1; setsid sleep 1000 & echo "$!" > "$FAKE/orphan-$id"; sleep 1000 & wait $!; end 0 ;;
    # USR1 before common.sh installs its trap (require_gpu): the default action kills it (138).
    notrap) sleep 30 & wait $!; end 0 ;;
    killed-once) if [[ ! -e "$FAKE/killed-$id" ]]; then touch "$FAKE/killed-$id"
                     echo "end $id $CUDA_VISIBLE_DEVICES $(date +%s%N) 137" >> "$EVENTS"; kill -9 $$; fi; end 0 ;;
    always75) end 75 ;;
    # The task cannot finish an uninterruptible stage before RCG_DEADLINE: commit, mark, 75.
    defer-once) if [[ ! -e "$FAKE/deferred-$id" ]]; then touch "$FAKE/deferred-$id"
                    echo "$RCG_DEADLINE" > "$FAKE/deadline-$id"; echo "eval needs time" > "$RCG_DEFER_FILE"; end 75; fi; end 0 ;;
esac
EOF
run_pool() { local rc=0; RCG_POOL_STATE="$scratch/state" RCG_POOL_LOG_ROOT="$scratch/logs" bash scripts/pbs/pool.sh > "$scratch/pool.out" 2>&1 || rc=$?; echo "$rc"; }
reset() { rm -rf "$scratch/state" "$scratch/logs" "$scratch/mode" "$EVENTS" "$scratch"/oom-* "$scratch"/killed-* "$scratch"/deferred-* "$scratch"/deadline-* "$scratch/jobstates" "$scratch/unknown" "$scratch/qstat-extra"; mkdir -p "$scratch/mode"; }
fail() { echo "FAIL: $*" >&2; tail -20 "$scratch/pool.out" >&2; exit 1; }

# Replays events: every task starts after all its prerequisites ended successfully, and
# per-GPU memory (costs from the pool log) and task count never exceed 70 GiB / 4.
check_events() {
    awk -F '\t' -v events="$EVENTS" -v log_file="$scratch/pool.out" '
        FILENAME == ARGV[1] {deps[$1] = $2; next}
        END {
            while ((getline line < log_file) > 0) {     # POSIX awk: no match() captures
                n = split(line, w, " "); id = ""; c = ""
                for (i = 1; i < n; i++) if (w[i] == "start" && w[i + 1] ~ /^[0-9]+$/) id = w[i + 1]
                for (i = 1; i <= n; i++) if (w[i] ~ /^cost=[0-9]+GiB$/) { c = w[i]; gsub(/cost=|GiB/, "", c) }
                if (id != "" && c != "") cost[id, ++attempts[id]] = c     # one cost per attempt
            }
            while ((getline line < events) > 0) {
                split(line, f, " ")
                if (f[1] == "start") {
                    # A SIGKILLed attempt writes no end event; its relaunch implies it ended.
                    if (f[2] in active) { mem[gpu_of[f[2]]] -= active[f[2]]; cnt[gpu_of[f[2]]]--; delete active[f[2]] }
                    n = split(deps[f[2]], d, ",")
                    for (i = 1; i <= n; i++) if (!(d[i] in ok)) { print "order: " f[2] " before " d[i]; bad = 1 }
                    active[f[2]] = cost[f[2], ++seen[f[2]]]; gpu_of[f[2]] = f[3]
                    mem[f[3]] += active[f[2]]; cnt[f[3]]++
                    if (mem[f[3]] > 70 || cnt[f[3]] > 4) { print "cap: gpu " f[3] " mem " mem[f[3]] " count " cnt[f[3]]; bad = 1 }
                } else {
                    mem[f[3]] -= active[f[2]]; cnt[f[3]]--; delete active[f[2]]
                    if (f[5] == 0) ok[f[2]] = 1
                }
            }
            exit bad
        }' "$scratch/deps" /dev/null || fail "event replay"
}

# 1. Full run: everything completes in dependency order within the caps.
reset
[[ $(run_pool) == 0 ]] || fail "full run exit"
[[ $(ls "$scratch/state/done" | wc -l) -eq 226 ]] || fail "not all done"
[[ -z "$(ls -A "$scratch/state/lease")" ]] || fail "leases left"
check_events
grep -q 'start 1 .*cost=70GiB' "$scratch/pool.out" || fail "chunk 1 must be exclusive"
awk '$1 == "start"' "$EVENTS" | awk '{print $3}' | sort -u | tr '\n' ' ' | grep -q '0 1' || fail "both GPUs used"
# Packing really happened: some GPU ran two 32 GiB training tasks at once.
awk '$1 == "start" {c[$3]++; if (c[$3] > m) m = c[$3]} $1 == "end" {c[$3]--} END {exit !(m >= 2)}' "$EVENTS" || fail "no packing"

# 2. Stop at the deadline: running tasks checkpoint (exit 75), leases released, no
#    failures; a second pool resumes and finishes.
reset
echo slow > "$scratch/mode/default"
RCG_POOL_SECONDS=4 rc=$(RCG_POOL_SECONDS=4 run_pool)
[[ "$rc" == 75 ]] || fail "stopped pool must exit 75 (got $rc)"
[[ -z "$(ls -A "$scratch/state/lease")" && -z "$(ls -A "$scratch/state/failed")" ]] || fail "stop left leases/failures"
grep -q 'incomplete' "$scratch/pool.out" || fail "no checkpointed task"
rm "$scratch/mode/default"
[[ $(run_pool) == 0 && $(ls "$scratch/state/done" | wc -l) -eq 226 ]] || fail "resume did not finish"

# 3. A failure blocks only its dependents: D sweep lr0 fails -> D official, gate D,
#    prepare J and revision judging/summaries blocked; llama-legacy still completes.
reset
id_of() { awk -F '\t' -v l="$1" '$4 == l {print $1}' <(bash scripts/pbs/chunks.sh); }
sweep=$(id_of d_trans_review_qlora_sweep_lr0); official=$(id_of d_trans_review_qlora_official)
echo fail > "$scratch/mode/$sweep"
[[ $(run_pool) == 1 ]] || fail "only-blocked pool must exit 1"
[[ -e "$scratch/state/failed/$sweep" && ! -e "$scratch/state/done/$official" ]] || fail "failure not isolated"
grep -q "^start $official " "$EVENTS" && fail "dependent of a failed task started"
[[ -e "$scratch/state/done/$(id_of c_qwen15_acra_by_ls_pt_10shot_bm25)" ]] || fail "independent work blocked"
for id in $(awk -F '\t' '$2 == "judge" && $3 == "llama-legacy" {print $1}' <(bash scripts/pbs/chunks.sh)); do
    [[ -e "$scratch/state/done/$id" ]] || fail "llama-legacy $id blocked"
done
# Blocked: D TR official, gate D, prepare J, 2 x (48 revision judge shards + summary).
grep -q "summary: 124/226 done, 1 failed, 101 blocked" "$scratch/pool.out" || fail "unexpected done/blocked counts"
# Retrying after a fix: removing the failure marker lets the pool finish.
rm "$scratch/mode/$sweep" "$scratch/state/failed/$sweep"
[[ $(run_pool) == 0 ]] || fail "retry after fix"

# 4. Out of memory once -> retried alone on a GPU (cost 70) and completes.
reset
echo oom-once > "$scratch/mode/2"
echo killed-once > "$scratch/mode/3"      # host-memory cgroup kill (SIGKILL, exit 137)
[[ $(run_pool) == 0 ]] || fail "oom run exit"
[[ -e "$scratch/state/exclusive/2" ]] && grep -q 'start 2 .*cost=70GiB' "$scratch/pool.out" || fail "oom retry not exclusive"
[[ -e "$scratch/state/exclusive/3" ]] && grep -q 'start 3 .*cost=70GiB' "$scratch/pool.out" || fail "killed task not retried alone"
check_events

# 4c. A whole-GPU task is not starved: with 2,3,4 (32 GiB, slow) on both GPUs, task 5
#     (exclusive, 70 GiB) holds the emptiest GPU; nothing else starts there before it.
reset
for id in 2 3 4; do echo ok2 > "$scratch/mode/$id"; done
mkdir -p "$scratch/state/exclusive"; touch "$scratch/state/exclusive/5"
[[ $(run_pool) == 0 ]] || fail "hold run exit"
held=$(grep -oE 'hold gpu [0-9]+ for 5 ' "$scratch/pool.out" | awk '{print $3}' || true)
[[ -n "$held" ]] || fail "no GPU held for the whole-GPU task"
awk -v g="gpu=$held" '/hold gpu .* for 5 / {on = 1; next} on && / start 5 / {exit} on && / start [0-9]+ / && index($0, g) {bad = 1} END {exit bad}' \
    "$scratch/pool.out" || fail "a task started on the held GPU before the whole-GPU task"
check_events

# 4b. A task that keeps exiting 75 outside shutdown is failed after 3 tries (no endless
#     relaunch/resubmission); its dependents are blocked and the pool exits 1.
reset
echo always75 > "$scratch/mode/2"
[[ $(run_pool) == 1 ]] || fail "repeated exit 75 must end as blocked"
[[ -e "$scratch/state/failed/2" && $(grep -c '^start 2 ' "$EVENTS") == 3 ]] || fail "exit-75 loop not capped at 3"

# 4d. A task that defers an uninterruptible stage it cannot finish before RCG_DEADLINE is
#     parked: not relaunched in this job, not a retry or failure; the next pool finishes it.
reset
echo defer-once > "$scratch/mode/2"
[[ $(run_pool) == 75 ]] || fail "a parked task must leave resumable work (75)"
[[ $(grep -c '^start 2 ' "$EVENTS") == 1 && -z "$(ls -A "$scratch/state/failed")" ]] || fail "deferred task relaunched or failed"
grep -q 'deferred 2 to the next pool job: eval needs time' "$scratch/pool.out" || fail "deferral not logged"
[[ "$(cat "$scratch/deadline-2")" =~ ^[0-9]+$ && ! -e "$scratch/state/defer/2" ]] || fail "deadline not passed or marker left"
[[ $(run_pool) == 0 && -e "$scratch/state/done/2" ]] || fail "deferred task did not finish in the next pool"

# 5. Leases: one from a finished job is reclaimed; one held by a running job is respected.
reset
mkdir -p "$scratch/state/lease/1" "$scratch/state/lease/2"
echo '900.fake host 1 0' > "$scratch/state/lease/1/owner"
echo '901.fake host 1 0' > "$scratch/state/lease/2/owner"
printf '900.fake F\n901.fake R\n' > "$scratch/jobstates"
rc=$(RCG_POOL_SECONDS=3 run_pool)
[[ "$rc" == 75 ]] || fail "live foreign lease: expected 75, got $rc"
grep -q 'reclaim lease 1 from 900.fake' "$scratch/pool.out" || fail "stale lease not reclaimed"
grep -q '^start 2 ' "$EVENTS" && fail "task leased by a running job was started"
[[ -d "$scratch/state/lease/2" ]] || fail "live lease removed"
# qstat errors are not evidence: only F/X or "Unknown Job Id" free a lease. An owner-less
# lease (another pool between mkdir and its owner write) is kept until 10 minutes old.
reset
for id in 1 2 3 4; do mkdir -p "$scratch/state/lease/$id"; done
echo '902.fake host 1 0' > "$scratch/state/lease/1/owner"; echo 902.fake > "$scratch/unknown"
echo '903.fake host 1 0' > "$scratch/state/lease/2/owner"            # qstat cannot connect
touch -d '-20 minutes' "$scratch/state/lease/4"                       # 3 young, 4 old, no owner
rc=$(RCG_POOL_SECONDS=3 run_pool)
grep -q 'reclaim lease 1 from 902.fake' "$scratch/pool.out" || fail "unknown job lease kept"
[[ -d "$scratch/state/lease/2" && -d "$scratch/state/lease/3" ]] || fail "lease freed without evidence"
[[ ! -d "$scratch/state/lease/4" ]] || fail "old owner-less lease kept"
# A requeued job keeps its PBS job ID: its old leases are released at start, and a
# pool refuses GPUs that do not match its allocation.
reset
mkdir -p "$scratch/state/lease/1"; echo '700.fake host 1 0' > "$scratch/state/lease/1/owner"
printf '700.fake R\n' > "$scratch/jobstates"; echo '    Resource_List.ngpus = 2' > "$scratch/qstat-extra"
rc=$(PBS_JOBID=700.fake RCG_POOL_SECONDS=3 run_pool)
grep -q 'release lease 1 left by an earlier run of this job' "$scratch/pool.out" && grep -q '^start 1 ' "$EVENTS" ||
    fail "requeued job's own lease not released"
reset
printf '701.fake R\n' > "$scratch/jobstates"; echo '    Resource_List.ngpus = 1' > "$scratch/qstat-extra"
rc=$(unset RCG_POOL_GPUS; CUDA_VISIBLE_DEVICES=0,1 PBS_JOBID=701.fake RCG_POOL_SECONDS=3 run_pool)
[[ "$rc" == 2 ]] && grep -q 'do not match the allocation' "$scratch/pool.out" || fail "GPU/allocation mismatch accepted"

# 6. A task that ignores USR1 is killed with its whole tree (setsid child included) and is
#    left incomplete, not failed.
reset
echo stuck > "$scratch/mode/1"
rc=$(RCG_POOL_SECONDS=3 run_pool)
[[ "$rc" == 75 ]] || fail "stuck task: expected 75, got $rc"
orphan=$(cat "$scratch/orphan-1")
sleep 0.5
kill -0 "$orphan" 2>/dev/null && fail "setsid grandchild survived"
[[ ! -e "$scratch/state/failed/1" && -z "$(ls -A "$scratch/state/lease")" ]] || fail "stuck task marked failed or lease kept"
# 6b. USR1 arriving before common.sh installs its trap (during require_gpu) kills the
#     task shell (exit 138); during shutdown that is incomplete, never failed.
reset
echo notrap > "$scratch/mode/1"
rc=$(RCG_POOL_SECONDS=3 run_pool)
[[ "$rc" == 75 && ! -e "$scratch/state/failed/1" && ! -e "$scratch/state/exclusive/1" ]] ||
    fail "early-signal task: rc $rc"
grep -q 'incomplete 1 (exit 138' "$scratch/pool.out" || fail "early-signal exit not recorded as incomplete"

# ---- phase 03: continuation submit, launcher guards, cron keeper ----------------------
cat > "$scratch/bin/qsub" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE/qsub-calls"
rc=$(cat "$FAKE/qsub-rc" 2>/dev/null || echo 0)
(( rc == 0 )) && echo "777.fake" || echo "qsub: would exceed queue generic's per-user limit"
exit "$rc"
EOF
cat > "$scratch/bin/qselect" <<'EOF'
#!/usr/bin/env bash
[[ ! -e "$FAKE/qselect-fail" ]] || { echo "qselect: cannot connect to server" >&2; exit 1; }
cat "$FAKE/qselect" 2>/dev/null || true
EOF
chmod +x "$scratch/bin/qsub" "$scratch/bin/qselect"
export PATH="$scratch/bin:$PATH" RCG_SUBMIT_RETRY_SLEEP=0
export RCG_POOL_STATE="$scratch/state" RCG_POOL_LOG_ROOT="$scratch/logs"
submit() { local rc=0; bash scripts/pbs/submit_pool.sh "$@" > "$scratch/submit.out" 2>&1 || rc=$?; echo "$rc"; }
calls() { [[ -f "$scratch/qsub-calls" ]] && wc -l < "$scratch/qsub-calls" || echo 0; }

# 7. Launcher: input check, argument guards, one pool at a time, explicit environment.
reset; rm -f "$scratch/qsub-calls" "$scratch/qselect" "$scratch/qsub-rc"
[[ $(RCG_JUDGE_MODEL_ROOT="$scratch/no-models" RCG_PYTHON=/bin/true submit) == 2 && $(calls) == 0 ]] || fail "missing judge weights must block qsub"
grep -q 'Missing judge input' "$scratch/submit.out" || fail "missing-weights message"
export RCG_SKIP_INPUT_CHECK=1
[[ $(RCG_PYTHON=/bin/true submit --gpus 5) == 2 ]] || fail "--gpus 5 accepted"
[[ $(RCG_PYTHON=relative/python submit) == 2 ]] || fail "relative RCG_PYTHON accepted"
echo "555.fake" > "$scratch/qselect"
[[ $(RCG_PYTHON=/bin/true submit) == 0 && $(calls) == 0 ]] || fail "second pool submitted"
rm "$scratch/qselect"
[[ $(RCG_PYTHON=/bin/true submit --gpus 3) == 0 && $(calls) == 1 ]] || fail "first submission"
call=$(cat "$scratch/qsub-calls")
[[ "$call" == *"-N rcg_pool -q a100q -P 20260203 "* && "$call" == *"select=1:ncpus=24:ngpus=3:mem=192gb"* && "$call" == *"walltime=100:00:00"* ]] || fail "resources: $call"
[[ "$call" == *"RCG_PYTHON=/bin/true"* && "$call" == *"RCG_ROOT=$ROOT"* && "$call" != *" -V "* ]] || fail "environment: $call"
# Command form (no spooled script: a broken MoM spool exits 126), so -j oe is explicit.
[[ "$call" == *" -j oe -v "*" -- /bin/bash $ROOT/scripts/pbs/pool.pbs" ]] || fail "command-form submission: $call"
[[ "$(cat "$scratch/state/unstarted")" == 777.fake ]] || fail "submission must mark the job unstarted"
grep -qx 'POOL_GPUS=3' "$scratch/state/submit.env" || fail "submission not recorded"

# 8. A pool that stops at walltime queues its continuation with the recorded settings;
#    a refused qsub is retried three times and then left to the keeper.
reset; rm -f "$scratch/qsub-calls"; mkdir -p "$scratch/state"
printf 'RCG_PYTHON=/bin/true\nHF_HOME=/h\nNLTK_DATA=/n\nPOOL_GPUS=2\n' > "$scratch/state/submit.env"
echo slow > "$scratch/mode/default"
cp scripts/pbs/pool.pbs "$scratch/spooled-pool.pbs"
echo "600.fake" > "$scratch/qselect"     # the finishing pool itself is still listed
rc=0; (cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PYTHON=/bin/true RCG_POOL_SECONDS=4 \
    PBS_JOBID=600.fake bash spooled-pool.pbs) > "$scratch/pool.out" 2>&1 || rc=$?
[[ "$rc" == 75 && $(calls) == 1 ]] || fail "continuation not submitted (rc $rc, calls $(calls))"
grep -q "ngpus=2:mem=128gb" "$scratch/qsub-calls" || fail "continuation resources: $(cat "$scratch/qsub-calls")"
[[ "$(cut -d' ' -f1 "$scratch/state/last_exit" 2>/dev/null)" == "" ]] || fail "continuation must clear last_exit"
[[ "$(cat "$scratch/state/unstarted")" == 777.fake ]] || fail "continuation must be marked unstarted"
# ...but another pool job (different number, any server suffix) blocks it.
printf '600.aimc-hn2\n599.aimc-hn2\n' > "$scratch/qselect"; rm -f "$scratch/qsub-calls"
[[ $(PBS_JOBID=600.fake submit --continue) == 0 && $(calls) == 0 ]] || fail "continuation ignored another pool"
rm "$scratch/qselect"
echo 1 > "$scratch/qsub-rc"; rm -f "$scratch/qsub-calls"; echo 601.fake > "$scratch/state/unstarted"
rc=0; (cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PYTHON=/bin/true RCG_POOL_SECONDS=4 \
    PBS_JOBID=601.fake bash spooled-pool.pbs) > "$scratch/pool.out" 2>&1 || rc=$?
[[ "$rc" == 75 && $(calls) == 3 ]] || fail "refused continuation: rc $rc calls $(calls)"
[[ ! -e "$scratch/state/unstarted" && "$(cat "$scratch/state/started")" == 601.fake ]] || fail "a started pool must clear its mark and record its ID"
[[ "$(cut -d' ' -f1 "$scratch/state/last_exit")" == 75 ]] || fail "refused continuation must leave last_exit 75"
rm "$scratch/qsub-rc" "$scratch/mode/default"

# 9. Keeper: submits only for resumable work with no pool job present; respects its lock.
keeper() { rm -f "$scratch/qsub-calls"; bash scripts/pbs/pool_keeper.sh > "$scratch/keeper.out" 2>&1; calls; }
reset; [[ $(keeper) == 0 ]] || fail "keeper without any submission"
mkdir -p "$scratch/state"; printf 'RCG_PYTHON=/bin/true\nPOOL_GPUS=2\n' > "$scratch/state/submit.env"
echo "0 9.fake 1" > "$scratch/state/last_exit"; [[ $(keeper) == 0 ]] || fail "keeper resubmitted a finished campaign"
echo "1 9.fake 1" > "$scratch/state/last_exit"; [[ $(keeper) == 0 ]] || fail "keeper resubmitted a blocked campaign"
echo "75 9.fake 1" > "$scratch/state/last_exit"; echo "556.fake" > "$scratch/qselect"
[[ $(keeper) == 0 ]] || fail "keeper ignored a queued pool"
rm "$scratch/qselect"; mkdir "$scratch/state/keeper.lock"
[[ $(keeper) == 0 ]] || fail "keeper ignored a fresh lock"
rmdir "$scratch/state/keeper.lock"
[[ $(keeper) == 1 ]] || fail "keeper did not resubmit resumable work"
grep -q 'ngpus=2' "$scratch/qsub-calls" && [[ ! -e "$scratch/state/keeper.lock" ]] || fail "keeper settings/lock"
rm -f "$scratch/state/last_exit"; [[ $(keeper) == 1 ]] || fail "keeper must resume a pool that died without a verdict"
mkdir "$scratch/state/keeper.lock"; touch -d '-2 hours' "$scratch/state/keeper.lock"
[[ $(keeper) == 1 ]] || fail "keeper blocked by a stale lock"
touch "$scratch/qselect-fail"; [[ $(keeper) == 0 ]] || fail "keeper submitted blind after qselect failed"
[[ $(RCG_PYTHON=/bin/true submit) == 3 && $(calls) == 0 ]] || fail "launcher submitted blind after qselect failed"
rm "$scratch/qselect-fail"

# 9b. Setup errors are verdict 2: the keeper retries, but gives up after 3 in a row; a
#     user submission re-arms it.
reset; rm -f "$scratch/qsub-calls"; mkdir -p "$scratch/state"
printf 'RCG_PYTHON=/bin/true\nPOOL_GPUS=2\n' > "$scratch/state/submit.env"
rc=0; (cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PYTHON=/nonexistent PBS_JOBID=602.fake \
    bash spooled-pool.pbs) > "$scratch/pool.out" 2>&1 || rc=$?
[[ "$rc" == 2 && "$(cut -d' ' -f1 "$scratch/state/last_exit")" == 2 && "$(cat "$scratch/state/errors")" == 1 ]] ||
    fail "setup error not recorded (rc $rc)"
[[ $(keeper) == 1 && "$(cat "$scratch/state/errors")" == 1 ]] || fail "keeper must retry a first setup error and keep its count"
echo "2 603.fake 1 rc=2" > "$scratch/state/last_exit"; echo 3 > "$scratch/state/errors"
[[ $(keeper) == 0 ]] || fail "keeper resubmitted after 3 setup errors"
rm -f "$scratch/qsub-calls"; [[ $(RCG_PYTHON=/bin/true submit) == 0 && ! -e "$scratch/state/errors" ]] || fail "user submission must re-arm"
# A job that leaves the queue without running pool.pbs (spool exit 126) writes no
# verdict: the keeper counts it as a setup error and gives up after the third.
rm -f "$scratch/state/errors" "$scratch/state/last_exit"; echo 610.fake > "$scratch/state/unstarted"
echo 610.fake > "$scratch/qselect"
[[ $(keeper) == 0 && "$(cat "$scratch/state/unstarted")" == 610.fake ]] || fail "keeper judged a queued job unstarted"
rm "$scratch/qselect"
[[ $(keeper) == 1 && "$(cat "$scratch/state/errors")" == 1 ]] || fail "unstarted job not counted (1)"
grep -q 'ended before its script ran (1 setup' "$scratch/keeper.out" || fail "unstarted verdict"
[[ $(keeper) == 1 && "$(cat "$scratch/state/errors")" == 2 ]] || fail "unstarted job not counted (2)"
[[ $(keeper) == 0 && "$(cat "$scratch/state/errors")" == 3 && ! -e "$scratch/state/unstarted" ]] ||
    fail "keeper looped on jobs that never start"
[[ $(keeper) == 0 ]] || fail "keeper resubmitted after 3 unstarted jobs"
# A job that started before its mark was written (fast start) is not a setup error.
rm -f "$scratch/state/errors"; echo "75 620.aimc-hn2 1" > "$scratch/state/last_exit"
echo 620.aimc-hn2 > "$scratch/state/unstarted"; echo 620.aimc-hn2 > "$scratch/state/started"
[[ $(keeper) == 1 && ! -e "$scratch/state/errors" ]] || fail "keeper counted a started job as unstarted"
grep -q -- '--continue' <(grep -n 'submit_pool.sh' scripts/pbs/pool_keeper.sh | tail -1) || fail "keeper must submit with --continue"

# 9c. Stop file: a running pool drains (checkpoint, no failures), queues no continuation;
#     keeper and launcher refuse until it is removed.
reset; rm -f "$scratch/qsub-calls"; mkdir -p "$scratch/state"
printf 'RCG_PYTHON=/bin/true\nPOOL_GPUS=2\n' > "$scratch/state/submit.env"
echo slow > "$scratch/mode/default"
(cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PYTHON=/bin/true RCG_POOL_SECONDS=600 \
    PBS_JOBID=604.fake bash spooled-pool.pbs) > "$scratch/pool.out" 2>&1 &
pool_pid=$!
until grep -q '^start ' "$EVENTS" 2>/dev/null; do sleep 0.2; done
touch "$scratch/state/stop"
rc=0; wait "$pool_pid" || rc=$?
[[ "$rc" == 75 && $(calls) == 0 && -z "$(ls -A "$scratch/state/failed")" ]] || fail "stop file: rc $rc calls $(calls)"
grep -q 'stop file present: draining' "$scratch/pool.out" && grep -q 'incomplete' "$scratch/pool.out" || fail "stop did not drain"
[[ $(keeper) == 0 ]] || fail "keeper resubmitted a stopped campaign"
[[ $(RCG_PYTHON=/bin/true submit) == 2 && $(calls) == 0 ]] || fail "launcher ignored the stop file"
rm "$scratch/state/stop" "$scratch/mode/default"
[[ $(RCG_PYTHON=/bin/true submit) == 0 && $(calls) == 1 ]] || fail "resume after removing the stop file"
unset RCG_SKIP_INPUT_CHECK

# 10. Status summary is read-only and reports progress and failures.
reset; mkdir -p "$scratch/state/done" "$scratch/state/failed"; touch "$scratch/state/done/1" "$scratch/state/done/2"
echo "9.fake 3 /x/log" > "$scratch/state/failed/3"
bash scripts/pbs/pool_status.sh > "$scratch/status.out"
grep -qE '^00_07_infrastructure +1 +1 ' "$scratch/status.out" && grep -q 'FAILED 3 .* exit 3 log /x/log' "$scratch/status.out" || fail "status output"

printf 'PASS: pool order/caps/packing, stop->resume, failure isolation, OOM retry, leases, tree kill, continuation + self-exclusion, launcher guards, keeper (verdicts, errors, locks, qselect failure), stop file, status\n'
