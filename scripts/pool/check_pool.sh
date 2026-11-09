#!/usr/bin/env bash
# Bash-only checks of the parallel pool over the real 226-task inventory with fake tasks,
# fake GPUs and fake squeue/sacct: dependency order, memory/count caps, stop -> resume, failure
# isolation, OOM retry, stale/live leases and killing a stuck task's process tree.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
scratch=$(mktemp -d)
trap 'pkill -f "$scratch/" 2>/dev/null || true; rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/pre/admissions"
export RCG_ROOT="$ROOT" RCG_PREFLIGHT_ROOT="$scratch/pre" RCG_POOL_ALLOW_LOCAL=1
export RCG_POOL_NVIDIA_SMI=/bin/false RCG_POOL_GPUS=0,1 RCG_POOL_POLL=0.05 RCG_POOL_LAUNCH_GAP=0 RCG_POOL_MARGIN=1 RCG_POOL_GRACE=2
export RCG_POOL_TASK_CMD="bash $scratch/task.sh" EVENTS="$scratch/events" FAKE="$scratch"
unset PBS_JOBID CUDA_VISIBLE_DEVICES SLURM_JOB_ID SLURM_JOB_NAME SLURM_GPUS_ON_NODE SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE SLURM_SUBMIT_DIR

# Pilot peaks: training 30 GiB (cost 32, two per GPU), ICL 10, judges 5.
bash scripts/pool/pool_deps.sh > "$scratch/deps"
for name in $(cut -f3 "$scratch/deps" | sort -u); do
    case "$name" in
        train_*) peak=30 ;; icl_*) peak=10 ;; judge_*) peak=5 ;; *) continue ;;
    esac
    printf '{"admitted": true, "peak_reserved_gib": %s}\n' "$peak" > "$scratch/pre/admissions/$name.json"
done
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
    # Another job grabs 8 GiB of this GPU, then the task runs out of memory.
    oom-foreign-once) if [[ ! -e "$FAKE/oom-$id" ]]; then touch "$FAKE/oom-$id"; g=a; [[ "$CUDA_VISIBLE_DEVICES" == 0 ]] || g=b
                          echo "999993, GPU-$g, 8192" >> "$FAKE/foreign-apps"; echo "torch.OutOfMemoryError: CUDA out of memory"; end 1; fi; end 0 ;;
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
run_pool() { local rc=0; RCG_POOL_STATE="$scratch/state" RCG_POOL_LOG_ROOT="$scratch/logs" bash scripts/pool/pool.sh > "$scratch/pool.out" 2>&1 || rc=$?; echo "$rc"; }
reset() { rm -rf "$scratch/state" "$scratch/logs" "$scratch/mode" "$EVENTS" "$scratch"/oom-* "$scratch"/killed-* "$scratch"/deferred-* "$scratch"/deadline-*; mkdir -p "$scratch/mode"; }
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
id_of() { awk -F '\t' -v l="$1" '$4 == l {print $1}' <(bash scripts/pool/chunks.sh); }
sweep=$(id_of d_trans_review_qlora_sweep_lr0); official=$(id_of d_trans_review_qlora_official)
echo fail > "$scratch/mode/$sweep"
[[ $(run_pool) == 1 ]] || fail "only-blocked pool must exit 1"
[[ -e "$scratch/state/failed/$sweep" && ! -e "$scratch/state/done/$official" ]] || fail "failure not isolated"
grep -q "^start $official " "$EVENTS" && fail "dependent of a failed task started"
[[ -e "$scratch/state/done/$(id_of c_qwen15_acra_by_ls_pt_10shot_bm25)" ]] || fail "independent work blocked"
for id in $(awk -F '\t' '$2 == "judge" && $3 == "llama-legacy" {print $1}' <(bash scripts/pool/chunks.sh)); do
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

# 4e. Another job's processes on our GPUs (KI-022): 52 GiB on GPU 1 caps it at 79-3-52 =
#     24 GiB, so no bigger task starts there. An OOM after other jobs took more memory is
#     requeued; an OOM beside memory already there at launch escalates as before.
reset
cat > "$scratch/bin/fake-nvsmi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *query-gpu=index,uuid*) printf '0, GPU-a\n1, GPU-b\n' ;;
    *query-compute-apps*) cat "$FAKE/foreign-apps" ;;
esac
EOF
chmod +x "$scratch/bin/fake-nvsmi"
foreign_pool() { RCG_POOL_NVIDIA_SMI="$scratch/bin/fake-nvsmi" RCG_POOL_FOREIGN_EVERY=0 run_pool; }
printf '999990, GPU-a, 3072\n999991, GPU-b, 53248\n999992, GPU-b, [N/A]\n' > "$scratch/foreign-apps"
echo oom-foreign-once > "$scratch/mode/2"
[[ $(foreign_pool) == 0 ]] || fail "foreign-memory run exit"
grep -q 'gpu 1: 52GiB held by other jobs; capacity 24GiB' "$scratch/pool.out" || fail "foreign capacity not logged"
awk '/ start [0-9]+ .* gpu=1 cost=/ {c = $0; sub(/.*cost=/, "", c); sub(/GiB.*/, "", c); if (c + 0 > 24) bad = 1}
     END {exit bad}' "$scratch/pool.out" || fail "task over the foreign-reduced capacity started on gpu 1"
grep -qE 'out of memory 2 after other jobs took 8GiB more of gpu [01]; requeued \(1/5\)' "$scratch/pool.out" &&
    [[ ! -e "$scratch/state/exclusive/2" && -e "$scratch/state/done/2" ]] || fail "foreign OOM not requeued"
check_events
reset
printf '999990, GPU-a, 3072\n999991, GPU-b, 53248\n' > "$scratch/foreign-apps"
echo oom-once > "$scratch/mode/2"
[[ $(foreign_pool) == 0 && -e "$scratch/state/exclusive/2" ]] || fail "OOM beside memory present at launch must escalate"
# Nothing fits beside other jobs (70 GiB on both GPUs): wait, then leave it (75) to a later pool.
reset
printf '999990, GPU-a, 71680\n999991, GPU-b, 71680\n' > "$scratch/foreign-apps"
[[ $(RCG_POOL_FOREIGN_WAIT=1 foreign_pool) == 75 ]] || fail "unfit pool must leave the work (75)"
grep -q 'still no fit; leaving the work for a later pool job' "$scratch/pool.out" && ! grep -q ' start ' "$scratch/pool.out" ||
    fail "unfit wait"

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

# 5s. Slurm allocation: time left from squeue %L ([D-]HH:MM:SS, MM:SS), GPU count from
#     SLURM_GPUS_ON_NODE, lease liveness from squeue then sacct (dead only on evidence).
cat > "$scratch/bin/squeue" <<'EOF'
#!/usr/bin/env bash
while (( $# )); do case "$1" in -j) id="$2"; shift 2 ;; -o) fmt="$2"; shift 2 ;; *) shift ;; esac; done
grep -qx "$id" "$FAKE/squeue-down" 2>/dev/null && { echo "squeue: error: Unable to contact slurm controller" >&2; exit 1; }
grep -qx "$id" "$FAKE/squeue-invalid" 2>/dev/null && { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
if [[ "$fmt" == %L ]]; then cat "$FAKE/squeue-left"; else awk -v j="$id" '$1 == j {print $2}' "$FAKE/squeue-states" 2>/dev/null; fi
EOF
cat > "$scratch/bin/sacct" <<'EOF'
#!/usr/bin/env bash
while (( $# )); do case "$1" in -j) id="$2"; shift 2 ;; *) shift ;; esac; done
awk -v j="$id" '$1 == j {$1 = ""; sub(/^ /, ""); print}' "$FAKE/sacct-states" 2>/dev/null
EOF
chmod +x "$scratch/bin/squeue" "$scratch/bin/sacct"
slurm_pool() { RCG_SQUEUE="$scratch/bin/squeue" RCG_SACCT="$scratch/bin/sacct" SLURM_JOB_ID=800 run_pool; }
slurm_reset() { reset; rm -f "$scratch"/squeue-* "$scratch/sacct-states"; echo "800 RUNNING" > "$scratch/squeue-states"; }
slurm_reset; echo "00:04" > "$scratch/squeue-left"; echo slow > "$scratch/mode/default"
[[ $(unset RCG_POOL_SECONDS; slurm_pool) == 75 ]] || fail "Slurm pool must stop at the squeue time left (75)"
grep -q 'incomplete' "$scratch/pool.out" && [[ -z "$(ls -A "$scratch/state/failed")" ]] || fail "Slurm stop did not checkpoint"
slurm_reset; echo "2-01:00:00" > "$scratch/squeue-left"
mkdir -p "$scratch/state"; touch "$scratch/state/stop"      # log the deadline, then drain at once
want=$(( $(date +%s) + 2 * 86400 + 3600 - 1 ))
[[ $(unset RCG_POOL_SECONDS; slurm_pool) == 75 ]] || fail "stopped Slurm pool exit"
stop_at=$(grep -oE 'stop_at=[0-9-]+ [0-9:]+' "$scratch/pool.out" | head -1 | cut -d= -f2)
diff=$(( $(date -d "$stop_at" +%s) - want ))
(( diff >= -10 && diff <= 10 )) || fail "Slurm D-HH:MM:SS deadline off by ${diff}s ($stop_at)"
slurm_reset; echo "garbage" > "$scratch/squeue-left"
[[ $(unset RCG_POOL_SECONDS; slurm_pool) == 2 ]] && grep -q 'Cannot read the time left' "$scratch/pool.out" || fail "unparsable squeue time accepted"
slurm_reset; echo "09:08:07" > "$scratch/squeue-left"; mkdir -p "$scratch/state"; touch "$scratch/state/stop"
want=$(( $(date +%s) + 9 * 3600 + 8 * 60 + 7 - 1 ))
[[ $(unset RCG_POOL_SECONDS; slurm_pool) == 75 ]] || fail "stopped Slurm pool exit (HH:MM:SS)"
stop_at=$(grep -oE "stop_at=[0-9-]+ [0-9:]+" "$scratch/pool.out" | head -1 | cut -d= -f2)
diff=$(( $(date -d "$stop_at" +%s) - want )); (( diff >= -10 && diff <= 10 )) || fail "Slurm 09:08:07 deadline off by ${diff}s"
slurm_reset
rc=$(unset RCG_POOL_GPUS; CUDA_VISIBLE_DEVICES=0,1 SLURM_GPUS_ON_NODE=1 RCG_POOL_SECONDS=3 slurm_pool)
[[ "$rc" == 2 ]] && grep -q 'do not match the allocation (ngpus=1)' "$scratch/pool.out" || fail "Slurm GPU mismatch accepted"
slurm_reset
rc=$(unset RCG_POOL_GPUS; CUDA_VISIBLE_DEVICES=0,1 SLURM_GPUS_ON_NODE=2 RCG_POOL_SECONDS=3 slurm_pool)
[[ "$rc" != 2 ]] && grep -q 'gpus=0 1 ' "$scratch/pool.out" || fail "matching Slurm GPUs refused (rc $rc)"
slurm_reset
for id in 1 2 3 4 5 6 7 8; do mkdir -p "$scratch/state/lease/$id"; done
for pair in '1 901' '2 902' '3 903' '4 904' '5 905' '6 906' '7 907' '8 908'; do read -r id owner <<< "$pair"; echo "$owner host 1 0" > "$scratch/state/lease/$id/owner"; done
printf '800 RUNNING\n902 RUNNING\n908 COMPLETED\n' > "$scratch/squeue-states"     # 902 alive; 908 listed but ended
printf '903\n906\n' > "$scratch/squeue-invalid"; echo 904 > "$scratch/squeue-down"   # 906: purged, no accounting; 907: unlisted, no record
printf '901 COMPLETED\n903 CANCELLED by 1234\n905 RUNNING\n' > "$scratch/sacct-states"   # 905: left squeue, sacct still running
rc=$(RCG_POOL_SECONDS=3 slurm_pool)
grep -q "reclaim lease 1 from 901" "$scratch/pool.out" && grep -q "reclaim lease 3 from 903" "$scratch/pool.out" &&
    grep -q "reclaim lease 6 from 906" "$scratch/pool.out" && grep -q "reclaim lease 8 from 908" "$scratch/pool.out" ||
    fail "finished Slurm owners' leases not reclaimed"
[[ -d "$scratch/state/lease/2" && -d "$scratch/state/lease/4" && -d "$scratch/state/lease/5" && -d "$scratch/state/lease/7" ]] || fail "Slurm lease freed without evidence"
grep -q '^start 2 ' "$EVENTS" && fail "task leased by a running job was started"
# An owner-less lease (another pool between mkdir and its owner write) is kept until it
# is 10 minutes old.
slurm_reset
for id in 3 4; do mkdir -p "$scratch/state/lease/$id"; done
touch -d '-20 minutes' "$scratch/state/lease/4"
rc=$(RCG_POOL_SECONDS=3 slurm_pool)
[[ -d "$scratch/state/lease/3" && ! -d "$scratch/state/lease/4" ]] || fail "owner-less lease handling"
# A requeued job keeps its Slurm job ID: its old leases are released at start.
slurm_reset
mkdir -p "$scratch/state/lease/1"; echo '800 host 1 0' > "$scratch/state/lease/1/owner"
rc=$(RCG_POOL_SECONDS=3 slurm_pool)
grep -q 'release lease 1 left by an earlier run of this job' "$scratch/pool.out" && grep -q '^start 1 ' "$EVENTS" ||
    fail "requeued job's own lease not released"

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

# 10. Status summary is read-only and reports progress and failures.
reset; mkdir -p "$scratch/state/done" "$scratch/state/failed"; touch "$scratch/state/done/1" "$scratch/state/done/2"
echo "9.fake 3 /x/log" > "$scratch/state/failed/3"
RCG_POOL_STATE="$scratch/state" bash scripts/pool/pool_status.sh > "$scratch/status.out"
grep -qE '^00_07_infrastructure +1 +1 ' "$scratch/status.out" && grep -q 'FAILED 3 .* exit 3 log /x/log' "$scratch/status.out" || fail "status output"

printf 'PASS: pool order/caps/packing, stop->resume, failure isolation, OOM retry, foreign GPU memory, Slurm deadline/GPUs/leases, tree kill, status\n'
