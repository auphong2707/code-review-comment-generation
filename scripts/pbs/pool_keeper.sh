#!/usr/bin/env bash
# Login-node fallback that keeps the pool going if a finishing pool could not queue its
# own continuation (quota, PBS hiccup, node crash). Install once with crontab -e:
#   */30 * * * * bash <repo>/scripts/pbs/pool_keeper.sh >> <repo>/assets/logs/a100/pool/keeper.log 2>&1
# It submits only when the last pool left resumable work (exit 75) or died without a
# verdict, and no rcg_pool job is queued/running. Finished (0) or failure-blocked (1)
# campaigns are left for the user; submit_pool.sh re-arms the keeper after a fix.
set -uo pipefail
ROOT="${RCG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT" || exit 2
export PATH="$PATH:/opt/pbs/bin"
STATE="${RCG_POOL_STATE:-${RCG_RESULTS_ROOT:-$ROOT/assets/results/revision-a100-native}/.pool}"
stamp() { printf '%s keeper: %s\n' "$(date '+%F %T')" "$*"; }
[[ -f "$STATE/submit.env" ]] || { stamp "no pool submitted yet; nothing to keep"; exit 0; }
lock="$STATE/keeper.lock"
if ! mkdir "$lock" 2>/dev/null; then
    # A keeper killed mid-run leaves its lock; ignore locks older than an hour.
    [[ -n "$(find "$lock" -maxdepth 0 -mmin +60 2>/dev/null)" ]] || { stamp "another keeper is running"; exit 0; }
    rm -rf "$lock"; mkdir "$lock" || exit 0
fi
trap 'rm -rf "$lock"' EXIT
[[ ! -e "$STATE/stop" ]] || { stamp "stop file present ($STATE/stop); not resubmitting"; exit 0; }
verdict=$(cut -d' ' -f1 "$STATE/last_exit" 2>/dev/null || true)
errors=$(cat "$STATE/errors" 2>/dev/null || echo 0)
case "$verdict" in
    0) stamp "campaign complete; nothing to do"; exit 0 ;;
    1) stamp "only failed/blocked tasks remain; waiting for the user (pool_status.sh)"; exit 0 ;;
    2) if (( errors >= 3 )); then
           stamp "$errors consecutive pool setup errors ($(cat "$STATE/last_exit")); see rcg_pool.o* - waiting for the user"
           exit 0
       fi ;;
esac
if ! jobs=$(qselect -u "${USER:-$(id -un)}" -N rcg_pool 2>&1); then
    stamp "qselect failed ($jobs); not submitting blind"; exit 0
fi
# PBS puts a job it failed to launch 20 times on a system hold ("too many failed attempts
# to run", seen 2026-10-20) that only an admin can release: delete it so its unstarted
# mark counts as a setup error below. User holds (qhold) are left alone.
for held in $(qselect -u "${USER:-$(id -un)}" -N rcg_pool -s H 2>/dev/null); do
    qstat -f "$held" 2>/dev/null | grep -Eq 'Hold_Types = [a-z]*s' || continue
    qdel "$held" 2>/dev/null || { stamp "could not delete system-held $held"; continue; }
    stamp "pool job $held was held by PBS after failed launches; deleted it"
    jobs=$(awk -F. -v id="${held%%.*}" 'NF && $1 != id' <<< "$jobs")
    # submit_pool.sh counts every listed job: wait (up to 2 min) until this one is gone.
    for _ in $(seq 24); do
        qselect -u "${USER:-$(id -un)}" -N rcg_pool 2>/dev/null | grep -q "^${held%%.*}\." || break
        sleep "${RCG_SUBMIT_RETRY_SLEEP:-5}"
    done
done
for mark in "$STATE"/unstarted.d/*; do
    [[ -e "$mark" ]] || continue
    lost=$(basename "$mark")
    # Still queued or running: not lost.
    awk -F. -v id="${lost%%.*}" '$1 == id {found = 1} END {exit !found}' <<< "$jobs" && continue
    rm -f "$mark"
    # A job that started before its mark was written left started.d/<id>.
    [[ ! -e "$STATE/started.d/$lost" ]] || continue
    # Submitted, gone from the queue, and pool.pbs never ran (e.g. a broken node spool,
    # exit 126): no verdict was written, so count it here or the keeper would loop.
    errors=$((errors + 1)); verdict=2
    echo "$errors" > "$STATE/errors"
    printf '2 %s %s rc=unstarted\n' "$lost" "$(date +%s)" > "$STATE/last_exit"
    stamp "pool job $lost ended before its script ran ($errors setup errors in a row; see rcg_pool.o*)"
done
(( errors < 3 )) || { stamp "$errors consecutive pool setup errors; waiting for the user"; exit 0; }
limit=$(sed -n 's/^POOL_JOBS=//p' "$STATE/submit.env"); limit="${limit:-1}"
running=$(grep -c . <<< "$jobs" || true)
(( running < limit )) || { stamp "pool job(s) present: $(tr '\n' ' ' <<< "$jobs")"; exit 0; }
stamp "$running of $limit pool job(s) and work remains (last verdict: ${verdict:-none}); submitting"
RCG_ROOT="$ROOT" bash scripts/pbs/submit_pool.sh --continue    # keeps .pool/errors counting
