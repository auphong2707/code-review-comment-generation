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
STATE="${RCG_POOL_STATE:-${RCG_RESULTS_ROOT:-$ROOT/assets/results/revision-a100}/.pool}"
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
[[ -z "$jobs" ]] || { stamp "pool job present: $(tr '\n' ' ' <<< "$jobs")"; exit 0; }
stamp "no pool job and work remains (last verdict: ${verdict:-none}); submitting"
RCG_ROOT="$ROOT" bash scripts/pbs/submit_pool.sh
