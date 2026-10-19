#!/usr/bin/env bash
# Pool progress from the shared state: counts by phase, running tasks (GPU, age, owner)
# and failures with their logs. Read-only; safe on the login node at any time.
set -euo pipefail
ROOT="${RCG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
STATE="${RCG_POOL_STATE:-${RCG_RESULTS_ROOT:-$ROOT/assets/results/revision-a100-native}/.pool}"
[[ -d "$STATE" ]] || { echo "No pool state yet at $STATE"; exit 0; }
now=$(date +%s)
bash scripts/pbs/chunks.sh | awk -F '\t' -v state="$STATE" -v now="$now" '
    function exists(p) { return (system("test -e \"" p "\"") == 0) }
    {
        group = ($2 == "step") ? $3 : $2 " " $3
        sub(/^1[2-5]_campaign_/, "campaign ", group); sub(/^phase /, "", group)
        total[group]++; order[group] = order[group] ? order[group] : NR
        if (exists(state "/done/" $1)) done[group]++
        else if (exists(state "/failed/" $1)) { failed[group]++; bad[$1] = $4 }
        else if (exists(state "/lease/" $1)) { running[group]++; run[$1] = $4 }
    }
    END {
        printf "%-28s %6s %6s %7s %6s\n", "group", "done", "total", "running", "failed"
        n = 0; for (g in total) keys[++n] = g
        for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (order[keys[j]] < order[keys[i]]) { t = keys[i]; keys[i] = keys[j]; keys[j] = t }
        for (i = 1; i <= n; i++) { g = keys[i]; printf "%-28s %6d %6d %7d %6d\n", g, done[g], total[g], running[g], failed[g] }
        for (id in run) {
            owner = ""; f = state "/lease/" id "/owner"; getline owner < f; close(f)
            split(owner, o, " "); cmd = "stat -c %Y \"" state "/lease/" id "\""; cmd | getline since; close(cmd)
            printf "running %s %s on gpu %s for %dm (job %s)\n", id, run[id], o[4], (now - since) / 60, o[1]
        }
        for (id in bad) {
            line = ""; f = state "/failed/" id; getline line < f; close(f)
            split(line, b, " "); printf "FAILED %s %s exit %s log %s\n", id, bad[id], b[2], b[3]
        }
    }'
[[ -z "$(ls -A "$STATE/unstarted.d" 2>/dev/null)" ]] || echo "submitted, script not started yet: $(ls "$STATE/unstarted.d" | tr '\n' ' ')"
[[ ! -f "$STATE/last_exit" ]] || echo "last pool exit: $(cat "$STATE/last_exit") (0 complete, 75 continuing, 1 blocked)"
command -v qselect >/dev/null && echo "pool jobs: $(qselect -u "${USER:-$(id -un)}" -N rcg_pool -s QRHW 2>/dev/null | tr '\n' ' ')"
true
