#!/usr/bin/env bash
# Bash-only regression checks: no Python, models, SSH or real scheduler calls.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
scratch=$(mktemp -d)
trap 'rc=$?; if (( rc != 0 )) && [[ -f "$scratch/stop-log" ]]; then cat "$scratch/stop-log"; fi; rm -rf "$scratch"' EXIT
export RCG_ROOT="$ROOT" RCG_PYTHON=/bin/false RCG_JSON=0
for script in scripts/pbs/*.sh scripts/pbs/*.pbs scripts/slurm/*.sh scripts/slurm/*.sbatch; do
    bash -n "$script"
done
for script in scripts/pbs/*.pbs; do
    phase=$(basename "$script" .pbs)
    # PBS spools the entry outside the checkout; only PBS_O_WORKDIR locates helpers.
    cp "$script" "$scratch/spooled.pbs"
    (cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" DRY=1 bash spooled.pbs) > "$scratch/pbs.txt"
    DRY=1 bash "scripts/slurm/$phase.sbatch" > "$scratch/slurm.txt"
    diff -u "$scratch/slurm.txt" "$scratch/pbs.txt"
done
bash scripts/pbs/submit_revision.sh --preview > "$scratch/preview"
[[ $(wc -l < "$scratch/preview") -eq 6 ]]
[[ $(rg -c 'depend=afterok:' "$scratch/preview") -eq 5 ]]

mkdir "$scratch/bin"
export PBS_TEST_DIR="$scratch"
cat > "$scratch/bin/qsub" <<'MOCK'
#!/usr/bin/env bash
set -eu
count=0
[[ ! -f "$PBS_TEST_DIR/count" ]] || read -r count < "$PBS_TEST_DIR/count"
count=$((count + 1))
printf '%s\n' "$count" > "$PBS_TEST_DIR/count"
printf '%s\n' "$*" >> "$PBS_TEST_DIR/calls"
[[ "$count" != "${FAIL_AT:-0}" ]] || exit 9
if [[ "${BAD_ID:-0}" == 1 ]]; then echo unexpected-output; else echo "$count.aimc-hn1"; fi
MOCK
chmod +x "$scratch/bin/qsub"
export PATH="$scratch/bin:$PATH"
bash scripts/pbs/submit_revision.sh --submit > "$scratch/ids"
[[ $(wc -l < "$scratch/calls") -eq 6 ]]
rg -q 'depend=afterok:1.aimc-hn1 scripts/pbs/12_campaign_a.pbs' "$scratch/calls"
rg -q 'depend=afterok:5.aimc-hn1 scripts/pbs/16_campaign_j.pbs' "$scratch/calls"
rm "$scratch/count" "$scratch/calls"
rc=0
FAIL_AT=3 bash scripts/pbs/submit_revision.sh --submit > "$scratch/failure" || rc=$?
[[ "$rc" -eq 9 && $(wc -l < "$scratch/calls") -eq 3 ]]
rm "$scratch/count" "$scratch/calls"
rc=0
BAD_ID=1 bash scripts/pbs/submit_revision.sh --submit > "$scratch/bad-id" 2>&1 || rc=$?
[[ "$rc" -eq 2 && $(wc -l < "$scratch/calls") -eq 1 ]]
for setting in RCG_PROBES=0 RUN_P1=1 RCG_PARTITION=wrong RCG_PBS_WALLTIME=00:05:00; do
    rc=0
    env "$setting" bash scripts/pbs/submit_revision.sh --preview > "$scratch/rejected" 2>&1 || rc=$?
    [[ "$rc" -eq 2 ]]
done

# A real one-second warning against a fake Bash worker checks stop propagation,
# committed worker exit, and that the next campaign step is never started.
mkdir -p "$scratch/repo/scripts/slurm" "$scratch/repo/src"
cp scripts/slurm/common.sh "$scratch/repo/scripts/slurm/common.sh"
touch "$scratch/repo/src/train.py"
cat > "$scratch/worker.sh" <<'WORKER'
#!/usr/bin/env bash
trap 'touch "$PBS_TEST_DIR/committed"; exit 75' USR1
while true; do sleep 0.05; done
WORKER
cat > "$scratch/repo/scripts/slurm/12_campaign_a.sbatch" <<'BODY'
#!/usr/bin/env bash
source "$RCG_ROOT/scripts/slurm/common.sh"
run_step fake_worker bash "$PBS_TEST_DIR/worker.sh"
touch "$PBS_TEST_DIR/incorrectly-completed"
BODY
cat > "$scratch/bin/qstat" <<'MOCK'
#!/usr/bin/env bash
printf '    Resource_List.walltime = %s\n' "${FAKE_WALLTIME:-00:10:01}"
MOCK
chmod +x "$scratch/bin/qstat"
rc=0
PBS_JOBID=42.aimc-hn1 RCG_ROOT="$scratch/repo" RCG_PYTHON=/bin/true \
    RCG_LOG_ROOT="$scratch/logs" WANDB_DIR="$scratch/wandb" \
    bash scripts/pbs/run_phase.sh 12_campaign_a > "$scratch/stop-log" 2>&1 || rc=$?
[[ "$rc" -eq 75 && -f "$scratch/committed" && ! -e "$scratch/incorrectly-completed" ]]
rc=0
PBS_JOBID=42.aimc-hn1 FAKE_WALLTIME=invalid \
    bash scripts/pbs/run_phase.sh 12_campaign_a > "$scratch/invalid-time" 2>&1 || rc=$?
[[ "$rc" -eq 2 ]]
printf 'PASS: syntax, six spooled preview parity checks, PBS dependencies/failures, scope guards and timed recovery\n'
