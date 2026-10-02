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
    [[ "$phase" != chunk && "$phase" != bundle ]] || continue
    # PBS spools the entry outside the checkout; only PBS_O_WORKDIR locates helpers.
    cp "$script" "$scratch/spooled.pbs"
    (cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" DRY=1 bash spooled.pbs) > "$scratch/pbs.txt"
    DRY=1 bash "scripts/slurm/$phase.sbatch" > "$scratch/slurm.txt"
    diff -u "$scratch/slurm.txt" "$scratch/pbs.txt"
done
bash scripts/pbs/submit_revision.sh --preview > "$scratch/preview"
[[ $(wc -l < "$scratch/preview") -eq 8 ]]
[[ $(rg -c 'depend=afterok:' "$scratch/preview") -eq 7 ]]
[[ $(rg -c 'walltime=100:00:00' "$scratch/preview") -eq 8 ]]

# Every producer command must appear exactly once across the split inventory,
# including all LR candidates before their dependent official runs.
bash scripts/pbs/chunks.sh > "$scratch/chunks"
: > "$scratch/expected"
for phase in 12_campaign_a 13_campaign_b 14_campaign_c 15_campaign_d; do
    DRY=1 bash "scripts/slurm/$phase.sbatch" | rg '^STEP ' >> "$scratch/expected"
done
: > "$scratch/actual"
for id in {1..8}; do
    RCG_PBS_BUNDLE="$id" DRY=1 bash scripts/pbs/bundle.pbs >> "$scratch/actual"
done
rg '^STEP [abcd]_' "$scratch/actual" > "$scratch/producers"
diff -u "$scratch/expected" "$scratch/producers"
[[ $(wc -l < "$scratch/producers") -eq 144 ]]
awk '/^PBS chunk / {print $3}' "$scratch/actual" | tr -d ':' > "$scratch/actual-ids"
cut -f1 "$scratch/chunks" > "$scratch/expected-ids"
diff -u "$scratch/expected-ids" "$scratch/actual-ids"
rc=0
RCG_PBS_STEP=nonexistent DRY=1 bash scripts/slurm/12_campaign_a.sbatch > "$scratch/empty" 2>&1 || rc=$?
[[ "$rc" -eq 2 ]]
RCG_PBS_CHUNK=151 DRY=1 bash scripts/pbs/chunk.pbs > "$scratch/judge"
rg -q -- '--job-shard-index 0 --job-shard-count 48' "$scratch/judge"
RCG_PBS_CHUNK=199 DRY=1 bash scripts/pbs/chunk.pbs > "$scratch/summary"
rg -q -- '--summary-only --require-complete' "$scratch/summary"
cp scripts/pbs/chunk.pbs "$scratch/spooled.pbs"
(cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PBS_CHUNK=2 DRY=1 bash spooled.pbs) > "$scratch/spooled-chunk"
[[ $(rg -c '^STEP ' "$scratch/spooled-chunk") -eq 1 ]]
rg -q '^STEP a_p0_codet5_ia3_seed42:' "$scratch/spooled-chunk"
cp scripts/pbs/bundle.pbs "$scratch/spooled.pbs"
(cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PBS_BUNDLE=5 DRY=1 bash spooled.pbs) > "$scratch/spooled-bundle"
[[ $(rg -c '^STEP d_' "$scratch/spooled-bundle") -eq 8 ]]
rg -q '^STEP gate_D:' "$scratch/spooled-bundle"

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
[[ $(wc -l < "$scratch/calls") -eq 8 ]]
rg -q 'depend=afterok:1.aimc-hn1 scripts/pbs/bundle.pbs' "$scratch/calls"
rg -q 'depend=afterok:7.aimc-hn1 scripts/pbs/bundle.pbs' "$scratch/calls"
rm "$scratch/count" "$scratch/calls"
bash scripts/pbs/submit_revision.sh --submit --start 7 --count 2 --afterok 6.aimc-hn1 > "$scratch/next"
[[ $(wc -l < "$scratch/calls") -eq 2 ]]
rg -q 'RCG_PBS_BUNDLE=7 -W depend=afterok:6.aimc-hn1' "$scratch/calls"
rm "$scratch/count" "$scratch/calls"
rc=0
FAIL_AT=3 bash scripts/pbs/submit_revision.sh --submit > "$scratch/failure" || rc=$?
[[ "$rc" -eq 9 && $(wc -l < "$scratch/calls") -eq 3 ]]
rm "$scratch/count" "$scratch/calls"
rc=0
BAD_ID=1 bash scripts/pbs/submit_revision.sh --submit > "$scratch/bad-id" 2>&1 || rc=$?
[[ "$rc" -eq 2 && $(wc -l < "$scratch/calls") -eq 1 ]]
for setting in RCG_PROBES=0 RUN_P1=1 RCG_PARTITION=wrong RCG_PBS_WALLTIME=00:05:00 RCG_PBS_WALLTIME=100:00:01 RCG_PBS_STEP=bad; do
    rc=0
    env "$setting" bash scripts/pbs/submit_revision.sh --preview > "$scratch/rejected" 2>&1 || rc=$?
    [[ "$rc" -eq 2 ]]
done
for args in '--submit --start 9' '--preview --count 9' '--preview --start 999'; do
    rc=0
    # Intentional splitting of fixed test arguments.
    bash scripts/pbs/submit_revision.sh $args > "$scratch/rejected" 2>&1 || rc=$?
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
printf 'query\n' >> "$PBS_TEST_DIR/qstat-calls"
MOCK
chmod +x "$scratch/bin/qstat"
rc=0
PBS_JOBID=42.aimc-hn1 RCG_ROOT="$scratch/repo" RCG_PYTHON=/bin/true \
    RCG_LOG_ROOT="$scratch/logs" WANDB_DIR="$scratch/wandb" \
    bash scripts/pbs/run_phase.sh 12_campaign_a > "$scratch/stop-log" 2>&1 || rc=$?
[[ "$rc" -eq 75 && -f "$scratch/committed" && ! -e "$scratch/incorrectly-completed" ]]
# The bundle must have only one allocation timer, and its stop signal must reach
# the nested worker without killing the child's log pipe or starting another task.
mkdir -p "$scratch/repo/scripts/pbs"
cp scripts/pbs/run_bundle.sh scripts/pbs/run_chunk.sh "$scratch/repo/scripts/pbs/"
cat > "$scratch/repo/scripts/pbs/chunks.sh" <<'INVENTORY'
printf '1\tstep\t12_campaign_a\tfake_worker\n2\tstep\t12_campaign_a\tnever_started\n'
INVENTORY
rm "$scratch/committed" "$scratch/qstat-calls"
rc=0
PBS_JOBID=42.aimc-hn1 RCG_PBS_BUNDLE=2 RCG_ROOT="$scratch/repo" RCG_PYTHON=/bin/true \
    RCG_LOG_ROOT="$scratch/logs" WANDB_DIR="$scratch/wandb" \
    bash scripts/pbs/run_phase.sh bundle > "$scratch/stop-log" 2>&1 || rc=$?
[[ "$rc" -eq 75 && -f "$scratch/committed" && ! -e "$scratch/incorrectly-completed" ]]
[[ $(wc -l < "$scratch/qstat-calls") -eq 1 ]]
rc=0
PBS_JOBID=42.aimc-hn1 FAKE_WALLTIME=invalid \
    bash scripts/pbs/run_phase.sh 12_campaign_a > "$scratch/invalid-time" 2>&1 || rc=$?
[[ "$rc" -eq 2 ]]
printf 'PASS: eight jobs, 267 internal tasks, 144 producer commands, spooled dispatch, dependencies/failures, scope guards and nested timed recovery\n'
