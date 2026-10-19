#!/usr/bin/env bash
# Bash-only regression checks: no Python, models, SSH or real scheduler calls.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
scratch=$(mktemp -d)
trap 'rc=$?; if (( rc != 0 )) && [[ -f "$scratch/stop-log" ]]; then cat "$scratch/stop-log"; fi; rm -rf "$scratch"' EXIT
export RCG_ROOT="$ROOT" RCG_PYTHON=/bin/false RCG_JSON=0 RCG_SKIP_INPUT_CHECK=1
for script in scripts/pbs/*.sh scripts/pbs/*.pbs scripts/slurm/*.sh scripts/slurm/*.sbatch; do
    bash -n "$script"
done
for script in scripts/pbs/*.pbs; do
    phase=$(basename "$script" .pbs)
    [[ "$phase" != chunk && "$phase" != bundle && "$phase" != pool ]] || continue
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
[[ $(wc -l < "$scratch/producers") -eq 103 ]]
# A100 profile everywhere; no CodeReviewer training; CodeReviewer ICL on the fixed subset.
[[ $(rg -c -- "--memory-profile a100 " "$scratch/producers") -eq 103 ]]
if rg -q -- "-m src.train .*--dataset acra_by_ls_pt" "$scratch/producers"; then echo "CodeReviewer training left" >&2; exit 1; fi
[[ $(rg -c -- "--dataset acra_by_ls_pt .*--sample-index-file paper/generated/cr_judge_subset.json" "$scratch/producers") -eq 33 ]]
[[ $(rg -c -- "^STEP a_p0_qwen(25|15)_" "$scratch/producers") -eq 4 && $(rg -c -- "^STEP a_p0_.*_seed42:" "$scratch/producers" || true) -eq 0 ]]
awk '/^PBS chunk / {print $3}' "$scratch/actual" | tr -d ':' > "$scratch/actual-ids"
cut -f1 "$scratch/chunks" > "$scratch/expected-ids"
diff -u "$scratch/expected-ids" "$scratch/actual-ids"
rc=0
RCG_PBS_STEP=nonexistent DRY=1 bash scripts/slurm/12_campaign_a.sbatch > "$scratch/empty" 2>&1 || rc=$?
[[ "$rc" -eq 2 ]]
chunk_id() { awk -F '\t' -v k="$1" -v p="$2" -v i="$3" '$2 == k && $3 == p && $4 == i {print $1}' "$scratch/chunks"; }
RCG_PBS_CHUNK=$(chunk_id judge qwen-revision 0) DRY=1 bash scripts/pbs/chunk.pbs > "$scratch/judge"
rg -q -- '--job-shard-index 0 --job-shard-count 48' "$scratch/judge"
rg -q -- '--judge-cache-path [^ ]*/qwen-revision/cache/shard_000\.jsonl' "$scratch/judge"
rg -q -- '--judge-k 1 ' "$scratch/judge"
RCG_PBS_CHUNK=$(chunk_id judge llama-revision 0) DRY=1 bash scripts/pbs/chunk.pbs > "$scratch/llama-judge"
rg -q -- '--judge-k 1 ' "$scratch/llama-judge"
RCG_PBS_CHUNK=$(chunk_id judge llama-legacy 0) DRY=1 bash scripts/pbs/chunk.pbs > "$scratch/legacy-judge"
rg -q -- '--judge-k 1 ' "$scratch/legacy-judge"
for pass in judge llama-judge legacy-judge; do rg -q -- '--judge-batch-size 16 ' "$scratch/$pass"; done
RCG_PBS_CHUNK=$(chunk_id summary qwen-revision all) DRY=1 bash scripts/pbs/chunk.pbs > "$scratch/summary"
rg -q -- '--summary-only --require-complete' "$scratch/summary"
if rg -q -- '--judge-cache-path' "$scratch/summary"; then echo "Summary must not take a shard cache" >&2; exit 1; fi
cp scripts/pbs/chunk.pbs "$scratch/spooled.pbs"
(cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PBS_CHUNK=2 DRY=1 bash spooled.pbs) > "$scratch/spooled-chunk"
[[ $(rg -c '^STEP ' "$scratch/spooled-chunk") -eq 1 ]]
rg -q '^STEP a_p0_codet5_ia3_seed1:' "$scratch/spooled-chunk"
cp scripts/pbs/bundle.pbs "$scratch/spooled.pbs"
(cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PBS_BUNDLE=5 DRY=1 bash spooled.pbs) > "$scratch/spooled-bundle"
[[ $(rg -c '^STEP d_trans_review_' "$scratch/spooled-bundle") -eq 4 && $(rg -c '^STEP d_' "$scratch/spooled-bundle") -eq 4 ]]
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
mkdir -p "$scratch/noweights/scripts/pbs" "$scratch/noweights/scripts/slurm"
cp scripts/pbs/*.sh scripts/pbs/*.pbs "$scratch/noweights/scripts/pbs/"; cp scripts/slurm/* "$scratch/noweights/scripts/slurm/"
# Missing judge weights must stop the launcher before any qsub call.
if (cd "$scratch/noweights"; RCG_ROOT="$scratch/noweights" RCG_SKIP_INPUT_CHECK=0 bash scripts/pbs/submit_revision.sh --submit) 2> "$scratch/noweights.err"; then
    echo "Launcher submitted without judge weights" >&2; exit 1
fi
rg -q "Missing judge input: assets/models/llm-judge/Llama-3.1-8B-Instruct/config.json" "$scratch/noweights.err"
[[ ! -f "$scratch/calls" ]]
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
# Parallel-pool prerequisites: complete, acyclic, and every edge points backwards, so
# inventory order is a valid run order. Pin the edges that protect shared outputs.
bash scripts/pbs/pool_deps.sh > "$scratch/deps"
diff -u <(cut -f1 "$scratch/chunks") <(cut -f1 "$scratch/deps")
awk -F '\t' '{n = split($2, d, ","); for (i = 1; i <= n; i++) if (d[i] + 0 >= $1 + 0) exit 1}' "$scratch/deps"
awk -F '\t' '{n = split($2, d, ","); if (n == 0) print $1, $1; for (i = 1; i <= n; i++) print d[i], $1}' "$scratch/deps" | tsort > /dev/null
dep_of() { awk -F '\t' -v id="$(awk -F '\t' -v l="$1" '$4 == l {print $1}' "$scratch/chunks")" '$1 == id {print $2}' "$scratch/deps"; }
id_of() { awk -F '\t' -v l="$1" '$4 == l {print $1}' "$scratch/chunks"; }
# B is five TransReview DoRA runs (controls are the archived paper LoRA runs): no
# shared run directories with A, so they wait only for the infrastructure job.
for model in codet5 plbart qwen05b qwen25 qwen15; do [[ "$(dep_of "b_tr_${model}_dora")" == 1 ]]; done
[[ -z "$(id_of b_tr_codet5_lora_control)" ]]
for lr in 0 1 2; do [[ ",$(dep_of d_trans_review_qlora_official)," == *",$(id_of "d_trans_review_qlora_sweep_lr$lr"),"* ]]; done
[[ -z "$(id_of d_acra_by_ls_pt_qlora_official)" ]]
[[ $(awk -F '\t' '$2 == "judge" && $3 == "llama-legacy"' "$scratch/chunks" | wc -l) -eq 18 ]]
awk -F '\t' 'NR == FNR {if ($2 == "judge" && $3 == "llama-legacy") legacy[$1] = 1; next} ($1 in legacy) && $2 != "1" {exit 1}' "$scratch/chunks" "$scratch/deps"
# The shared BM25 index pickle is built by the first BM25 step of each dataset.
for ds in trans_review acra_by_ls_pt; do
    bm25=($(awk -F '\t' -v ds="$ds" '$2 == "step" && $4 ~ ("_" ds "_[0-9]+shot_bm25$") {print $4}' "$scratch/chunks"))
    [[ ${#bm25[@]} -eq 9 ]]
    for label in "${bm25[@]:1}"; do [[ ",$(dep_of "$label")," == *",$(id_of "${bm25[0]}"),"* ]]; done
done
# The pool previews under DRY=1 and refuses to run outside PBS; neither writes state.
cp scripts/pbs/pool.pbs "$scratch/spooled-pool.pbs"
(cd "$scratch"; unset RCG_ROOT; PBS_O_WORKDIR="$ROOT" RCG_PYTHON=/bin/false DRY=1 bash spooled-pool.pbs) > "$scratch/pool-preview"
rg -q '^Pool preview: 226 tasks' "$scratch/pool-preview"
rc=0; (unset PBS_JOBID; RCG_POOL_STATE="$scratch/pool-state" bash scripts/pbs/pool.sh) > /dev/null 2>&1 || rc=$?
[[ "$rc" -eq 2 && ! -e "$scratch/pool-state" ]]
# Exactly these step-to-step edges: 3 sweep->official + 16 BM25.
[[ $(awk -F '\t' 'NR == FNR {if ($2 == "step") step[$1] = 1; next}
    ($1 in step) {n = split($2, d, ","); for (i = 1; i <= n; i++) if (d[i] != 1) c++} END {print c + 0}' \
    "$scratch/chunks" "$scratch/deps") -eq 19 ]]
# Seed 42 of all 18 A configurations and the 5 B controls are archived paper runs.
[[ $(rg -c '"paper_run": "assets/models/review-code-generation/' paper/generated/seed42_paper_reuse.json) -eq 18 ]]
[[ $(rg -c '"control_run": "assets/models/review-code-generation/' paper/generated/seed42_paper_reuse.json) -eq 5 ]]

printf 'PASS: eight jobs, 226 internal tasks, acyclic pool prerequisites, 103 A100 producer commands, CR ICL subset, judge k, spooled dispatch, dependencies/failures, scope guards and nested timed recovery\n'
