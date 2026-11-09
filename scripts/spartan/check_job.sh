#!/usr/bin/env bash
# Local Bash-only checks of scripts/spartan/revision.slurm (not a job file): a scratch copy
# of the repository scripts, fake sbatch/squeue/nvidia-smi/python/hf and a fake pool.
# Nothing is submitted, downloaded or uploaded, and nothing is written under assets/.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
R="$scratch/repo" B="$scratch/bin"
mkdir -p "$R/src" "$R/paper/generated" "$R/assets/data/Trans_Review" "$R/assets/data/Code_Review" "$B"
cp -r "$REPO/scripts" "$R/"
touch "$R/src/train.py" "$R/assets/data/Trans_Review/train.jsonl" "$R/assets/data/Code_Review/train.jsonl"
echo '[1]' > "$R/paper/generated/cr_judge_subset.json"
export FAKE="$scratch" RCG_ROOT="$R" RCG_PYTHON="$B/python" RCG_SBATCH="$B/sbatch" RCG_SQUEUE="$B/squeue"
export RCG_NVIDIA_SMI="$B/nvidia-smi" RCG_SKIP_INPUT_CHECK=1 HF_HOME="$scratch/hf-home"
export RCG_SPARTAN_POOL_CMD="bash $B/pool" RCG_SPARTAN_POLL=0.2 RCG_SPARTAN_PUSH_EVERY=0 RCG_SPARTAN_PROGRESS_EVERY=0
unset SLURM_JOB_ID SLURM_JOB_NAME SLURM_GPUS_ON_NODE SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE SLURM_SUBMIT_DIR PBS_JOBID CUDA_VISIBLE_DEVICES RCG_SPARTAN_ACCOUNT HF_TOKEN RCG_HF_SYNC
JOB="$R/scripts/spartan/revision.slurm"
STATE="$R/assets/results/revision-a100-native/.pool"
LOGS="$R/assets/logs/spartan"

cat > "$B/sbatch" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE/sbatch-calls"; env | grep -E '^SLURM_' >> "$FAKE/sbatch-env" || true
echo "Submitted batch job 901"
EOF
cat > "$B/squeue" <<'EOF'
#!/usr/bin/env bash
[[ ! -e "$FAKE/squeue-fail" ]] || { echo "squeue: error: Unable to contact slurm controller" >&2; exit 1; }
cat "$FAKE/queued" 2>/dev/null || true
EOF
cat > "$B/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
cat "$FAKE/gpus"
EOF
# Fake interpreter: HF sync (stdin script), torch probe (-c), campaign_status.py.
cat > "$B/python" <<'EOF'
#!/usr/bin/env bash
case "$1" in
    -) cat > "$FAKE/sync-script"; printf '%s\n' "$*" >> "$FAKE/sync-calls"
       echo "HF $2: 9 files, 5 in scope, 3 already here, 2 to download"; exit "$(cat "$FAKE/sync-rc" 2>/dev/null || echo 0)" ;;
    -c) echo "python 3.10.0 torch 2.5.1 cuda 12.4 devices ${SLURM_GPUS_ON_NODE:-0}" ;;
    tools/campaign_status.py)
        if [[ "$2" == B ]]; then echo "INCOMPLETE b_tr_qwen15_dora: missing" >&2; echo "Phase B: incomplete (1/5 commands missing/invalid)"
        else echo "Phase $2: selected scope complete; 3 commands, 3 unique artifacts"; fi ;;
    *) echo "unexpected python call: $*" >&2; exit 9 ;;
esac
EOF
cat > "$B/hf" <<'EOF'
#!/usr/bin/env bash
printf "%s\n" "$*" >> "$FAKE/hf-calls"; [[ -n "${HF_TOKEN:-}" ]] || { echo "no token passed" >&2; exit 5; }
echo "$HF_TOKEN" > "$FAKE/hf-token"; [[ -e "$FAKE/sbatch-calls" ]] && echo after >> "$FAKE/hf-order" || echo before >> "$FAKE/hf-order"
EOF
# Fake pool: completes two tasks over ~1 s, then exits with $FAKE/pool-rc.
cat > "$B/pool" <<'EOF'
#!/usr/bin/env bash
[[ ! -e "$FAKE/pool-wait" ]] || trap 'exit 75' TERM     # like pool.sh: checkpoint, exit 75
echo "fake pool on ${CUDA_VISIBLE_DEVICES:-none} root=$RCG_ROOT logs=$RCG_POOL_LOG_ROOT"
touch "$FAKE/pool-ran"
mkdir -p "$STATE/done"; touch "$STATE/done/1"; sleep 0.6; touch "$STATE/done/2"; sleep 0.6
[[ ! -e "$FAKE/pool-wait" ]] || { while :; do sleep 0.1; done; }
exit "$(cat "$FAKE/pool-rc" 2>/dev/null || echo 0)"
EOF
chmod +x "$B"/*
export STATE
fail() { echo "FAIL: $*" >&2; [[ -f "$scratch/out" ]] && tail -40 "$scratch/out" >&2; exit 1; }
reset() {
    rm -rf "$R/assets/results" "$LOGS" "$R/assets/logs/a100" "$R/.env" "$scratch"/{sbatch-calls,sbatch-env,sync-calls,sync-rc,hf-calls,hf-token,hf-order,pool-rc,pool-ran,pool-wait,queued,squeue-fail,out}
    printf '0, NVIDIA A100-SXM4-80GB, 81920\n1, NVIDIA A100-SXM4-80GB, 81920\n' > "$scratch/gpus"
    echo 'HF_TOKEN="hf_FAKEtokenFAKEtokenFAKEtokenFAKEtoken12"' > "$R/.env"
}
calls() { [[ -f "$scratch/$1" ]] && wc -l < "$scratch/$1" || echo 0; }
job() {     # run the file as the Slurm batch script of job 900 with $1 GPUs
    local rc=0 gpus="$1"; shift
    (cd "$R"; env "$@" SLURM_JOB_ID=900 SLURM_JOB_NAME=rcg_revision SLURM_GPUS_ON_NODE="$gpus" SLURM_SUBMIT_DIR="$R" \
        CUDA_VISIBLE_DEVICES=0,1 SLURM_CPUS_PER_TASK=16 bash "$JOB") > "$scratch/out" 2>&1 || rc=$?
    echo "$rc"
}

bash -n "$JOB" || fail "syntax"

# 1. Launcher: 1-4 A100s (default 2), CPUs and RAM per GPU, partition, walltime, account.
reset
for g in 1 2 4; do
    line=$(bash "$JOB" --gpus "$g" --preview 2>/dev/null)
    [[ "$line" == *"--gres=gpu:$g "* && "$line" == *"--cpus-per-task=$((8 * g)) "* && "$line" == *"--mem=$((32 * g))G "* ]] ||
        fail "preview for $g GPUs: $line"
done
line=$(bash "$JOB" --preview 2>/dev/null)
[[ "$line" == *"--gres=gpu:2 "* && "$line" == *"--partition=gpu-a100 "* && "$line" == *"--time=7-00:00:00 "* &&
   "$line" == *"--job-name=rcg_revision "* && "$line" == *"RCG_PYTHON=$B/python"* && "$line" == *" $JOB "* ]] || fail "default preview: $line"
[[ "$line" != *--account* ]] || fail "account without RCG_SPARTAN_ACCOUNT"
RCG_SPARTAN_ACCOUNT=punim0001 bash "$JOB" --preview 2>/dev/null | grep -q -- '--account=punim0001' || fail "account flag"
bash "$JOB" --gpus 3 --mem-gb 200 --time 2-12:00:00 --preview 2>/dev/null | grep -q -- '--mem=200G --time=2-12:00:00' || fail "overrides"
for bad in "--gpus 0" "--gpus 5" "--gpus x" "--mem-gb 999" "--time 8-00:00:00" "--time 12" "--bogus"; do
    rc=0; bash "$JOB" $bad --preview > /dev/null 2>&1 || rc=$?
    [[ "$rc" == 2 ]] || fail "accepted: $bad (rc $rc)"
done
[[ ! -e "$LOGS" ]] || fail "preview wrote logs"

# 2. Submit: one sbatch; not while a job is queued/running, after a stop file, without
#    judge weights, or when squeue cannot answer.
reset
bash "$JOB" --gpus 4 > "$scratch/out" 2>&1 || fail "submit"
[[ $(calls sbatch-calls) == 1 ]] && grep -q -- '--gres=gpu:4' "$scratch/sbatch-calls" && [[ -d "$LOGS" ]] || fail "submit call"
echo 555 > "$scratch/queued"; bash "$JOB" > "$scratch/out" 2>&1 || fail "queued submit rc"
[[ $(calls sbatch-calls) == 1 ]] && grep -q 'already queued/running: 555' "$scratch/out" || fail "second job submitted"
rm "$scratch/queued"; touch "$scratch/squeue-fail"
rc=0; bash "$JOB" > "$scratch/out" 2>&1 || rc=$?; [[ "$rc" == 3 && $(calls sbatch-calls) == 1 ]] || fail "blind submit"
rm "$scratch/squeue-fail"; mkdir -p "$STATE"; touch "$STATE/stop"
rc=0; bash "$JOB" > "$scratch/out" 2>&1 || rc=$?; [[ "$rc" == 2 && $(calls sbatch-calls) == 1 ]] || fail "stop file ignored"
rm "$STATE/stop"
rc=0; RCG_SKIP_INPUT_CHECK=0 bash "$JOB" > "$scratch/out" 2>&1 || rc=$?
[[ "$rc" == 2 && $(calls sbatch-calls) == 1 ]] && grep -q 'Missing judge input' "$scratch/out" || fail "missing judge weights submitted"

# 3. Full job: six sections with banners and logs, HF check before the pool, per-section
#    report, progress tables, pushes during and after the campaign, exit 0, no resubmit.
reset
[[ $(job 2) == 0 ]] || fail "job exit"
for s in 01-environment 02-hf-status 03-inputs 04-campaign 05-hf-push 06-summary; do
    [[ -s "$LOGS/900/$s.log" ]] || fail "missing section log $s"
    n="${s%%-*}" name="${s#*-}"
    grep -q "^==== \[$n\] $name | start " "$scratch/out" && grep -qE "^==== \[$n\] $name \| end .* \| exit 0 \|" "$scratch/out" ||
        fail "banners for $s"
done
[[ $(grep -c '^==== \[' "$scratch/out") == 12 ]] || fail "banner count"
grep -q 'OK: 2 x A100 80GB' "$LOGS/900/01-environment.log" || fail "environment check"
grep -q "PhucInAI/code-review-comment-generation $R assets/results/revision-a100-native/\* assets/results/a100-native-judges/\*" "$scratch/sync-calls" ||
    fail "sync scope: $(cat "$scratch/sync-calls")"
grep -q '\*.safetensors' "$scratch/sync-calls" && grep -q '\*/checkpoint-\*' "$scratch/sync-calls" || fail "sync excludes"
grep -q 'snapshot_download' "$scratch/sync-script" && grep -q 'os.path.exists' "$scratch/sync-script" || fail "sync script"
grep -q '^section A: Phase A: selected scope complete' "$scratch/out" && grep -q '^section B: Phase B: incomplete' "$scratch/out" &&
    grep -q '^    INCOMPLETE b_tr_qwen15_dora' "$scratch/out" || fail "per-section HF status"
grep -qE '^judge llama-legacy +scored files: 0' "$scratch/out" || fail "judge status"
grep -q 'OK judge weights' "$LOGS/900/03-inputs.log" || fail "inputs"
grep -q "fake pool on 0,1" "$LOGS/900/04-campaign-pool.log" && grep -q "logs=$R/assets/logs/a100/pool" "$LOGS/900/04-campaign-pool.log" || fail "pool env"
grep -q -- '-- progress ' "$scratch/out" && grep -qE '^group +done +total' "$scratch/out" || fail "progress table"
grep -q 'pool exit 0' "$scratch/out" && grep -q 'Campaign complete.' "$scratch/out" || fail "summary"
(( $(calls hf-calls) >= 2 )) || fail "expected a push during the campaign and one at the end ($(calls hf-calls))"
call=$(tail -1 "$scratch/hf-calls")
[[ "$call" == "upload-large-folder PhucInAI/code-review-comment-generation $R --repo-type dataset --include "* ]] || fail "push call: $call"
[[ "$call" == *"assets/logs/spartan/*"* && "$call" == *"--exclude */checkpoint-* */.pool/* *.lock"* && "$call" != *--private* ]] ||
    fail "push scope: $call"
grep -q 'auto-push started' "$scratch/out" || fail "auto-push not reported"
[[ $(calls sbatch-calls) == 0 ]] || fail "finished campaign resubmitted"

# 4. No token: pushes are skipped with a note, the job still succeeds.
reset; rm "$R/.env"
[[ $(job 2) == 0 && $(calls hf-calls) == 0 ]] || fail "no-token job"
grep -q 'auto-push skipped' "$LOGS/900/05-hf-push.log" || fail "no-token note"
# 5. A token inside the files to push blocks the push (public dataset).
reset; mkdir -p "$R/assets/results/revision-a100-native/x"
echo 'token=hf_abcdefghijklmnopqrstuvwxyz0123456789' > "$R/assets/results/revision-a100-native/x/leak.txt"
[[ $(job 2) == 0 && $(calls hf-calls) == 0 ]] || fail "leak pushed"
grep -q 'Possible secret' "$LOGS/900/05-hf-push.log" && grep -q 'x/leak.txt' "$LOGS/900/05-hf-push.log" || fail "leak not reported"

# 6. HF check fails (offline node): exit 3 before the pool, with the login-node command.
reset; echo 1 > "$scratch/sync-rc"
[[ $(job 2) == 3 && ! -e "$scratch/pool-ran" ]] || fail "failed HF check must stop the job before the pool"
grep -q 'revision.slurm --hf-sync-only' "$scratch/out" || fail "login-node hint"
# ...RCG_HF_SYNC=0 runs from local files only.
reset; echo 1 > "$scratch/sync-rc"
[[ $(job 2 RCG_HF_SYNC=0) == 0 && $(calls sync-calls) == 0 && -e "$scratch/pool-ran" ]] || fail "RCG_HF_SYNC=0"

# 7. Work left at the walltime: the job resubmits itself (same GPUs, after this job,
#    without leaking its own Slurm variables); not with a stop file.
reset; echo 75 > "$scratch/pool-rc"; for i in 0 1 2; do echo "$i, NVIDIA A100-SXM4-80GB, 81920"; done > "$scratch/gpus"
[[ $(job 3 RCG_SPARTAN_MEM_GB=96 RCG_SPARTAN_TIME=7-00:00:00) == 75 ]] || fail "work-left exit"
[[ $(calls sbatch-calls) == 1 ]] || fail "continuation not submitted"
grep -q -- '--gres=gpu:3 --cpus-per-task=24 --mem=96G --time=7-00:00:00' "$scratch/sbatch-calls" &&
    grep -q -- '--dependency=afterany:900' "$scratch/sbatch-calls" || fail "continuation: $(cat "$scratch/sbatch-calls")"
[[ ! -s "$scratch/sbatch-env" ]] || fail "Slurm variables leaked into the continuation: $(cat "$scratch/sbatch-env")"
reset; echo 75 > "$scratch/pool-rc"; mkdir -p "$STATE"; touch "$STATE/stop"
[[ $(job 2) == 75 && $(calls sbatch-calls) == 0 ]] && grep -q 'continuation not submitted: stop file present' "$scratch/out" || fail "stop file resubmitted"
# Only failed/blocked work: no resubmit, instructions.
reset; echo 1 > "$scratch/pool-rc"
[[ $(job 2) == 1 && $(calls sbatch-calls) == 0 ]] && grep -q 'Only failed or blocked tasks remain' "$scratch/out" || fail "blocked campaign"

# 7b. The continuation is queued before any push (the walltime may be minutes away).
reset; echo 75 > "$scratch/pool-rc"
[[ $(job 2) == 75 && $(calls sbatch-calls) == 1 ]] || fail "continuation (order run)"
[[ "$(tail -1 "$scratch/hf-order")" == after ]] || fail "final push ran before the continuation was queued"
# 7c. scancel (TERM): the pool checkpoints (75) but nothing is resubmitted or pushed at the end.
reset; touch "$scratch/pool-wait"
(cd "$R"; exec env SLURM_JOB_ID=900 SLURM_JOB_NAME=rcg_revision SLURM_GPUS_ON_NODE=2 SLURM_SUBMIT_DIR="$R" \
    CUDA_VISIBLE_DEVICES=0,1 bash "$JOB") > "$scratch/out" 2>&1 & jpid=$!
until [[ -e "$scratch/pool-ran" ]]; do sleep 0.1; done
kill -TERM "$jpid"; rc=0; wait "$jpid" || rc=$?
[[ "$rc" == 75 && $(calls sbatch-calls) == 0 ]] || fail "terminated job resubmitted (rc $rc)"
grep -q 'not submitted: the job was terminated' "$scratch/out" && grep -q 'final push skipped' "$scratch/out" || fail "termination handling"
# 7d. .env parsing (comment after the value) and any .env value in pushed files blocks the push.
reset; printf 'HF_TOKEN=hf_FAKEtokenFAKEtokenFAKEtokenFAKEtoken12  # write token\nWANDB_API_KEY="0123456789abcdef0123"\n' > "$R/.env"
[[ $(job 2) == 0 && "$(cat "$scratch/hf-token")" == hf_FAKEtokenFAKEtokenFAKEtokenFAKEtoken12 ]] || fail ".env comment parsing"
reset; printf 'WANDB_API_KEY="0123456789abcdef0123"\nHF_TOKEN=hf_FAKEtokenFAKEtokenFAKEtokenFAKEtoken12\n' > "$R/.env"
mkdir -p "$R/assets/logs/a100/wandb"; printf 'bin\0key=0123456789abcdef0123\0' > "$R/assets/logs/a100/wandb/run.wandb"
[[ $(job 2) == 0 && $(calls hf-calls) == 0 ]] && grep -q 'wandb/run.wandb' "$LOGS/900/05-hf-push.log" || fail "wandb key in a binary file pushed"
! grep -rq 0123456789abcdef0123 "$LOGS" || fail "secret value written to the logs"

# 8. Allocation checks: GPU count, A100 80GB only, 1-4 GPUs, no job arguments.
reset; printf '0, NVIDIA A100-SXM4-80GB, 81920\n' > "$scratch/gpus"
[[ $(job 2) == 2 && ! -e "$scratch/pool-ran" ]] && grep -q 'nvidia-smi shows 1 GPUs, the allocation 2' "$scratch/out" || fail "GPU count"
reset; printf '0, NVIDIA A100-PCIE-40GB, 40960\n1, NVIDIA A100-PCIE-40GB, 40960\n' > "$scratch/gpus"
[[ $(job 2) == 2 ]] && grep -q 'Every GPU must be an A100 80GB' "$scratch/out" || fail "40GB GPU accepted"
reset; [[ $(job 5) == 2 ]] || fail "5 GPUs accepted"
reset; rc=0; (cd "$R"; SLURM_JOB_ID=900 SLURM_JOB_NAME=rcg_revision bash "$JOB" --gpus 2) > "$scratch/out" 2>&1 || rc=$?
[[ "$rc" == 2 && $(calls sbatch-calls) == 0 ]] || fail "job with arguments"

# 9. Login-node HF sync and status: own log, no GPU, no submission.
reset
bash "$JOB" --hf-sync-only > "$scratch/out" 2>&1 || fail "hf-sync-only"
ls "$LOGS"/login-*/02-hf-status.log > /dev/null 2>&1 && [[ $(calls sync-calls) == 1 && $(calls sbatch-calls) == 0 ]] || fail "sync-only log/calls"
bash "$JOB" --status > "$scratch/out" 2>&1 || fail "status"

echo "PASS: launcher (1-4 GPUs, guards), six logged sections, HF status before the pool, auto-push (during/end, token, leak), offline HF, continuation, allocation checks, login-node sync/status"
