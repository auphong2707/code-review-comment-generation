# SUTD Mega revision campaigns (PBS Pro)

Access: **local Ubuntu → macOS `bridge` → `sutd_mega` → PBS GPU allocation**.
Follow [SERVER_SETUP.md](SERVER_SETUP.md) for the environment and inputs.
Live `qstat -Qf a100q` on 2026-10-02 reports a **100-hour maximum**, a 24-hour
default, and a per-user queued GPU resource limit of eight. Each job explicitly
requests 100 hours, one GPU, 8 CPUs and 64 GB host RAM. Unlimited walltime is not
available under that policy. Runtime requires a full A100 80 GB; the queue name
alone does not establish its VRAM capacity.

## Default: one parallel pool job

Verified limits for this account on a100q (2026-10-13): at most **4 running GPUs /
4 running jobs**, **8 GPUs queued+running** (`max_queued_res.ngpus`), 100 h walltime;
the only A100 node is `aimc-gna1` (8 × A100 80 GB, shared). The pool runs all
226 tasks inside **one** submission:

- `pool_deps.sh` lists each task's prerequisites (smoke first; the D official after its
  sweeps; one BM25 index build per dataset; gates, judge preparation, shards, summaries
  in order).
- `pool.sh` starts every ready task with exactly one visible GPU, running the unchanged
  `run_chunk.sh --body` command, and packs several tasks per GPU by their A100 pilot
  peak (`a100-native-preflight/admissions`, + 2 GiB, ≤ 70 GiB and ≤ 4 tasks per GPU). A
  task that runs out of memory (CUDA OOM or a host-memory kill) is retried once alone.
- Each task gets the job's deadline (`RCG_DEADLINE`). A task that cannot finish an epoch
  evaluation or its corpus metrics before then commits a checkpoint and is parked until
  the next job instead of being killed mid-stage; test predictions are saved as they are
  generated.
- 30 minutes before the walltime it sends USR1 to every task (checkpoint, exit 75),
  releases their leases and **queues its own continuation** with the same resources.
  If PBS refuses (quota), the login-node cron keeper resubmits within 30 minutes.
- State lives in `assets/results/revision-a100-native/.pool/` (`done/`, `failed/`, `lease/`);
  any later pool continues from it. A failure blocks only its dependents.

```bash
# login node, repository root, rcg-a100 environment prepared (SERVER_SETUP.md)
export RCG_PYTHON=/scratch/congthanh_le/conda/envs/rcg-a100/bin/python
bash scripts/pbs/submit_pool.sh --preview          # print the qsub command
bash scripts/pbs/submit_pool.sh --gpus 2 --jobs 2  # up to two 2-GPU pool jobs (4 running GPUs)
crontab -e   # add once (keeper fallback):
# */30 * * * * bash /home/users/congthanh_le/JSS/code-review-comment-generation/scripts/pbs/pool_keeper.sh >> /home/users/congthanh_le/JSS/code-review-comment-generation/assets/logs/a100/pool/keeper.log 2>&1
bash scripts/pbs/pool_status.sh                    # progress, running tasks, failures
```

`--gpus 2` (default) fits beside other jobs within the 4-GPU running cap; `--gpus 4`
is fastest but waits for four free GPUs on the shared node. Only one pool is queued at
a time. Logs: `assets/logs/a100/pool/<jobid>/chunk_<id>_<time>.log` plus the PBS
`rcg_pool.o<jobid>` file.

**Command-form submission.** `submit_pool.sh` runs `qsub ... -- /bin/bash
<repo>/scripts/pbs/pool.pbs` instead of spooling the script: since its 2026-10-09
reboot, aimc-gna1's PBS spool directory (`/var/spool/pbs/mom_priv/jobs`, mode 0644)
makes every spooled batch script exit 126 with `.SC: Permission denied` within a
second (jobs 19492, 19561). Interactive jobs and command-form jobs are unaffected. Do
not `qsub scripts/pbs/pool.pbs` directly; the eight-job fallback below still spools
its scripts and fails on that node until the cluster admins restore the directory.

**Failures.** `pool_status.sh` lists failed tasks and their logs. After fixing the
cause, delete `assets/results/revision-a100-native/.pool/failed/<id>` and run
`submit_pool.sh` again. The keeper never resubmits a finished (verdict 0) or
failure-blocked (1) campaign; a pool that dies during setup (verdict 2, e.g. GPU
allocation mismatch, a bad `RCG_PYTHON`, or a job that ended before its script ran,
`rc=unstarted`) is retried, but after three setup errors in a row the keeper waits for
you (`.pool/errors`; a manual `submit_pool.sh` re-arms it).

**Stopping.** `touch assets/results/revision-a100-native/.pool/stop` makes the running pool
checkpoint its tasks and exit within a minute, and stops all resubmission (pool,
keeper, launcher). Remove the file and run `submit_pool.sh` to resume. `qdel` alone is
not a stop: tasks get only PBS's ~10 s kill delay (progress since the last checkpoint
is redone) and the keeper would requeue the campaign. `qsig -s USR1 <jobid>` drains one
job and lets it queue its continuation.

**Quota note.** A continuation is queued while its own job is still running, so a
`--gpus 4` pool needs 8 queued+running GPUs at that moment — the whole quota. If any
other job of yours is queued or running then, qsub is refused and the keeper submits
within 30 minutes of the old job ending. `--gpus 2` does not have this gap.
`RCG_PYTHON` (absolute path) is required: jobs are submitted without `-V`, so a conda
environment activated in your shell is not inherited. Do not run the eight-job chain
below against the same roots at the same time.

## Fallback: eight serial jobs

The fallback launcher submits exactly **eight serial PBS jobs**, linked by
`afterok`. Each job groups smaller internal tasks under one allocation timer.
There are no additional automatically submitted jobs.

| Job | Work | Internal tasks |
|---|---|---|
| 1 | Smoke and qualification | Dependency checks, recovery/memory pilots, bounded inference integration |
| 2 | A: TransReview seeds 1 and 2 | 28 training runs, then full-campaign completion gate |
| 3 | B: DoRA | 5 TransReview runs (controls are paper LoRA runs), then completion gate |
| 4 | C: ICL | 66 configurations (CodeReviewer on the 2,000-row subset), then completion gate |
| 5 | D: 14B QLoRA | Three TransReview sweep candidates and one official, then completion gate |
| 6 | Qwen revision judge | Producer/admission checks, 48 artifact groups, final summary |
| 7 | Llama legacy judge | 18 artifact groups, final summary |
| 8 | Llama revision judge | 48 artifact groups, final summary |

The 226 internal tasks are an execution/recovery inventory; they are
**not 226 PBS submissions**. `chunks.sh` derives producer commands from the canonical
Bash campaign previews; `run_bundle.sh` groups them without changing their order
or scientific scope. All D sweep results must exist before selecting an official
LR. Whole-campaign gates verify completed outputs and configuration/input identity.

Judge groups retain whole artifacts, every row, k=1 (all passes), 16 rows per judge call
(`JUDGE_BS`), scored filenames, cache keys
and the existing three output roots. Partial manifests/summaries have unique names;
each final summary verifies scored coverage and input identity before writing its
canonical `summary.csv`. In this layout all work is serial; the pool above instead
relies on its prerequisite graph and per-shard judge caches for concurrent safety.

**Eight jobs is an allocation layout, not a promise of completion within 800 GPU
hours.** A, C and judge jobs may exceed 100 hours and require another allocation.
Recovery preserves committed training checkpoints, ICL row journals and judge
caches. No automatic resubmission or cancellation occurs. Do not run an old chain
and this chain against the same roots concurrently.

## Preview and submit

From the prepared repository on `sutd_mega`, after clearing any obsolete queue
and confirming no earlier chain is writing the same roots:

```bash
export RCG_PBS_QUEUE=a100q
export RCG_PBS_PROJECT=20260203
export RCG_PBS_WALLTIME=100:00:00
bash scripts/pbs/bundles.sh                        # eight-job inventory
bash scripts/pbs/submit_revision.sh --preview      # exactly eight qsub commands
bash scripts/pbs/submit_revision.sh --submit       # submits those eight jobs
```

Save the printed full PBS IDs, including server suffix. Other queued jobs may
consume part of your eight-GPU queued quota. `--count 1` through `--count 8` can
submit fewer jobs; `--start N --afterok PREVIOUS_ID` continues at job N (1–8).
The launcher does not count other jobs or reserve capacity. If qsub fails, it stops
and reports the job where submission stopped; inspect the accepted IDs and `qstat`
before retrying. Never blindly repeat a partially accepted submission.

Walltime overrides must exceed ten minutes and cannot exceed 100 hours. The default
launcher rejects scope-changing environment variables such as `ONLY`, `SKIP` and
`RUN_P*`. Keep the source, inputs, output roots and environment fixed while jobs
are queued or running. Legacy whole-phase `.pbs` entries and `chunk.pbs` remain
advanced entry points, but are not used by the default launcher.

Local verification is Bash-only (no local Python or GPU work):

```bash
bash scripts/pbs/check_handoff.sh
```

This checks exactly eight submissions, all 226 internal tasks in order, identical
103 A100 producer commands, spooled paths, dependency failures and nested signal/recovery
handling. Server smoke runs pytest, including mock judge partition/summary tests.

## Monitor

On the SUTD login node:

```bash
qstat -u "$USER"
qstat -f JOB_ID
qstat -Qf a100q
myhistory
```

From Ubuntu, connect using the bridge's server credentials:

```bash
ssh -t bridge 'ssh -t sutd_mega'
```

PBS merges stdout/stderr into `rcg_b<N>.o<job-number>` in the submission directory;
that file may arrive only at job end. Live grouped logs use
`assets/logs/a100/bundle_<N>/`; producer logs also remain under their phase folders,
and judge/gate logs under `chunk_<ID>/`. For example:

```bash
tail -f assets/logs/a100/00_07/infra_pytest.log
```

## Stop and resume

One `run_phase.sh` timer surrounds the entire job. It reads the allocation's actual
walltime and sends USR1 ten minutes before the deadline. The group signals its
active campaign shell, which signals its Python worker and waits for recovery
state to be committed. Inner tasks do not start fresh 100-hour timers. A save longer
than the warning window can still be killed. Incomplete work exits nonzero and
cannot release dependent jobs.

After a failure/timeout, inspect `qstat -f`, logs and `myhistory`. PBS may delete
dependent jobs; cancel any surviving obsolete successors before rebuilding the
chain. Resubmit the same job number and fixed configuration. For example:

```bash
# User-run recovery: N is the failed job (1–8), all earlier jobs verified complete.
qsub -v RCG_PBS_BUNDLE=N scripts/pbs/bundle.pbs
# If N < 8, resubmit remaining jobs using the replacement's full PBS ID:
bash scripts/pbs/submit_revision.sh --submit --start NEXT_N --afterok REPLACEMENT_ID
```

Completed training/ICL runs validate and skip; unfinished runs resume committed
state. Judging reuses its cache. Producer and final-summary gates still reject
missing output. Recovery can require more than eight allocations over the whole
experiment, though the default chain contains only eight jobs. If a successful
predecessor ID has expired from PBS history, use the direct `qsub` form above only
after verifying all preceding work, then chain later jobs to the new ID.

## Research settings

Every producer runs `--memory-profile a100` (native batch sizes, batched generation,
GPU metrics; see `scripts/slurm/README.md`); the target is `RCG_TARGET_GPU=a100_80gb`,
with full-device admission and ≤76 GiB peak reserved memory per task (chunk 1 pilots
every configuration at its campaign batch sizes). No GaLore, 7B FMFT, ordinary 14B
LoRA or 14B ICL is enabled. Frozen artifacts, metrics, specs and judge cache keys are
unchanged. Seed 42 and the DoRA controls are the archived paper runs
(`paper/generated/seed42_paper_reuse.json`).
