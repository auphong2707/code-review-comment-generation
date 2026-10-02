# SUTD Mega revision campaigns (PBS Pro)

Access: **local Ubuntu → macOS `bridge` → `sutd_mega` → PBS GPU allocation**.
Follow [SERVER_SETUP.md](SERVER_SETUP.md) for the environment and inputs.
Live `qstat -Qf a100q` on 2026-10-02 reports a **100-hour maximum**, a 24-hour
default, and a per-user queued GPU resource limit of eight. Each job explicitly
requests 100 hours, one GPU, 8 CPUs and 64 GB host RAM. Unlimited walltime is not
available under that policy. Runtime requires a full A100 80 GB; the queue name
alone does not establish its VRAM capacity.

## Eight jobs total

The default launcher submits exactly **eight serial PBS jobs**, linked by
`afterok`. Each job groups smaller internal tasks under one allocation timer.
There are no additional automatically submitted jobs.

| Job | Work | Internal tasks |
|---|---|---|
| 1 | Smoke and qualification | Dependency checks, recovery/memory pilots, bounded inference integration |
| 2 | A: TransReview seeds | 54 training runs, then full-campaign completion gate |
| 3 | B: DoRA and LoRA controls | 16 commands (five reuse A outputs), then completion gate |
| 4 | C: ICL | 66 configurations, then completion gate |
| 5 | D: 14B QLoRA | Six sweep candidates and two officials, then completion gate |
| 6 | Qwen revision judge | Producer/admission checks, 48 artifact groups, final summary |
| 7 | Llama legacy judge | 18 artifact groups, final summary |
| 8 | Llama revision judge | 48 artifact groups, final summary |

The existing 267 internal tasks remain an execution/recovery inventory; they are
**not 267 PBS submissions**. `chunks.sh` derives producer commands from the canonical
Bash campaign previews; `run_bundle.sh` groups them without changing their order
or scientific scope. All D sweep results must exist before selecting an official
LR. Whole-campaign gates verify completed outputs and configuration/input identity.

Judge groups retain whole artifacts, every row, k=3, scored filenames, cache keys
and the existing three output roots. Partial manifests/summaries have unique names;
each final summary verifies scored coverage and input identity before writing its
canonical `summary.csv`. All work remains serial to avoid shared-root/cache writes.

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

This checks exactly eight submissions, all 267 internal tasks in order, identical
144 producer commands, spooled paths, dependency failures and nested signal/recovery
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

## Preserved research settings

The numerical/recovery profile remains `--memory-profile a5000`; the target is
`RCG_TARGET_GPU=a100_80gb`, with full-device admission and ≤76 GiB peak reserved
memory. No new GaLore, 7B FMFT, ordinary 14B LoRA or 14B ICL is enabled. Frozen
artifacts, metrics, specs, effective batch and judge cache keys are unchanged.
Actual server qualification remains necessary. The historical
[A5000 budget](../slurm/BUDGET.md) is not an A100 benchmark.
