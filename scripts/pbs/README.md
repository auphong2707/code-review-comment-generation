# SUTD Mega revision campaigns (PBS Pro)

Access: **local Ubuntu → macOS `bridge` → `sutd_mega` → PBS GPU allocation**.
Follow [SERVER_SETUP.md](SERVER_SETUP.md) for the environment and inputs.
Live `qstat -Qf a100q` on 2026-10-02 reports a **100-hour maximum**, a 24-hour
default, and a per-user queued GPU resource limit of eight. The scripts explicitly
request 100 hours; unlimited walltime is unavailable under that queue policy.
Each job requests one A100, 8 CPUs and 64 GB host RAM. Runtime still requires a
full A100 80 GB device; the queue name alone does not establish its VRAM capacity.

## Smaller jobs

The default launcher now schedules **267 serial chunks**, with `afterok`
dependencies. This is the same research scope as the previous six campaigns:

| Chunk IDs | Work | Size per compute job |
|---|---|---|
| 1 | Infrastructure, recovery/memory pilots and bounded inference integration | One qualification job |
| 2–55, then gate 56 | A: 54 TransReview seed runs | One model/method/seed training run |
| 57–72, then gate 73 | B: 16 LoRA/DoRA commands | One training run; five controls reuse A outputs |
| 74–139, then gate 140 | C: 66 ICL configurations | One model/dataset/prompt configuration |
| 141–148, then gate 149 | D: six 14B QLoRA sweeps and two officials | One sweep candidate or official run |
| 150 | Check all producers, build judge inventory and check judge admission | Preparation gate |
| 151–198, then summary 199 | Qwen revision judging | 48 groups, normally 2–3 artifacts each |
| 200–217, then summary 218 | Llama legacy judging | 18 groups, three artifacts each |
| 219–266, then summary 267 | Llama revision judging | 48 groups, normally 2–3 artifacts each |

`chunks.sh` derives producer commands from the existing Bash campaign previews;
it does not maintain a second list of scientific runs. All three D sweep results
must exist before selecting the official LR. A chunk's success means only its
own work finished; separate full-campaign gates check complete outputs and identities.
The six original `.pbs` entries remain available for advanced whole-phase runs,
now with 100-hour limits, but the default launcher uses `chunk.pbs`.

Judge groups partition whole artifacts, retaining every row, k=3, cache keys,
scored filenames and the existing three output roots. Shard manifests/summaries
have distinct suffixes. A final summary job verifies scored coverage and input
identity, then writes the canonical `manifest.jsonl` and `summary.csv` for that
pass. Judge groups and all other jobs remain serial to avoid shared-cache writes.
Do not run the old campaign and new chunks against the same roots concurrently.

Splitting reduces work per allocation; it cannot guarantee that any individual
training run or judge group fits 100 hours. Training checkpoints, ICL row journals
and judge caches retain the existing recovery behavior. No automatic resubmission
or job cancellation occurs.

## Preview and submit in batches

From the prepared repository on `sutd_mega`, after cancelling any obsolete queued
chain and confirming no previous chain is writing the same roots:

```bash
export RCG_PBS_QUEUE=a100q
export RCG_PBS_PROJECT=20260203
export RCG_PBS_WALLTIME=100:00:00
bash scripts/pbs/chunks.sh                         # numbered work inventory
bash scripts/pbs/submit_revision.sh --preview      # all 267 commands, no submission
bash scripts/pbs/submit_revision.sh --submit       # first eight chunks only
```

The launcher prints the accepted job IDs and the exact continuation command:

```bash
# Example only: replace LAST_JOB_ID with the last accepted ID from your batch.
bash scripts/pbs/submit_revision.sh --submit --start 9 --afterok LAST_JOB_ID
```

Wait for the previous batch to succeed and queued capacity to become available,
then run that continuation command. Keep the full PBS ID including server suffix.
Use `--count 1` through `--count 8` to reduce a batch if other jobs consume quota.
The launcher does not count other jobs or reserve queue capacity; a rejected qsub
stops immediately and reports the chunk where submission stopped. Preserve its
printed IDs and inspect `qstat` before continuing; never blindly repeat a batch.
PBS must still recognize a supplied dependency ID. If it has expired from server
history, use the single-chunk recovery procedure below after inspecting completion.

`--preview --start N --count 8` shows one batch. Noninitial batch submissions require
`--afterok`; dependencies also chain every job within a batch. Walltime overrides
must exceed ten minutes and cannot exceed 100 hours. Scope-changing environment
variables such as `ONLY`, `SKIP` and `RUN_P*` are rejected by the default launcher.
Keep source, inputs, output roots and environment fixed for the entire campaign;
chunk IDs depend on this source's canonical run inventory.

Local verification is Bash-only (no local Python or GPU work):

```bash
bash scripts/pbs/check_handoff.sh
```

This checks exact producer command coverage, spooled entries, batch bounds,
dependencies, submission failures and timed recovery. The server qualification
job runs pytest, including mock judge partition and final-summary coverage tests.

## Monitor

Run on the SUTD login node:

```bash
qstat -u "$USER"
qstat -f JOB_ID
qstat -Qf a100q
myhistory
```

From Ubuntu, connect through the bridge's server credentials:

```bash
ssh -t bridge 'ssh -t sutd_mega'
```

PBS merges stdout/stderr into `rcg_c<ID>.o<job-number>` in the submission directory;
that file may arrive only at job end. Live producer logs remain under
`assets/logs/a100/<phase>/`. Judge and gate logs use
`assets/logs/a100/chunk_<ID>/`. For example:

```bash
tail -f assets/logs/a100/00_07/infra_pytest.log
```

## Stop and resume

`run_phase.sh` reads the allocation's actual walltime and sends USR1 ten minutes
before the deadline. The campaign forwards it to the worker and waits for recovery
state to be committed. Timeout/incomplete work exits nonzero and cannot release
successor jobs. A save longer than the warning window can still be killed.

After a failed/timed-out chunk, inspect `qstat -f`, logs and `myhistory`. PBS may
delete dependent jobs; cancel any surviving obsolete successors before rebuilding
the dependency chain. Resume the same chunk ID and fixed configuration. For example:

```bash
# User-run recovery example: N is the failed chunk, all predecessors verified complete.
qsub -v RCG_PBS_CHUNK=N scripts/pbs/chunk.pbs
# Then continue at N+1 using the replacement job's full ID:
bash scripts/pbs/submit_revision.sh --submit --start NEXT_N --afterok REPLACEMENT_ID
```

Reserve queued capacity for that replacement too (use `--count 7` or less when
needed). Downstream full-campaign gates still reject missing producer artifacts.
For an initial smoke failure, restart from chunk 1. Training uses full-state
checkpoints; ICL resumes from row journals; judge work reuses per-pass caches.

## Preserved research settings

The numerical/recovery profile remains `--memory-profile a5000`; the target is
`RCG_TARGET_GPU=a100_80gb`, with full-device admission and ≤76 GiB peak reserved
memory. No new GaLore, 7B FMFT, ordinary 14B LoRA or 14B ICL is enabled. Frozen
artifacts, metrics, specs, effective batch and judge cache keys are unchanged.
Actual server qualification remains necessary. The historical
[A5000 budget](../slurm/BUDGET.md) is not an A100 benchmark.
