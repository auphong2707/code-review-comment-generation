# SUTD Mega revision campaigns (PBS Pro)

Access: **local Ubuntu → macOS `bridge` → `sutd_mega` login node → PBS GPU job**.
Follow [SERVER_SETUP.md](SERVER_SETUP.md) for the SSH hops, storage, inputs and
environment. SUTD uses `qsub` and `qstat`/`myjob`; `squeue` is a Slurm command.

The January 2026 user guide (`assets/Research_GPU_Cluster-User-Guide_v2.2.pdf`,
sections “Queue Configuration”, “Job Submission”, “Resource: Storage”) documents
`a100q`, `select=1:ncpus=8:ngpus=1:mem=64gb`-style requests, and PBS monitoring.
The guide lists no queue walltime cap; these jobs retain a 72-hour request for
bounded recovery. Confirm current policy and quota on the server. It does not
establish that an allocated A100 has 80 GB; the runtime checks full-device capacity.

## Six serial jobs

| Order | PBS entry | Default work |
|---|---|---|
| 1 | `00_07_infrastructure.pbs` | Dependency/unit/mock checks; 35 training recovery pilots; 3 ICL and 2 judge memory pilots; bounded real ICL/judge integration |
| 2 | `12_campaign_a.pbs` | 54 TransReview runs: 18 configurations × seeds 42, 1, 2 |
| 3 | `13_campaign_b.pbs` | 8 DoRA runs and 8 matching LoRA controls |
| 4 | `14_campaign_c.pbs` | 66 ICL v2 runs |
| 5 | `15_campaign_d.pbs` | 14B QLoRA: 3-point sweep → official, both datasets |
| 6 | `16_campaign_j.pbs` | Qwen revision judging; Llama legacy/revision judging |

Each requests one GPU, 8 CPUs, 64 GB host RAM and 72 hours. The launcher submits
one `afterok` chain: smoke → A → B → C → D → J. Failure or an incomplete smoke
blocks downstream work. Work exceeding walltime needs another allocation.

PBS entries execute the existing `scripts/slurm/*.sbatch` files **as Bash scripts**;
`#SBATCH` comments have no effect in this path. This keeps one scientific run list
for previews, completion checks and both schedulers. No Slurm tools or fabricated
`SLURM_JOB_ID` are needed. Keep the complete checkout fixed while jobs are queued.

## Preview and submit

On local Ubuntu, Bash checks only; no Python, installation, GPU work or submission:

```bash
for script in scripts/pbs/*.sh scripts/pbs/*.pbs scripts/slurm/*.sh scripts/slurm/*.sbatch; do
    bash -n "$script" || break
done
DRY=1 RCG_JSON=0 RCG_PYTHON=/bin/false RCG_ROOT="$PWD" \
    bash scripts/pbs/12_campaign_a.pbs
bash scripts/pbs/submit_revision.sh --preview
bash scripts/pbs/check_handoff.sh  # local fake-scheduler/parity/recovery gate
```

On `sutd_mega`, from the prepared repository root, after setting the environment
in SERVER_SETUP.md:

```bash
export RCG_PBS_QUEUE=a100q
export RCG_PBS_PROJECT=20260203
# export RCG_PBS_WALLTIME=72:00:00          # default in each PBS entry
bash scripts/pbs/submit_revision.sh --preview
bash scripts/pbs/submit_revision.sh --submit
```

The default is preview. Only `--submit` contacts PBS; save the six printed IDs.
The launcher and all six PBS entries default to project `20260203`. An explicit
`RCG_PBS_PROJECT` overrides the launcher's default; direct `qsub` submissions use
the entry's `#PBS -P 20260203` unless overridden with `qsub -P`.
The launcher rejects partial/scope overrides and Slurm site settings. Optional
scope uses individual `qsub` submissions and matching completion-check settings,
as described in the [shared research scope](../slurm/README.md#research-scope-and-conservative-settings).
For example, on the server: `qsub scripts/pbs/00_07_infrastructure.pbs` runs only smoke.
The `#PBS -V` directive exports the prepared server environment to each job.

## Monitor (replacement for squeue)

Run on the SUTD login node:

```bash
myjob
qstat -u "$USER"
qstat -f JOB_ID                  # full ID, e.g. 123456.aimc-hn1
myhistory
pbsnodes -aSj
myquota
myusage
```

From Ubuntu, use the bridge's SSH configuration for the second hop:

```bash
ssh -t bridge 'ssh -t sutd_mega'
# Now on sutd_mega: qstat -u "$USER"
```

PBS merges stdout/stderr into its default `rcg_<phase>.o<job-number>` output in
the submission directory. PBS may deliver that file only at job end. Live step
logs are on shared storage under `assets/logs/a100/<phase>/`; for example:

```bash
tail -f assets/logs/a100/00_07/infra_pytest.log
```

`qstat -f` reports output paths, requested walltime and dependency state. Review
smoke logs/admission reports before drawing conclusions about GPU fit or runtime.

## Stop, resume and dependencies

`run_phase.sh` queries the allocation's `Resource_List.walltime` using `qstat -f`,
then uses GNU `timeout` to send USR1 ten minutes before that walltime. The existing
campaign shell forwards USR1 to its active worker and waits for committed progress.
Timeout/incomplete execution exits nonzero, so it cannot satisfy `afterok`. The
warning is a best-effort recovery window, not a guarantee that a slow save finishes.
PBS termination or node failure can still replay work since the last checkpoint.
Do not change a running job's walltime: the timer is calculated at startup.

Resubmit the same phase with the same source/settings/output roots. Training
checkpoints every 100 optimizer steps by default; ICL journals rows and judging
caches entries. A full replacement chain skips outputs that pass completion checks.

PBS may delete dependent jobs after a failed prerequisite; inspect `myjob`,
`qstat -f` and `myhistory` first. For a surviving queued dependent job, the user can
retarget it with `qalter -W depend=afterok:REPLACEMENT_ID PENDING_ID`. Otherwise,
submit the missing phases again with explicit dependencies, or cancel remaining
obsolete queued jobs using `qdel JOB_ID` before submitting a fresh chain. These
are user-run recovery actions. Never leave two chains writing the same roots.
No automatic submission, cancellation or requeue is performed by these helpers.

## Preserved research settings

The numerical/recovery profile remains `--memory-profile a5000`; the hardware
target remains `RCG_TARGET_GPU=a100_80gb`, with full-device admission and ≤76 GiB
peak reserved memory. There is no new qualification of GaLore, 7B FMFT, ordinary
14B LoRA or 14B ICL. Frozen artifacts, metrics, schemas and default judge keys are
unchanged. New outputs retain the isolated revision-a100/preflight/judge roots.
See the [shared scope and paths](../slurm/README.md#paths-and-advanced-scope).

Server package checks, GPU qualification and experiments are still pending.
Local Bash gates cannot establish runtime compatibility or A100 throughput.
The old [A5000 budget](../slurm/BUDGET.md) is historical.

PBS option/dependency reference: [PBS Professional Reference Guide](https://help.altair.com/2024.1.0/PBS%20Professional/PBSReferenceGuide2024.1.pdf).
