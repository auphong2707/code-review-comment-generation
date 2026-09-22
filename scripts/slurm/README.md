# A100 80GB revision campaigns

Start with [SERVER_SETUP.md](SERVER_SETUP.md) to create the server environment and
transfer inputs. The six existing sbatch files are retained as a small serial queue:

| Order | Job | Default work |
|---|---|---|
| 1 | `00_07_infrastructure.sbatch` | Dependency/unit/mock checks; 35 training recovery pilots; 3 ICL and 2 judge memory pilots; small real ICL/judge pipelines |
| 2 | `12_campaign_a.sbatch` | 54 TransReview runs: 18 configurations × seeds 42, 1, 2 |
| 3 | `13_campaign_b.sbatch` | 8 DoRA runs and 8 matching LoRA controls |
| 4 | `14_campaign_c.sbatch` | 66 ICL v2 runs |
| 5 | `15_campaign_d.sbatch` | 14B QLoRA: 3-point sweep → official, both datasets |
| 6 | `16_campaign_j.sbatch` | Qwen revision judging; Llama legacy/revision judging |

Each job requests one GPU, 8 CPUs, 64GB host RAM and up to 72 hours. The default
launcher chains all six with `afterok` dependencies. This serializes shared LoRA
controls and holds judging until all producers succeed. A failed or interrupted
smoke job cannot release campaigns. Six job definitions may need more than six
allocations if work exceeds walltime.

## Local review: no Python

Only Bash/static checks are permitted locally for this task, including on the RTX
3090. Do not run pytest, Python dry runs, imports, mocks or model workloads locally.

```bash
for script in scripts/slurm/*.sh scripts/slurm/*.sbatch; do bash -n "$script"; done
DRY=1 RCG_JSON=0 RCG_PYTHON=/bin/false bash scripts/slurm/12_campaign_a.sbatch
bash scripts/slurm/submit_revision.sh --preview
```

Plain `DRY=1 RCG_JSON=0` prints commands without Python, writes or conda activation.
`RCG_JSON=1` invokes Python and is reserved for the server's completion checker.
Real wrappers require a Slurm allocation and reject GPUs other than the selected
hardware target. No RTX 3090 qualification or runnability claim follows from shell checks.

## Launch on the server

After setup, export the environment and site settings from SERVER_SETUP.md:

```bash
bash scripts/slurm/submit_revision.sh --preview
bash scripts/slurm/submit_revision.sh --submit
```

The launcher defaults to preview. Only the user's explicit `--submit` calls sbatch.
It accepts `RCG_PARTITION`, `RCG_ACCOUNT`, `RCG_QOS` and `RCG_GRES`; use the site's
A100 80GB resource type. No fixed partition name can be portable across clusters.
Run from the repository root or set absolute `RCG_ROOT`. The helper uses that path
or `SLURM_SUBMIT_DIR`, so scripts work when copied to Slurm's spool directory.

The first job exercises training, optimizer state, validation, checkpoint interruption
and reload, generation and metrics. It also runs full-length synthetic inference
pilots and real two-row ICL / three-row judge integration checks. Training and
inference admissions require peak reserved VRAM ≤76 GiB on a full A100 80GB.
Reports bind to source, software, driver and target hardware. Changes require
requalification. Source/settings must remain fixed while queued jobs are running.

These are bounded smoke tests, not a guarantee that every full-corpus input works.
Judge inputs over `JUDGE_MAX_INPUT_TOKENS` fail explicitly; increase that limit only
after rerunning matching judge pilots. `RCG_PROBES=0` or filtered execution exits
75 in real jobs, so an incomplete smoke cannot satisfy an `afterok` dependency.

## Research scope and conservative settings

The historical CLI name `--memory-profile a5000` is retained as the numerical and
recovery policy: microbatch 1, effective batch preserved by accumulation, generation
batch 1, SDPA, checkpointing, CPU BERTScore and durable progress. The Slurm hardware
target is independently `RCG_TARGET_GPU=a100_80gb`. This intentionally preserves
the retained research comparisons; it does not tune throughput to fill 80GB.

The existing excluded scope stays excluded: 7B FMFT (missing DeepSpeed config),
GaLore (unqualified optimizer/recovery path), ordinary 14B LoRA and 14B ICL
(not enabled in this conservative profile). The latter exclusions are policy limits,
not claims that those workloads cannot fit on an A100. They need a separate
qualification extension before being added. DoRA/QLoRA are not substitutes for
missing full-FT/LoRA evidence. Later analysis/manuscript phases are outside this queue.

Five B controls reuse matching A outputs. A seed 42 uses the canonical leaf name.
Optional A `RUN_P1=1` adds ten grid commands (eight new outputs); B `RUN_P2=1`
adds two CR DoRA/control pairs; C `RUN_P2=1` adds twelve ICL runs. A accepts `SEEDS`.
The full-chain launcher rejects scope overrides; use individual submissions for
these options and preserve the same scope during resume and completion checks.

## Recovery and dependencies

USR1 is sent ten minutes before walltime; the shell forwards it to the process
group and waits for committed progress before exiting 75. `--requeue` permits
administrator/scheduler requeueing; it does not automatically continue timed-out work.
Resubmit the same script, checkout, settings and persistent output roots.

If an upstream job fails, downstream `afterok` dependencies do not automatically
follow a replacement job ID. After resubmitting, either retarget pending dependencies
with `scontrol update JobId=<pending-id> Dependency=afterok:<replacement-id>`,
or cancel the obsolete pending chain and submit a fresh chain yourself. Check the
actual queue state first. Do not leave both chains able to write the same outputs.
Reusing a fresh complete chain validates/skips finished runs.

Training commits recovery state every `RCG_RECOVERY_STEPS` (default 100) optimizer
steps. Test/ICL predictions persist per row and judge results per cache entry.
Hard kills can replay uncommitted work; validation must finish before a safe stop.
Measure checkpoint/validation time and increase the warning interval when needed.
An incomplete LR bracket cannot select an official LR; failed commands stop the job.

## Paths and advanced scope

| Variable | Default |
|---|---|
| `RCG_RESULTS_ROOT` | `assets/results/revision-a100` |
| `RCG_PREFLIGHT_ROOT` | `assets/results/a100-preflight` |
| `RCG_JUDGE_ROOT` | `assets/results/a100-judges` |
| `RCG_LOG_ROOT` | `assets/logs/a100` (per-phase directories) |
| `JUDGE_MAX_INPUT_TOKENS` | 4096 |
| `CR_EVAL_MAX` | 2000 validation rows for D's CR official |
| `WANDB_MODE` | offline |

Use separate roots from A5000 runs and frozen inputs. Training leaf names and
result/judge schemas are unchanged. Judge roots remain separate by backend and
legacy/revision input set. Slurm logs go to `assets/logs/slurm`; create it before
individual sbatch submissions (the launcher creates it for actual submission).

J checks A/B/C/D by default. Server-only completion command:

```bash
RCG_TARGET_GPU=a100_80gb "$RCG_PYTHON" tools/campaign_status.py A \
  --results-root assets/results/revision-a100
```

For optional producer scopes, set `RCG_A_RUN_P1=1 RCG_B_RUN_P2=1 RCG_C_RUN_P2=1`
on J's individual submission; producer jobs themselves use their own `RUN_P*`.
Do not apply global `RUN_P2=1` to D. `J_REQUIRE_PHASES='A B C'` requests a partial
judge pass and cannot claim D complete. `RUN_REVISION=0` selects legacy-only judging;
`RUN_QWEN`, `RUN_LLAMA`, `RUN_LEGACY` control the passes (default all on).

[BUDGET.md](BUDGET.md) preserves historical A5000 timing assumptions. Those numbers
are not an A100 estimate. Calibrate elapsed training, generation, metrics and
checkpoint times on the server before estimating the new campaign duration.
