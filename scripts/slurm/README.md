# A100 80GB revision campaign definitions

These six files define the campaign steps. On Spartan they are run by the single job
file [`scripts/spartan/revision.slurm`](../spartan/README.md), not submitted one by one.
Its pool (`scripts/pool/pool.sh`) splits them into 226 tasks (`scripts/pool/chunks.sh`),
runs them in dependency order on the job's 1–4 A100s, and executes each step through
`scripts/pool/run_chunk.sh`, which sources the matching file below. The `#SBATCH`
lines are not used by the pool.

| File | Default work |
|---|---|
| `00_07_infrastructure.sbatch` | Dependency/unit/mock checks; training recovery pilots; ICL and judge memory pilots; small real ICL/judge pipelines |
| `12_campaign_a.sbatch` | TransReview seed replicates (seeds 1, 2; seed 42 is the archived paper run) |
| `13_campaign_b.sbatch` | TransReview DoRA runs (controls are the paper LoRA runs) |
| `14_campaign_c.sbatch` | ICL v2 runs (CodeReviewer on the fixed 2,000-row subset) |
| `15_campaign_d.sbatch` | 14B QLoRA: 3-point sweep → official, TransReview |
| `16_campaign_j.sbatch` | Qwen revision judging; Llama legacy/revision judging |

`common.sh` is the shared runtime (batch sizes, admissions, `run_step`).

## Local review: no Python

Only Bash/static checks are permitted locally, including on the RTX 3090. Do not run
pytest, Python dry runs, imports, mocks or model workloads locally.

```bash
for script in scripts/slurm/*.sh scripts/slurm/*.sbatch; do bash -n "$script"; done
DRY=1 RCG_JSON=0 RCG_PYTHON=/bin/false bash scripts/slurm/12_campaign_a.sbatch
```

Plain `DRY=1 RCG_JSON=0` prints commands without Python, writes or conda activation.
`RCG_JSON=1` invokes Python and is reserved for the server's completion checker.
Real wrappers require a Slurm allocation and reject GPUs other than the selected
hardware target. No RTX 3090 qualification or runnability claim follows from shell checks.

## Infrastructure task

The infrastructure task (first in the pool) exercises training, optimizer state, validation, checkpoint interruption
and reload, generation and metrics. It also runs full-length synthetic inference
pilots and real two-row ICL / three-row judge integration checks. Training and
inference admissions require peak reserved VRAM ≤76 GiB on a full A100 80GB.
Reports bind to source, software, driver and target hardware. Changes require
requalification. Source/settings must remain fixed while queued jobs are running.

These are bounded smoke tests, not a guarantee that every full-corpus input works.
Judge inputs over `JUDGE_MAX_INPUT_TOKENS` fail explicitly; increase that limit only
after rerunning matching judge pilots. `RCG_PROBES=0` or filtered execution exits
75 in real jobs, so an incomplete smoke never releases its dependents.

## Research scope and the A100 profile

Every producer runs `--memory-profile a100`: the native training protocol of the frozen
specs (batch sizes as in the paper runs, no accumulation rewrite, gradient checkpointing
only where the spec enables it), batched validation/test generation (`GEN_BS` in
`common.sh`) and ICL (`ICL_BS`), SDPA, BERTScore on the GPU, and durable progress
(full-state checkpoints every 100 steps, test predictions journaled as they are
generated). The hardware target is `RCG_TARGET_GPU=a100_80gb`. Under the pool a task
that cannot finish an epoch evaluation or its corpus metrics before the allocation
ends checkpoints first and resumes in the next job.

Scope (D-049): A trains seeds 1 and 2 of the 18 TransReview configurations (7B: LoRA
only); seed 42 is the archived paper run. B is five TransReview DoRA runs whose
controls are the paper LoRA runs (`paper/generated/seed42_paper_reuse.json`). C runs
CodeReviewer ICL on the fixed 2,000-row subset (`--sample-index-file`). D trains 14B
QLoRA on TransReview only. Revision outputs are judged once per judge (`--judge-k 1`).

The existing excluded scope stays excluded: 7B FMFT (missing DeepSpeed config),
GaLore (unqualified optimizer/recovery path), ordinary 14B LoRA and 14B ICL
(not enabled in this conservative profile). The latter exclusions are policy limits,
not claims that those workloads cannot fit on an A100. They need a separate
qualification extension before being added. DoRA/QLoRA are not substitutes for
missing full-FT/LoRA evidence. Later analysis/manuscript phases are outside this queue.

Five B controls reuse matching A outputs. A seed 42 uses the canonical leaf name.
Optional A `RUN_P1=1` adds ten grid commands (eight new outputs); B `RUN_P2=1`
adds two CR DoRA/control pairs; C `RUN_P2=1` adds twelve ICL runs. A accepts `SEEDS`.
The pool runs the default scope only (run_chunk.sh rejects scope overrides);
preserve the same scope during resume and completion checks.

## Recovery

Before the walltime the pool sends USR1 to every task. The task's shell forwards it to the
worker and waits for committed progress before exiting 75. The job then submits its own
continuation, which resumes every task from its checkpoint. Keep the checkout, settings and
output roots fixed while jobs are queued or running.

Training commits recovery state every `RCG_RECOVERY_STEPS` (default 100) optimizer
steps. Test/ICL predictions persist per row and judge results per cache entry.
Hard kills can replay uncommitted work; validation must finish before a safe stop.
An incomplete LR bracket cannot select an official LR; a failed task blocks only its
dependents.

## Paths and advanced scope

| Variable | Default |
|---|---|
| `RCG_RESULTS_ROOT` | `assets/results/revision-a100-native` |
| `RCG_PREFLIGHT_ROOT` | `assets/results/a100-native-preflight` |
| `RCG_JUDGE_ROOT` | `assets/results/a100-native-judges` |
| `RCG_LOG_ROOT` | `assets/logs/a100` (per-phase directories) |
| `JUDGE_MAX_INPUT_TOKENS` | 4096 |
| `WANDB_MODE` | offline |

Use separate roots from earlier runs and frozen inputs. Training leaf names and
result/judge schemas are unchanged. Judge roots remain separate by backend and
legacy/revision input set. Job and section logs go to `assets/logs/spartan/` (scripts/spartan/README.md).

J checks A/B/C/D by default. Server-only completion command:

```bash
RCG_TARGET_GPU=a100_80gb "$RCG_PYTHON" tools/campaign_status.py A \
  --results-root assets/results/revision-a100-native
```

Optional scopes (`RCG_A_RUN_P1=1`, `RCG_C_RUN_P2=1`, `J_REQUIRE_PHASES`, `RUN_REVISION`,
`RUN_QWEN`, `RUN_LLAMA`, `RUN_LEGACY`) are read by the campaign files; the pool runs the
default scope and rejects them. Do not apply global `RUN_P2=1` to D.

Calibrate elapsed training, generation, metrics and checkpoint times on the server
before estimating the campaign duration.
