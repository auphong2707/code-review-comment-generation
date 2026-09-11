# A5000 server campaigns

Exactly six jobs cover phases 00–07 and 12–16. Each requests one GPU and at most
72 hours. Submit from the repository root, using your site's A5000 partition/GRES.
The runtime rejects other GPUs. Phases 08–11 and 17–20 are outside this migration.

| Script | Selected work by default |
|---|---|
| `00_07_infrastructure.sbatch` | Code/mock checks, 35 training lifecycle pilots, 3 ICL and 2 judge memory pilots |
| `12_campaign_a.sbatch` | 54 TransReview runs: 18 configurations × seeds 42, 1, 2 |
| `13_campaign_b.sbatch` | 8 DoRA runs and 8 matching A5000 LoRA controls |
| `14_campaign_c.sbatch` | 66 ICL v2 runs, actual input cap 2048 tokens |
| `15_campaign_d.sbatch` | 14B QLoRA: 3-point sweep then official run, both datasets |
| `16_campaign_j.sbatch` | Qwen over revision outputs; Llama over legacy and revision outputs, CR subset |

A seed-42 run uses the canonical name without a seed suffix under the A5000 profile.
Five B controls reuse A outputs. A's optional `RUN_P1=1` adds ten grid commands,
eight new outputs. B's `RUN_P2=1` adds two CR 7B DoRA/control pairs. C's
`RUN_P2=1` adds twelve 10-shot demo-seed runs. These options must be preserved when
resubmitting or checking a selected scope. A accepts `SEEDS` (default `42 1 2`).

Ordinary 14B BF16 LoRA, 7B FMFT and GaLore are unavailable in this profile.
Their missing research evidence must be disclosed; QLoRA/DoRA are separate methods.
All remaining combinations require a matching measured admission report before
campaign training/inference. No A5000 memory or throughput claim follows from CPU tests.

## Preview and server launch

Local verification is limited to unit tests, mocks and tiny CPU smoke tests. Do not
run pretrained-model smoke tests or these jobs locally. A safe command-only preview is:

```bash
DRY=1 RCG_PYTHON=.venv/bin/python bash scripts/slurm/12_campaign_a.sbatch
```

`DRY=1` does not activate conda, load models, read secrets or create output directories.
On the server, provision the declared environment, datasets, legacy artifacts and
local judge models first. These files are gitignored. No script installs packages,
logs into HF, uploads results, or submits another job. Use an already activated
Python environment, `RCG_PYTHON`, or both `CONDA_SH` and `RCG_CONDA_ENV`.

The user launches the server jobs, for example (replace `YOUR_A5000_PARTITION`):

```bash
mkdir -p assets/logs/slurm
sbatch --partition=YOUR_A5000_PARTITION scripts/slurm/00_07_infrastructure.sbatch
# After qualification completes, submit 12, 13, 14, 15; submit 16 after producers finish.
sbatch --partition=YOUR_A5000_PARTITION scripts/slurm/12_campaign_a.sbatch
```

Phase 00–07 includes real server pilots by default. `RCG_PROBES=0` selects code/mock
checks only and does not qualify campaigns. Pilots exercise optimizer, validation,
checkpoint interruption/reload and generation, with maximum-length inputs. Admission
requires measured peak reserved memory <=22 GiB on an A5000. Reports bind to source
and environment; changing either requires requalification. Judge input over its admitted
budget fails explicitly without truncating the rubric. Increase
`JUDGE_MAX_INPUT_TOKENS` only after rerunning matching judge pilots at that budget.

## Recovery across allocations

Slurm sends USR1 to the batch shell ten minutes before the limit. The shell forwards
it to the active process group, waits for a safe stop and exits 75. Resubmit the
**same script, settings, checkout and output roots** to continue. `--requeue` permits
requeueing but does not automatically resubmit a timed-out job.

Training resumes optimizer/scheduler/RNG state from the latest committed checkpoint
(default every 100 optimizer steps). Validation boundaries finish evaluation/save before
stopping. Test and ICL generations persist per row; judge generations persist per cache
entry. A hard kill can replay work since the last committed checkpoint/row/call,
including an unfinished validation pass. Measure save and validation time on the server
and increase the warning interval if ten minutes is insufficient. Checkpoint files must
stay on persistent storage. Concurrent writers to the same training/ICL run are rejected.

Completed outputs are validated before skipping. An incomplete sweep cannot select an
official LR. Failures stop dependent steps immediately. `ONLY=<regex>` and `SKIP=<regex>`
are useful for diagnosis; filtered real execution exits 75 and does not claim campaign
completion. A later unfiltered submission validates/skips completed work.

Read-only producer completion checks use the exact selected sbatch run lists:

```bash
python tools/campaign_status.py A --results-root assets/results/revision-a5000
```

Phase J requires A, B, C and D by default. `J_REQUIRE_PHASES='A B C'` permits an explicit
partial pass; this does not mean D or the whole revision is complete. Preserve producer
`RUN_P*`/`SEEDS` scope when checking. Judge generation/resource errors fail the job;
resubmission reuses cached calls. `RUN_REVISION=0` selects legacy-only judging;
`RUN_QWEN`, `RUN_LLAMA`, `RUN_LEGACY` select backend/input passes (default all enabled).

## Paths and settings

| Variable | Default / purpose |
|---|---|
| `RCG_ROOT` | Submission directory; repository root |
| `RCG_RESULTS_ROOT` | `assets/results/revision-a5000` |
| `RCG_PREFLIGHT_ROOT` | `assets/results/a5000-preflight` |
| `RCG_JUDGE_ROOT` | `assets/results/a5000-judges` |
| `RCG_LOG_ROOT` | `assets/logs/a5000`, with per-phase subdirectories |
| `RCG_RECOVERY_STEPS` | 100 optimizer steps |
| `JUDGE_MAX_INPUT_TOKENS` | 4096, guarded before GPU transfer |
| `CR_EVAL_MAX` | 2000 validation rows for the 14B CR official only |
| `WANDB_MODE` | offline |

Training uses `<root>/<dataset>/<stem>_official_<lr>[_seed<n>]`; sweeps are in
`<root>/<dataset>/sweeps`. ICL uses `<root>/icl/<dataset>`. Judge roots are separate
`qwen-revision`, `llama-legacy`, `llama-revision` directories. Frozen artifacts remain
inputs. Agent tests use scratch directories; server experiments produce their own
isolated output trees. The planning scenarios in BUDGET.md must be calibrated with server throughput.

## Review and planning budget (2026-09-11)

See [BUDGET.md](BUDGET.md) for counts, explicit timing assumptions and calibration.
The default queue is approximately 1,260–6,460 occupied hours in the planning
scenarios; all supported optional work is 1,565–7,980 hours. These are unmeasured
scenarios, not an A5000 performance claim.

Slurm copies the submitted script to its spool directory. All wrappers locate
`common.sh` through `RCG_ROOT` or `SLURM_SUBMIT_DIR`, so the complete tracked checkout
must exist on the execution node at that path. `environment.yml` includes pytest
for the infrastructure gate; provision it before submission.

For J after all supported optional blocks, leave global RUN_P* unset and set:

```bash
RCG_A_RUN_P1=1 RCG_B_RUN_P2=1 RCG_C_RUN_P2=1 \
  sbatch --partition=YOUR_A5000_PARTITION scripts/slurm/16_campaign_j.sbatch
```

The completion checker accepts `RCG_<A|B|C|D>_<RUN_P0|RUN_P1|RUN_P2|SEEDS>`
overrides. These select expected coverage only: launch producers with their own
RUN_P*/SEEDS flags. Preserve these settings when resubmitting J. A global RUN_P2=1
would enable D's explicitly unsupported ordinary LoRA request.

## Transfer preparation

Git carries code, environment.yml, scripts, and paper/generated/cr_judge_subset.json.
Data and model files require a separate server transfer; do not add assets or ignored
AI plans to git. Before qualification, preserve these relative paths:

- `assets/data/Trans_Review/{train,val,test}.jsonl` and
  `assets/data/Code_Review/msg-{train,valid,test}.jsonl` (about 4.18 GB decimal total).
- Local judge weights/tokenizers at `assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct`
  and `assets/models/llm-judge/Llama-3.1-8B-Instruct` (latter requires authorized access).
- Legacy artifacts at their original `assets/models/...` paths for the mock gate,
  plus `assets/models/llm-as-judge/table-results/inputs/` for legacy Llama judging.
- Base-model, tokenizer, evaluate/BERTScore and NLTK caches, or provision them on
  the server before starting timed jobs. Base models use Hub IDs; copying legacy
  adapters alone does not supply their base weights. `HF_HOME` defaults to
  `assets/cache/huggingface`; keep symlink targets when transferring caches.

Create `assets/logs/slurm` before sbatch (Slurm opens output before shell startup).
Use persistent storage for revision roots, recovery checkpoints, judge caches and
logs, with room for best/latest checkpoints for every retained run; the input-only
transfer size is not the final storage requirement. Keep legacy inputs read-only.
A failed conditional 14B pilot stops 00–07 before later pilots. To qualify independent
ICL/judge work use `ONLY='probe_icl_|probe_qwen-judge|probe_llama-judge'` on the same
infra file; a filtered job exits 75 deliberately. Submit only campaigns whose own
admission reports pass; unavailable D requires explicitly partial J prerequisites.
