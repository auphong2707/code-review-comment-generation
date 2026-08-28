# H100 campaign jobs

Slurm wrappers for the revision experiments. One file per campaign, submitted in
queue order; every file is a thin, **idempotent** run list — finished runs are
skipped instantly (`--skip-if-complete`), so a job that hits the wall clock is
continued by re-submitting the same script.

```bash
mkdir -p assets/logs/slurm                        # slurm needs the dir to exist
sbatch scripts/slurm/00_preflight.sbatch          # once, before anything else
```

| Order | Script | Phase | Content | H100-h |
|---|---|---|---|---|
| 0 | `00_preflight.sbatch` | — | env report, pytest, model downloads, VRAM probes | <1 |
| 1 | `10_campaign_b.sbatch` | 13 | GaLore 7B full FT (P0), DoRA (P1), DoRA 7B×CR + GaLore seeds (P2) | 60 / 30 / 65 |
| 2 | `20_campaign_a.sbatch` | 12 | TransReview seed replicates (P0), HP-transfer grid (P1) | 110 / 12 |
| 3 | `30_campaign_c.sbatch` | 14 | ICL prompt-v2 regeneration (P0), 10-shot demo seeds (P2) | 60 / 15 |
| 4 | `40_campaign_j.sbatch` | 16 | Qwen judge + Llama-3.1 judge over new and legacy rows | 40 / 15 / 10 |
| 5 | `50_campaign_d.sbatch` | 15 | 14B QLoRA/LoRA (P1), 14B LoRA CR (P2) | 80 / 40 |

Submission order across priorities: **P0 of 1 → 2 → 3 → 4**, then
**P1 of 1 → 5 → 2 → 5 → 4**, then P2. Campaign J is re-run after every block —
it only judges artifacts that already exist and replays cached scores.

## Knobs (environment variables, all optional)

| Variable | Effect |
|---|---|
| `RUN_P0` / `RUN_P1` / `RUN_P2` | Select the priority block (defaults differ per file; see its header) |
| `DRY=1` | Print every command with its arguments, execute nothing. Works with `bash`, no GPU needed |
| `ONLY=<regex>` / `SKIP=<regex>` | Run/skip only steps whose label matches (e.g. `ONLY='seed1'`) |
| `RCG_CONDA_ENV` | Conda env to activate (default `kse`; locally `code_review`) |
| `RCG_ROOT` | Repo root, when not submitting from it |
| `WANDB_MODE` | `offline` by default so a node without egress cannot stall a 20 h run. Sync later with `wandb sync assets/wandb/offline-run-*` |
| `RCG_DOWNLOAD=1`, `RCG_PIP_INSTALL=1` | Preflight only: fetch models / install `galore-torch` + `rank-bm25` |

Per-campaign knobs (`GALORE_VARIANT`, `GALORE_LR_TR`, `LR_14B_QLORA_TR`, `SEEDS`,
`ICL_14B`, `ICL_ATTN`, `CR_EVAL_MAX`, `RUN_QWEN`/`RUN_LLAMA`/`RUN_LEGACY`, …) are
documented in each file's header comment. If the preflight reports `flash_attn
MISSING`, run campaign C with `ICL_ATTN=sdpa` (it does not change output filenames).

## Before you submit

1. **Preview the queue offline** — `DRY=1 bash scripts/slurm/20_campaign_a.sbatch`
   prints all 36 commands without touching the GPU. Do this after any edit.
2. **Read the preflight probes.** If GaLore peaks above ~75 GB, submit campaign B
   with `GALORE_VARIANT=galore_adamw_8bit_layerwise`.
3. **Edit the `#SBATCH` headers** — `--partition`, and `--gres=gpu:h100:1` if the
   cluster does not use typed gres.
4. **`assets/` must be present on the cluster** (data, legacy artifacts, judge
   models). It is gitignored, so a fresh `git clone` will not have it.

## Where things land

```
assets/results/revision/<dataset>/<stem>_official_<lr>[_seed<n>]/complete_results.json
assets/results/revision/<dataset>/sweeps/<stem>_<id>/            # 3-point LR brackets
assets/results/revision/icl/<dataset>/<model>_icl_<k>ex_<sel>[_s<seed>]_v2.json
assets/models/llm-as-judge/table-results-revision/               # Qwen judge, new rows
assets/models/llm-as-judge/table-results-llama31-legacy/         # Llama judge, frozen 54 jobs
assets/models/llm-as-judge/table-results-llama31-revision/       # Llama judge, new rows
assets/logs/slurm/<job-name>-<job-id>.out                        # slurm stdout
assets/logs/campaign/<job-name>/<step-label>.log                 # one log per run
```

Nothing writes to the frozen artifact trees (`assets/models/review-code-generation*`,
`assets/models/llm-as-judge/table-results/`); the revision roots are new siblings.

## Supporting tools

- `tools/select_sweep_lr.py` — picks the official LR from a finished 3-point bracket
  (best validation BLEU-4 in `trainer_state.json`); campaigns B and D call it
  automatically, and any `LR_*` env var overrides it.
- `tools/revision_judge_jobs.py` — scans `assets/results/revision/` and writes the
  `--jobs-file` campaign J feeds to `tools/update_llm_judge_tables.py`.
