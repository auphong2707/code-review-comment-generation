# A100 80GB server setup

**For SUTD Mega, use [the PBS setup guide](../pbs/SERVER_SETUP.md).** Its access
path is Ubuntu → macOS `bridge` → `sutd_mega`, with PBS `qsub`/`qstat`.
This page preserves setup/submission instructions for other Slurm clusters.

Run these commands **on the server only**, from the repository root. Local RTX 3090
work is limited to Bash/static checks for this task: no Python, pytest, imports,
model loads, environment installation or GPU workloads locally.

Use Linux x86_64, one full A100 80GB per job (not a 40GB device or MIG slice),
8 CPUs and 64GB host RAM. Check the site's partition, account, GRES and walltime
limits. The scripts request 72 hours; override with `sbatch --time=...` if needed.
The checkout, data, caches and outputs must be on persistent storage visible to
compute nodes. Run `nvidia-smi` in a GPU allocation to check the driver/device.

## Create the environment

The historical `environment.yml` is a machine-specific Python 3.13/CUDA 12.9
export. Use a new environment for this campaign. The candidate stack below is
pinned at the direct-dependency level; the smoke job must validate it on the server.
It uses SDPA and does not require FlashAttention or DeepSpeed for the retained jobs.

```bash
conda create -n rcg-a100 python=3.12 pip -y
conda activate rcg-a100
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-server.txt
python -m pip check
mkdir -p assets/logs/slurm
python -m pip freeze > assets/logs/slurm/environment-a100.txt
conda list --explicit > assets/logs/slurm/conda-a100.txt
git rev-parse HEAD > assets/logs/slurm/source-a100.txt
```

The CUDA wheel command comes from [PyTorch's previous-version installation guide](https://pytorch.org/get-started/previous-versions/).
The server driver must support that CUDA runtime; ask the administrator if it does
not. Installing a CUDA toolkit into conda does not upgrade the host driver.
Keep `requirements-server.txt` pins when adding packages. Do not overlay the old
environment export or the broadly ranged `requirements.txt` onto this environment.

Use an absolute interpreter path in all submissions so noninteractive jobs use the
same environment:

```bash
export RCG_PYTHON="$CONDA_PREFIX/bin/python"
export RCG_ROOT="$PWD"
export WANDB_MODE=offline
export HF_HOME="$PWD/assets/cache/huggingface"
```

Alternatively set both `RCG_CONDA_ENV=rcg-a100` and an absolute `CONDA_SH` pointing
to the installation's `etc/profile.d/conda.sh`. Preserve these exports in subsequent
login sessions and resubmissions. No job logs into Hugging Face or uploads results.

## Transfer inputs and prepare caches

Git transfers code and `paper/generated/cr_judge_subset.json`. Separately transfer:

- `assets/data/Trans_Review/{train,val,test}.jsonl` and
  `assets/data/Code_Review/msg-{train,valid,test}.jsonl`.
- Judge weights and tokenizers in `assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct`
  and `assets/models/llm-judge/Llama-3.1-8B-Instruct` (authorized Llama access required).
- Frozen result JSONs at their original `assets/models/...` paths used by
  `tools/update_llm_judge_tables.py`, plus all 54 legacy input JSONLs in
  `assets/models/llm-as-judge/table-results/inputs/`. Keep legacy inputs read-only.
- Base model/tokenizer caches for CodeT5, PLBART, Qwen 0.5B/7B and Qwen 14B, plus
  evaluate metric modules, BERTScore's English model, and NLTK data. Legacy adapters
  alone do not supply base weights. Preserve cache symlinks and their targets.

If caches are incomplete, the smoke job needs network access for model/metric
downloads; otherwise prepopulate caches on the server before using an offline
compute partition. NLTK resources can be provisioned on the server with:

```bash
export NLTK_DATA="$PWD/assets/cache/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4 punkt punkt_tab
```

Reserve storage for every retained run's best/latest full recovery checkpoints,
predictions, judge caches and logs, beyond the input transfer size. Do not place
recovery roots in node-local temporary storage. A hard kill may replay work since
the last committed checkpoint; the same output roots are needed to resume.

## Submit and inspect

See [README.md](README.md) for the six-job chain and recovery instructions.
Set `RCG_PARTITION` (and account/GRES if required), preview, then submit yourself:

```bash
export RCG_PARTITION=YOUR_A100_80GB_PARTITION
# export RCG_ACCOUNT=YOUR_ACCOUNT
# export RCG_GRES=gpu:a100_80gb:1  # use the site's actual type string
bash scripts/slurm/submit_revision.sh --preview
bash scripts/slurm/submit_revision.sh --submit
squeue -u "$USER"
```

`--preview` uses Bash only and submits nothing. `--submit` prints each job ID;
save those IDs. The first job must finish successfully before any training campaign
starts. Review its logs and admission JSONs; a two-step pilot checks bounded
execution, memory, metrics and recovery, not convergence or arbitrary corpus inputs.
No A100 runtime or fit claim is made until those server measurements exist.
