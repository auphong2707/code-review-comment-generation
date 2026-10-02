# SUTD Mega setup through the macOS bridge

The connection path is **Ubuntu → macOS `bridge` → `sutd_mega`**. The final
host is the SUTD PBS login node; Python/GPU qualification runs inside a PBS
allocation. The bridge relays the SSH connection; transferred files go directly
to `sutd_mega`.

## Connect from Ubuntu

```bash
ssh bridge
# On the macOS bridge (where the sutd_mega SSH alias is configured):
ssh sutd_mega
# Now on the SUTD login node:
hostname
command -v qsub qstat
myquota
pbsnodes -aSj
```

One interactive command from Ubuntu is `ssh -t bridge 'ssh -t sutd_mega'`.
The guide gives `login1.gpucluster.sutd.edu.sg` (`aimc-hn1`) as the login node;
retain the existing `sutd_mega` alias and its actual account/authentication settings.
An alias configured on the Mac is not automatically available on Ubuntu.
`ssh -J bridge sutd_mega` only works if Ubuntu also has a correct target HostName,
User and usable target authentication. The nested SSH path does not require that.

## Storage and allocation

The supplied January 2026 guide lists shared home `/home/users/$USER` (100 GB)
and shared scratch `/scratch/$USER` (1 TB, files unused for 180 days may be purged).
The active checkout is
`/home/users/congthanh_le/JSS/code-review-comment-generation` (home, 100 GB quota,
not purged). About 54 GiB of input assets plus new recovery checkpoints can exceed
that quota; if `myquota` runs short, move `assets/models` to `/scratch/$USER` and
symlink it back (scratch is purged after 180 unused days; back results up).
`/raid` is node-local (60-day purge), unsuitable for the only copy of recovery
state across jobs. Check actual quotas with `myquota`.

On the login node, request an interactive allocation for environment validation:

```bash
qsub -I -P 20260203 -q a100q -l select=1:ncpus=8:ngpus=1:mem=64gb -l walltime=02:00:00
# Inside the allocated compute node:
cd /home/users/congthanh_le/JSS/code-review-comment-generation
nvidia-smi
```

Run these commands **on the server only**, from the repository root. Local RTX 3090
work is limited to Bash/static checks for this task: no Python, pytest, imports,
model loads, environment installation or GPU workloads locally.

Use Linux x86_64, one full A100 80GB per job (not a 40GB device or MIG slice),
8 CPUs and 64GB host RAM. The guide specifies PBS queue `a100q`, but does not
confirm A100 memory capacity; the runtime admission check must pass on the device.
The PBS scripts request 100 hours, the verified a100q maximum. The launcher rejects larger requests.
The checkout, data, caches and outputs must be on persistent storage visible to
compute nodes. Run `nvidia-smi` in a GPU allocation to check the driver/device.

## Create the environment

Run `bash assets/rsync_to_server.txt` on Ubuntu first (see the transfer section
below); the checkout and `SOURCE_COMMIT` must exist on the server before this step.

The historical `environment.yml` is a machine-specific Python 3.13/CUDA 12.9
export. Use a new environment for this campaign. The candidate stack below is
pinned at the direct-dependency level; the smoke job must validate it on the server.
It uses SDPA and does not require FlashAttention or DeepSpeed for the retained jobs.

The server has no conda by default. Full Anaconda is unnecessary; install
Miniforge (conda-forge only) once on the login node, in home, without admin rights.
First check `module avail` for an existing conda/miniforge module and use it instead.

```bash
cd $HOME
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda --version
```

If the login node has no internet, download the installer on Ubuntu and copy it
with `scp Miniforge3-Linux-x86_64.sh sutd_mega:` first. The environment and caches
use the 100 GB home quota; after installing, run `conda clean -a -y` and
`python -m pip cache purge`.

Each later login: `source "$HOME/miniforge3/etc/profile.d/conda.sh"` before
`conda activate rcg-a100`.

```bash
cd /home/users/congthanh_le/JSS/code-review-comment-generation
conda create -n rcg-a100 python=3.12 pip -y
conda activate rcg-a100
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-server.txt
python -m pip check
mkdir -p assets/logs/pbs
python -m pip freeze > assets/logs/pbs/environment-a100.txt
conda list --explicit > assets/logs/pbs/conda-a100.txt
cat SOURCE_COMMIT > assets/logs/pbs/source-a100.txt  # rsync copies no .git
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
export NLTK_DATA="$PWD/assets/cache/nltk_data"
```

Alternatively set both `RCG_CONDA_ENV=rcg-a100` and an absolute `CONDA_SH` pointing
to the installation's `etc/profile.d/conda.sh`. Preserve these exports in subsequent
login sessions and resubmissions. After exiting the interactive job, activate
`rcg-a100` and repeat these exports from the same checkout on the login node
before submitting. Do not carry an interactive job's `CUDA_VISIBLE_DEVICES`
override into submission; PBS must assign the new job's GPU. No job logs into
Hugging Face or uploads results.

## Transfer inputs and prepare caches

Create and update the server checkout (code, `paper/generated/cr_judge_subset.json`
and local assets) by running `bash assets/rsync_to_server.txt` on Ubuntu. It creates
the target directory, excludes `.git`, and writes `SOURCE_COMMIT` (`-dirty` marks
uncommitted changes). Transfers are user-run. Do not copy Ubuntu's `.venv`, macOS environments, `.env`, or ignored AI
documentation into the server checkout. Separately transfer:

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
compute queue. NLTK resources can be provisioned on the server with:

```bash
export NLTK_DATA="$PWD/assets/cache/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4 punkt punkt_tab
```

Reserve storage for every retained run's best/latest full recovery checkpoints,
predictions, judge caches and logs, beyond the input transfer size. Do not place
recovery roots in node-local temporary storage. A hard kill may replay work since
the last committed checkpoint; the same output roots are needed to resume.

Run rsync on **Ubuntu**, with `sutd_mega` as the destination and `bridge` as the
SSH jump host. Files are stored on the server; no intermediate copy is made on
the Mac. The server account is `congthanh_le` (the PBS project ID `20260203` is not a
username).

For this ProxyJump command, define the target alias in Ubuntu's `~/.ssh/config`
using the real server account and authentication settings. Keep the existing
`bridge` alias. The target alias on the Mac alone is insufficient:

```sshconfig
Host sutd_mega
    HostName login1.gpucluster.sutd.edu.sg
    User congthanh_le
    ProxyJump bridge
```

Ubuntu must be able to authenticate to the server through the bridge. rsync runs
on Ubuntu and `sutd_mega`; the Mac only forwards SSH traffic. After the target
directory exists on `sutd_mega`, preview and then transfer from the local repo root:

```bash
rsync -a --progress --dry-run -e 'ssh -J bridge' assets/data/ \
    sutd_mega:/home/users/congthanh_le/JSS/code-review-comment-generation/assets/data/
rsync -a --progress -e 'ssh -J bridge' assets/data/ \
    sutd_mega:/home/users/congthanh_le/JSS/code-review-comment-generation/assets/data/
```

Repeat for the required judge models, frozen JSONs/inputs and caches listed above,
preserving their repository-relative paths. `-a` preserves cache symlinks; include
their targets and ensure the links resolve on the server. No `--delete` is needed.
The September cleanup left about
54 GiB of local assets, not the earlier 284+ GB inventory; this is input footprint,
not the free space required for new recovery checkpoints. Fetch new results back
with the same `-e 'ssh -J bridge'` transport, using `sutd_mega:/home/users/congthanh_le/JSS/...` as the
source and a separate local staging directory as the destination before inspecting
them; preserve frozen local results. The archived quickstart under `assets/`
predates this corrected bridge/revision workflow; use this document for launch.

## Submit and inspect

See [README.md](README.md) for the split-job batches and recovery instructions.
On the SUTD login node, set the PBS queue/project, preview, then submit yourself:

```bash
export RCG_PBS_QUEUE=a100q
export RCG_PBS_PROJECT=20260203
bash scripts/pbs/submit_revision.sh --preview
bash scripts/pbs/submit_revision.sh --submit
qstat -u "$USER"
myjob
```

`--preview` uses Bash only and submits nothing. `--submit` submits at most eight
chunks, prints each job ID and the continuation command for the next batch.
Save those IDs; wait for batch success and available queued quota before continuing.
The first job must finish successfully before any training campaign
starts. Review its logs and admission JSONs; a two-step pilot checks bounded
execution, memory, metrics and recovery, not convergence or arbitrary corpus inputs.
No A100 runtime or fit claim is made until those server measurements exist.
