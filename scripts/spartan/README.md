# Revision campaign on Spartan (University of Melbourne)

[Spartan](https://rcs-knowledge-hub.atlassian.net/wiki/spaces/KB/pages/880345089/Spartan+HPC)
runs **Slurm**. The whole campaign is one job file,
[`revision.slurm`](revision.slurm). It runs on the public `gpu-a100` partition: 4 × A100 80GB,
32 cores and ~495 GB RAM per node, 7-day maximum walltime, no QoS needed. A job uses
1–4 GPUs of one node (default 2), 8 cores per GPU and 32 GB host RAM per GPU (`--mem-gb`
overrides). Past 2-GPU pools peaked at 14–17 GB.

## 1. Access

```bash
# ~/.ssh/config on your machine
Host spartan
    Hostname spartan.hpc.unimelb.edu.au
    User <spartan-username>
    ServerAliveInterval 120
```

`ssh spartan`. Keep the checkout in project storage, not in home (50 GB):
`/data/gpfs/projects/<punimXXXX>/code-review-comment-generation`. Project storage defaults
to 500 GB (`check_project_usage`). Scratch (`/data/scratch/projects/<punimXXXX>`) is
purged after about 90 days, so do not keep results there.

## 2. Code and inputs

From your machine, at the repository root (code, data, judge weights and `.env`; no results,
logs, `.git`, `.venv` or AI files):

```bash
rsync -av --partial \
  --exclude .git --exclude .venv --exclude assets/results --exclude assets/logs \
  --exclude assets/cache --exclude 'wandb/' --exclude AGENTS.md --exclude CLAUDE.md \
  --exclude .claude --exclude /ai --exclude /plans --exclude /docs/ai --exclude /scripts/ai \
  ./ spartan:/data/gpfs/projects/<punimXXXX>/code-review-comment-generation/
```

Inputs the job checks in section 3:

- `assets/data/Trans_Review/` and `assets/data/Code_Review/`.
- `paper/generated/cr_judge_subset.json` (in the code).
- Judge weights in `assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct` and
  `assets/models/llm-judge/Llama-3.1-8B-Instruct`. Fetch them on the login node with
  `bash scripts/pool/fetch_judge_models.sh` (needs `HF_TOKEN` in `.env` with gated-repo
  read access). The launcher refuses to submit while either is incomplete.
- `assets/models/llm-as-judge/table-results/inputs/` (54 legacy judge inputs, read-only).
- Base model caches (CodeT5, PLBART, Qwen 0.5B/7B/14B, metric models) under
  `assets/cache/huggingface`. If the cache is empty they download at first use, which needs
  internet on the compute node.

`.env` must stay mode 600. Its `HF_TOKEN` (write access to
`PhucInAI/code-review-comment-generation`) enables the auto-push. Without it, pushes are
skipped and the campaign still runs.

## 3. Environment (once, on the login node)

Use a conda module if `module avail` lists one (`module load Anaconda3` or similar).
Otherwise install Miniforge in project storage:

```bash
cd /data/gpfs/projects/<punimXXXX>
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$PWD/miniforge3"
source "$PWD/miniforge3/etc/profile.d/conda.sh"
conda create -n rcg-a100 python=3.12 pip -y
conda activate rcg-a100
cd code-review-comment-generation
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-server.txt
export NLTK_DATA="$PWD/assets/cache/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4 punkt punkt_tab
```

Keep the `requirements-server.txt` pins. The job only needs the interpreter path:

```bash
export RCG_PYTHON=/data/gpfs/projects/<punimXXXX>/miniforge3/envs/rcg-a100/bin/python
export RCG_SPARTAN_ACCOUNT=<punimXXXX>        # Slurm --account (optional: default project)
```

Put both lines in `~/.bashrc` so the job, which inherits your environment, and its
continuations see them.

## 4. Run

```bash
bash scripts/spartan/revision.slurm --preview          # print the sbatch command
bash scripts/spartan/revision.slurm                    # 2 x A100 (default)
bash scripts/spartan/revision.slurm --gpus 4           # 1, 2, 3 or 4 A100s
bash scripts/spartan/revision.slurm --time 2-00:00:00  # shorter walltime (default 7-00:00:00)
bash scripts/spartan/revision.slurm --status           # per-section progress + queue
```

Only one `rcg_revision` job is queued at a time. Before running, the job checks what is
already done on Hugging Face: it downloads any in-scope result missing here from the public
dataset [PhucInAI/code-review-comment-generation](https://huggingface.co/datasets/PhucInAI/code-review-comment-generation).
It never overwrites local files and skips checkpoints, resume state and weights
(`RCG_HF_WEIGHTS=1` includes weights). Then it reports done/missing per campaign A–D, the
judges and the pool. Validated finished runs are skipped (`--skip-if-complete`, judge
cache).

The first job on Spartan also re-runs the infrastructure task (pytest, A100 probes and
pilots, about 3 h), because pool state is never uploaded.

If compute nodes cannot reach huggingface.co, section 2 stops the job with exit 3. Then
run the check on the login node and submit without the download:

```bash
bash scripts/spartan/revision.slurm --hf-sync-only
RCG_HF_SYNC=0 bash scripts/spartan/revision.slurm
```

## 5. Logs

Everything is under `assets/logs/spartan/`:

| File | Content |
|---|---|
| `rcg_revision-<jobid>.out` | every section's start/end banner (time, exit code, duration), its output, progress tables every 30 min |
| `<jobid>/01-environment.log` | modules, interpreter, torch, allocated GPUs |
| `<jobid>/02-hf-status.log` | HF download and the per-section done/missing report |
| `<jobid>/03-inputs.log` | datasets, judge weights, model cache |
| `<jobid>/04-campaign.log` | progress tables |
| `<jobid>/04-campaign-pool.log` | the pool scheduler (task start/end, OOM retries, leases) |
| `<jobid>/05-hf-push.log` | every push (during the campaign and the final one) |
| `<jobid>/06-summary.log` | final table, continuation |

Task logs are in `assets/logs/a100/pool/<jobid>/chunk_<id>_<time>.log` and
`assets/logs/a100/chunk_<id>/`.

## 6. Results on Hugging Face

While the campaign runs, the job pushes `assets/results/revision-a100-native`,
`assets/results/a100-native-judges`, `assets/logs/a100` and `assets/logs/spartan` to the
same **public** dataset. It pushes after new tasks complete, at most every 20 minutes, and
once at the end. Checkpoints, `.pool` state, locks and per-row progress are excluded. Before
each push the files are scanned for token patterns and for every `.env` value, and any hit
skips the push. A failed push never fails the campaign. Files still being written can be
pushed in a partial state; the next push replaces them, and `tools/campaign_status.py`
validates every result before it counts. On your machine, `hf download
PhucInAI/code-review-comment-generation --repo-type dataset --local-dir <dir>` fetches the
results. Once on the server, `hf upload-large-folder --help` should list `--include`,
`--exclude` and `--num-workers` (huggingface-hub 0.34).

## 7. Stop, resume, failures

- Stop: `touch assets/results/revision-a100-native/.pool/stop`. Running tasks checkpoint, and
  nothing is resubmitted. Remove the file and run the launcher again to resume.
- Walltime: 30 minutes before the limit the pool checkpoints every task. The job then
  submits its own continuation (same GPUs, `--dependency=afterany:<jobid>`).
- Failed task: `--status` lists it with its log. After fixing the cause, delete
  `assets/results/revision-a100-native/.pool/failed/<id>` and run the launcher again. A failure
  blocks only its dependents.
- Rerunning finished work: if you delete a run's outputs to force a rerun, its old copy is
  still on Hugging Face, and the next job's section 2 would download it again. Submit that
  job with `RCG_HF_SYNC=0`, or delete the files from the dataset too.
- `scancel <jobid>` also stops the campaign. Tasks resume from their checkpoints in the next
  job.

## Local checks (no GPU, no Python)

```bash
bash scripts/spartan/check_job.sh     # this job file, with fake sbatch/squeue/hf/python
bash scripts/pool/check_pool.sh       # the pool scheduler, with fake tasks and Slurm
```
