#!/usr/bin/env bash
#SBATCH --job-name=rcg_preflight
#SBATCH --partition=main             # EDIT: cluster partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1            # EDIT: plain gpu:1 if the cluster has no typed gres
#SBATCH --time=08:00:00
#SBATCH --output=assets/logs/slurm/%x-%j.out
#SBATCH --error=assets/logs/slurm/%x-%j.err
#
# jss-revision preflight — run this ONCE on the H100 before any campaign.
# Verifies the environment, fetches the models the campaigns need, and probes
# peak VRAM for the two runs that can OOM (GaLore 7B, 14B). Nothing here trains.
#
#     mkdir -p assets/logs/slurm
#     sbatch scripts/slurm/00_preflight.sbatch                 # checks + probes
#     RCG_DOWNLOAD=1 sbatch scripts/slurm/00_preflight.sbatch  # + model downloads
#     RCG_PIP_INSTALL=1 sbatch scripts/slurm/00_preflight.sbatch  # + missing deps
#
# Read the probe output before submitting campaign B / D:
#   * GaLore peak > ~75 GB  -> add --galore-variant galore_adamw_8bit_layerwise
#   * 14B peak   > ~75 GB   -> lower --gen-batch-size or use --eval-max-samples
# ---------------------------------------------------------------------------
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DOWNLOAD="${RCG_DOWNLOAD:-0}"
PIP_INSTALL="${RCG_PIP_INSTALL:-0}"
SNAPSHOT='import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], local_dir=(sys.argv[2] if len(sys.argv) > 2 else None)))'

# --- 1. environment report -------------------------------------------------
banner "1. environment"
run_step "env_versions" python -c '
import importlib
for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes",
             "galore_torch", "rank_bm25", "flash_attn", "datasets", "evaluate", "wandb"]:
    try:
        module = importlib.import_module(name)
        print("  %-16s %s" % (name, getattr(module, "__version__", "?")))
    except Exception as error:
        print("  %-16s MISSING (%s)" % (name, type(error).__name__))
import torch
print("  cuda available   %s" % torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device           %s" % torch.cuda.get_device_name(0))
    print("  capability       %s" % (torch.cuda.get_device_capability(0),))
'

# Campaign B needs galore-torch, campaign C needs rank-bm25 (both absent from the
# legacy env). Install only when explicitly asked — compute nodes are often offline.
if [[ "$PIP_INSTALL" == "1" ]]; then
    run_step "env_install_deps" pip install --no-input "galore-torch>=1.0" "rank-bm25>=0.2.2"
fi

# --- 2. code checks (cheap; catches a bad merge before 20 GPU-hours) --------
banner "2. code checks"
run_step "check_pytest" python -m pytest -q
run_step "check_dryrun_galore" env TRAIN_DRY_RUN=1 ./run_training.sh qwen25 galore fixed 5e-05
run_step "check_dryrun_dora" env TRAIN_DRY_RUN=1 ./run_training.sh codet5 dora fixed 0.001
run_step "check_dryrun_14b" env TRAIN_DRY_RUN=1 ./run_training.sh qwen25-14b qlora fixed 5e-05
run_step "check_icl_help" python -m src.icl_inference --help

# --- 3. model downloads (login node / node with egress) --------------------
banner "3. models  (RCG_DOWNLOAD=$DOWNLOAD)"
if [[ "$DOWNLOAD" == "1" ]]; then
    # training backbone for campaign D — cached under $HF_HOME, loaded by repo id
    run_step "fetch_qwen25_14b" python -c "$SNAPSHOT" Qwen/Qwen2.5-Coder-14B-Instruct
    # second judge (D-019); gated repo -> needs HUGGINGFACE_TOKEN from .env
    run_step "fetch_judge_llama31" python -c "$SNAPSHOT" \
        meta-llama/Llama-3.1-8B-Instruct assets/models/llm-judge/Llama-3.1-8B-Instruct
else
    echo "skipped (set RCG_DOWNLOAD=1 on a node with internet access)"
    echo "  campaign D needs Qwen/Qwen2.5-Coder-14B-Instruct in \$HF_HOME"
    echo "  campaign J needs assets/models/llm-judge/Llama-3.1-8B-Instruct (gated)"
fi

# --- 4. judge cache seeding ------------------------------------------------
# Reuse the legacy Qwen judge cache so already-scored (prediction, reference)
# pairs are never re-generated. cp -n: the frozen file is read-only, and an
# existing revision cache is never clobbered.
banner "4. judge cache"
LEGACY_CACHE=assets/models/llm-as-judge/table-results/qwen_judge_cache.jsonl
REVISION_CACHE=assets/models/llm-as-judge/table-results-revision/qwen_judge_cache.jsonl
if [[ -f "$LEGACY_CACHE" ]]; then
    run_step "seed_judge_cache" bash -c \
        "mkdir -p '$(dirname "$REVISION_CACHE")' && cp -n '$LEGACY_CACHE' '$REVISION_CACHE' && ls -la '$REVISION_CACHE'"
else
    echo "legacy judge cache not found at $LEGACY_CACHE — skipping (nothing to reuse)"
fi

# --- 5. memory probes ------------------------------------------------------
# One forward/backward/optimizer step at the real batch size; prints
# PROBE RESULT {...} with peak_vram_reserved_gb. No checkpoints, no results dir.
banner "5. memory probes"
# --protocol archived is the default; spelled out because it sets the TR batch size
run_step "probe_galore_qwen25_tr" python -m src.train --model qwen25 --method galore --probe \
    --protocol archived --dataset trans_review --learning_rate 5e-05
run_step "probe_galore_qwen25_cr" python -m src.train --model qwen25 --method galore --probe \
    --protocol archived --dataset acra_by_ls_pt --learning_rate 5e-05
run_step "probe_14b_qlora_tr" python -m src.train --model qwen25-14b --method qlora --probe \
    --protocol archived --dataset trans_review --learning_rate 5e-05
run_step "probe_14b_qlora_cr" python -m src.train --model qwen25-14b --method qlora --probe \
    --protocol archived --dataset acra_by_ls_pt --learning_rate 5e-05
run_step "probe_14b_lora_tr" python -m src.train --model qwen25-14b --method lora --probe \
    --protocol archived --dataset trans_review --learning_rate 5e-05

finish_report
