#!/usr/bin/env bash
# Download missing LLM-judge weights into assets/models/llm-judge/. Run on the SUTD
# login node (it has internet) from the repository root, with rcg-a100 activated:
#   bash scripts/pbs/fetch_judge_models.sh
# Reads HF_TOKEN from .env. Llama-3.1-8B-Instruct is gated: the HF account must be
# approved on its model page, and a fine-grained token also needs "Read access to
# contents of all public gated repos you can access". Re-running resumes.
set -euo pipefail
cd "${RCG_ROOT:-$PWD}"
[[ -f scripts/pbs/run_phase.sh ]] || { echo "Run from the repository root" >&2; exit 2; }
if [[ -f .env ]]; then set -a; source ./.env; set +a; fi
: "${HF_TOKEN:?Set HF_TOKEN (in .env or the environment)}"
command -v hf >/dev/null || { echo "'hf' not found; activate rcg-a100" >&2; exit 2; }
export HF_HUB_DISABLE_TELEMETRY=1

fetch() {
    local repo="$1" dest="$2"
    local status
    status=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 \
        -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/$repo/resolve/main/config.json")
    if [[ "$status" != 200 ]]; then
        echo "No access to $repo (HTTP $status). Request access at https://huggingface.co/$repo" >&2
        echo "and enable gated-repo read on the token at https://huggingface.co/settings/tokens" >&2
        return 1
    fi
    # Only the HF-format safetensors/tokenizer are used; skip Meta's original/ checkpoint.
    hf download "$repo" --local-dir "$dest" --exclude 'original/*'
}

[[ -s assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct/config.json ]] ||
    fetch Qwen/Qwen2.5-Coder-3B-Instruct assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
fetch meta-llama/Llama-3.1-8B-Instruct assets/models/llm-judge/Llama-3.1-8B-Instruct
echo "Judge weights ready under assets/models/llm-judge/"
