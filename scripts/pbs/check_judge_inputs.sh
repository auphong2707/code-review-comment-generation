#!/usr/bin/env bash
# Fail before queueing anything if judge weights are missing or partial: the smoke task
# otherwise spends hours on training probes before its judge probes reach them.
# Sourced by the launchers from the repository root; RCG_SKIP_INPUT_CHECK=1 is for tests.
check_judge_inputs() {
    [[ "${RCG_SKIP_INPUT_CHECK:-0}" != 1 ]] || return 0
    local missing=() model file shard
    local root="${RCG_JUDGE_MODEL_ROOT:-assets/models/llm-judge}"
    for model in "$root/Qwen2.5-Coder-3B-Instruct" "$root/Llama-3.1-8B-Instruct"; do
        for file in config.json tokenizer_config.json; do
            [[ -s "$model/$file" ]] || missing+=("$model/$file")
        done
        if [[ -s "$model/model.safetensors.index.json" ]]; then
            for shard in $(grep -oE '"[^"]+\.safetensors"' "$model/model.safetensors.index.json" | tr -d '"' | sort -u); do
                [[ -s "$model/$shard" ]] || missing+=("$model/$shard")
            done
        elif ! compgen -G "$model/*.safetensors" >/dev/null; then
            missing+=("$model/*.safetensors")
        fi
    done
    if (( ${#missing[@]} )); then
        printf 'Missing judge input: %s\n' "${missing[@]}" >&2
        echo "Fetch with: bash scripts/pbs/fetch_judge_models.sh (login node)" >&2
        return 2
    fi
}
