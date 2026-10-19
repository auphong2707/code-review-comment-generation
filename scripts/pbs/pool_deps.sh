#!/usr/bin/env bash
# Prerequisites of every task in the chunks.sh inventory, for out-of-order (parallel)
# execution: prints "id<TAB>dep,dep,...<TAB>admission" (deps empty for the first task).
# admission names the A100 pilot report holding the task's measured peak GPU memory
# ($PREFLIGHT_ROOT/admissions/<name>.json), "light" for CPU-side gates/summaries, or
# "exclusive" for a whole GPU. Bash only.
set -euo pipefail
cd "${RCG_ROOT:-$(dirname "${BASH_SOURCE[0]}")/../..}"
inventory=$(bash scripts/pbs/chunks.sh)

# Campaign previews give each step's exact argv, the same text chunks.sh parses.
declare -A ARGV=()
for phase in 12_campaign_a 13_campaign_b 14_campaign_c 15_campaign_d; do
    preview=$(unset RCG_PBS_STEP; DRY=1 RCG_JSON=0 RCG_PYTHON=/bin/false bash "scripts/slurm/$phase.sbatch")
    while read -r marker label rest; do
        [[ "$marker" == STEP ]] || continue
        ARGV["${label%:}"]="$rest"
    done <<< "$preview"
done

# Two training steps with the same key write the same run directory (registry
# resolve_run: results root, model, method, LR or experiment id, non-42 seed suffix).
output_key() {
    local -a words; read -ra words <<< "$1"
    [[ " ${words[*]} " == *" -m src.train "* ]] || return 0
    local root="" model="" method="" lr="" seed=42 exp="" subset="" i
    for ((i=0; i<${#words[@]}; i++)); do
        case "${words[i]}" in
            --results-root) root="${words[i+1]}" ;;
            --model) model="${words[i+1]}" ;;
            --method) method="${words[i+1]}" ;;
            --learning_rate) lr="${words[i+1]}" ;;
            --seed) seed="${words[i+1]}" ;;
            --experiment_id) exp="${words[i+1]}" ;;
            --subset) subset="${words[i+1]}" ;;
        esac
    done
    printf '%s|%s|%s|%s|%s|%s|%s' "$root" "$model" "$method" "$lr" "$seed" "$exp" "$subset"
}

# Pilot report names written by tools/a100_preflight.py and a100_inference_probe.py.
admission_name() {
    local -a words; read -ra words <<< "$1"
    local model="" method="" dataset="" i
    for ((i=0; i<${#words[@]}; i++)); do
        case "${words[i]}" in
            --model|--model_name) model="${words[i+1]}" ;;
            --method) method="${words[i+1]}" ;;
            --dataset) dataset="${words[i+1]}" ;;
        esac
    done
    case " ${words[*]} " in
        *" -m src.train "*) printf 'train_%s_%s_%s' "$model" "$method" "$dataset" ;;
        *" -m src.icl_inference "*) printf 'icl_%s_2048' "$model" ;;
        *) printf exclusive ;;
    esac
}

declare -A KEY_OWNER=() PHASE_STEPS=() BACKEND_SHARDS=() LETTER=(
    [12_campaign_a]=A [13_campaign_b]=B [14_campaign_c]=C [15_campaign_d]=D)
declare -A GATE_ID=() LABEL_ID=() BM25_FIRST=()
prepare_id=""
first_id=""
while IFS=$'\t' read -r id kind phase item; do
    [[ -n "$first_id" ]] || first_id="$id"
    case "$kind" in
        step)
            LABEL_ID["$item"]="$id"
            PHASE_STEPS["${LETTER[$phase]}"]+="${PHASE_STEPS[${LETTER[$phase]}]:+,}$id" ;;
        gate) GATE_ID["$phase"]="$id" ;;
        prepare) prepare_id="$id" ;;
        judge) BACKEND_SHARDS["$phase"]+="${BACKEND_SHARDS[$phase]:+,}$id" ;;
    esac
done <<< "$inventory"

while IFS=$'\t' read -r id kind phase item; do
    deps=()
    admission=light
    [[ "$id" == "$first_id" ]] || deps+=("$first_id")
    case "$kind" in
        step)
            argv="${ARGV[$item]:?No preview argv for $item}"
            admission=$(admission_name "$argv")
            key=$(output_key "$argv")
            if [[ -n "$key" ]]; then
                [[ -z "${KEY_OWNER[$key]:-}" ]] || deps+=("${KEY_OWNER[$key]}")
                KEY_OWNER["$key"]="$id"
            fi
            # src/icl/selection.py writes the per-dataset BM25 index pickle in place on
            # first use; later BM25 steps of that dataset wait so none reads it half-written.
            if [[ " $argv " == *" --selection_method bm25 "* && " $argv " =~ \ --dataset\ ([^ ]+)\  ]]; then
                bm25_ds="${BASH_REMATCH[1]}"
                if [[ -n "${BM25_FIRST[$bm25_ds]:-}" ]]; then deps+=("${BM25_FIRST[$bm25_ds]}")
                else BM25_FIRST["$bm25_ds"]="$id"; fi
            fi
            if [[ "$item" =~ ^(d_.+_qlora)_official$ ]]; then
                for lr in 0 1 2; do
                    deps+=("${LABEL_ID[${BASH_REMATCH[1]}_sweep_lr$lr]:?Missing sweep for $item}")
                done
            fi ;;
        gate) IFS=, read -ra steps <<< "${PHASE_STEPS[$phase]}"; deps+=("${steps[@]}") ;;
        prepare) for letter in A B C D; do deps+=("${GATE_ID[$letter]:?Missing gate $letter}"); done ;;
        judge)
            admission=judge_llama-judge_4096
            [[ "$phase" != qwen-revision ]] || admission=judge_qwen-judge_4096
            [[ "$phase" == llama-legacy ]] || deps+=("${prepare_id:?Missing judge preparation}") ;;
        summary) IFS=, read -ra shards <<< "${BACKEND_SHARDS[$phase]}"; deps+=("${shards[@]}") ;;
        phase) admission=exclusive ;;
        *) echo "Unknown task kind $kind for $id" >&2; exit 2 ;;
    esac
    joined=$(IFS=,; printf '%s' "${deps[*]}")
    printf '%s\t%s\t%s\n' "$id" "$joined" "$admission"
done <<< "$inventory"
