#!/usr/bin/env bash
# Eight serial PBS jobs total, each grouping resumable internal tasks.
set -euo pipefail
mode=--preview; start=1; count=8; after=""; count_set=0
while (( $# )); do
    case "$1" in
        --preview|--submit) mode="$1"; shift ;;
        --start|--count|--afterok)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --start) start="$2" ;;
                --count) count="$2"; count_set=1 ;;
                --afterok) after="$2" ;;
            esac
            shift 2 ;;
        *) echo "Usage: $0 [--preview|--submit] [--start N] [--count 1..8] [--afterok JOB_ID]" >&2; exit 2 ;;
    esac
done
[[ "$start" =~ ^[1-9][0-9]*$ && "$count" =~ ^[1-8]$ ]] || { echo "Invalid start/count" >&2; exit 2; }
[[ -z "$after" || "$after" =~ ^[0-9]+(\.[A-Za-z0-9_.-]+)?$ ]] || { echo "Invalid dependency ID" >&2; exit 2; }
if [[ "$mode" == --submit && "$start" != 1 && -z "$after" ]]; then
    echo "Continuing a batch requires --afterok with its preceding successful job ID" >&2; exit 2
fi
cd "${RCG_ROOT:-$PWD}"
[[ -f scripts/pbs/run_phase.sh ]] || { echo "Run from the repository root" >&2; exit 2; }
export RCG_ROOT="$PWD" RCG_TARGET_GPU=a100_80gb
bash scripts/slurm/submit_revision.sh --preview >/dev/null
for key in RCG_PARTITION RCG_ACCOUNT RCG_QOS RCG_GRES RCG_PBS_STEP RCG_PBS_CHUNK RCG_PBS_BUNDLE; do
    [[ -z "${!key:-}" ]] || { echo "Unset $key for the PBS launcher" >&2; exit 2; }
done
walltime="${RCG_PBS_WALLTIME:-100:00:00}"
[[ "$walltime" =~ ^([0-9]{1,3}):([0-5][0-9]):([0-5][0-9])$ ]] || { echo "Walltime must be HH:MM:SS" >&2; exit 2; }
seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
(( seconds > 600 && seconds <= 360000 )) || { echo "Walltime must exceed 600s and not exceed 100h" >&2; exit 2; }
# Fail before queueing anything if judge weights are missing or partial; bundle 1
# otherwise spends hours on training probes before its judge probes reach them.
if [[ "$mode" == --submit && "${RCG_SKIP_INPUT_CHECK:-0}" != 1 ]]; then
    missing=()
    for model in assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct assets/models/llm-judge/Llama-3.1-8B-Instruct; do
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
        exit 2
    fi
fi
options=(-V -q "${RCG_PBS_QUEUE:-a100q}" -P "${RCG_PBS_PROJECT:-20260203}" -l "walltime=$walltime")
inventory=$(bash scripts/pbs/bundles.sh)
total=$(wc -l <<< "$inventory")
(( start <= total )) || { echo "Start exceeds $total jobs" >&2; exit 2; }
stop=$((start + count - 1))
[[ "$mode" != --preview || "$count_set" != 0 ]] || stop="$total"
[[ "$mode" != --submit ]] || command -v qsub >/dev/null
previous="$after"
while read -r id name; do
    (( id >= start && id <= stop )) || continue
    command_args=(qsub "${options[@]}" -N "rcg_b$id" -v "RCG_PBS_BUNDLE=$id")
    [[ -z "$previous" ]] || command_args+=(-W "depend=afterok:$previous")
    command_args+=(scripts/pbs/bundle.pbs)
    if [[ "$mode" == --preview ]]; then
        printf '%q ' "${command_args[@]}"; printf '# %s\n' "$name"
        previous="JOB_ID_$id"
    else
        submitted=$("${command_args[@]}") || { rc=$?; echo "Stopped at job $id; last accepted job: ${previous:-none}" >&2; exit "$rc"; }
        [[ "$submitted" =~ ^[0-9]+(\.[A-Za-z0-9_.-]+)?$ ]] || {
            echo "Unexpected qsub response: $submitted; inspect qstat before retrying" >&2; exit 2;
        }
        previous="$submitted"
        printf '%s bundle=%s %s\n' "$previous" "$id" "$name"
    fi
done <<< "$inventory"
if [[ "$mode" == --submit ]] && (( stop < total )); then
    echo "After this batch succeeds, continue with:" >&2
    printf 'bash scripts/pbs/submit_revision.sh --submit --start %s --afterok %q\n' "$((stop + 1))" "$previous" >&2
fi
