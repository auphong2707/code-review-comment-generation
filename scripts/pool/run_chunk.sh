#!/usr/bin/env bash
# One pool task (RCG_PBS_CHUNK = its inventory ID from chunks.sh), started by pool.sh with
# one visible GPU. It runs inside the pool's allocation and shares its deadline.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && "$1" == --body ) ]] || { echo "Usage: $0 [--body]" >&2; exit 2; }
export RCG_ROOT="${RCG_ROOT:?Set RCG_ROOT (pool.sh exports it)}"
cd "$RCG_ROOT"
[[ "${RCG_PBS_CHUNK:-}" =~ ^[1-9][0-9]*$ ]] || { echo "Set RCG_PBS_CHUNK to a listed chunk ID" >&2; exit 2; }
for key in RCG_PBS_STEP ONLY SKIP RUN_P0 RUN_P1 RUN_P2 SEEDS RCG_PROBES \
           RUN_GALORE RUN_14B_LORA ICL_14B A_7B_METHODS J_REQUIRE_PHASES RUN_REVISION RUN_LEGACY RUN_QWEN RUN_LLAMA \
           RCG_A_RUN_P0 RCG_A_RUN_P1 RCG_A_RUN_P2 RCG_A_SEEDS \
           RCG_B_RUN_P0 RCG_B_RUN_P1 RCG_B_RUN_P2 RCG_B_SEEDS \
           RCG_C_RUN_P0 RCG_C_RUN_P1 RCG_C_RUN_P2 RCG_C_SEEDS \
           RCG_D_RUN_P0 RCG_D_RUN_P1 RCG_D_RUN_P2 RCG_D_SEEDS; do
    [[ -z "${!key:-}" ]] || { echo "Unset $key for a default-scope chunk" >&2; exit 2; }
done
inventory=$(bash scripts/pool/chunks.sh)
row=$(awk -F '\t' -v id="$RCG_PBS_CHUNK" '$1 == id' <<< "$inventory")
[[ -n "$row" ]] || { echo "Unknown chunk $RCG_PBS_CHUNK" >&2; exit 2; }
IFS=$'\t' read -r id RCG_CHUNK_KIND RCG_CHUNK_PHASE RCG_CHUNK_ITEM <<< "$row"
export RCG_CHUNK_KIND RCG_CHUNK_PHASE RCG_CHUNK_ITEM
echo "Pool task $id: $RCG_CHUNK_KIND $RCG_CHUNK_PHASE $RCG_CHUNK_ITEM"
case "$RCG_CHUNK_KIND" in
    phase) phase="$RCG_CHUNK_PHASE" ;;
    step) export RCG_PBS_STEP="$RCG_CHUNK_ITEM"; phase="$RCG_CHUNK_PHASE" ;;
    *) phase=split_work ;;
esac
script="scripts/slurm/$phase.sbatch"
[[ "$phase" != split_work ]] || script=scripts/pool/split_work.sh
exec bash "$script"
