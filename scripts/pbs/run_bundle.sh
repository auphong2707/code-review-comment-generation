#!/usr/bin/env bash
# One allocation timer surrounds the entire group, including all child tasks.
set -euo pipefail
[[ "${RCG_PBS_BUNDLE:-}" =~ ^[1-8]$ ]] || { echo "Set RCG_PBS_BUNDLE to 1..8" >&2; exit 2; }
[[ -z "${RCG_PBS_CHUNK:-}${RCG_PBS_STEP:-}" ]] || { echo "Unset chunk/step selection for a bundle" >&2; exit 2; }
RCG_PHASE="bundle_$RCG_PBS_BUNDLE"
source "$RCG_ROOT/scripts/slurm/common.sh"
# Target the child campaign shell only. It owns its worker process group; sending
# USR1 to the whole shell group would also kill its tee logger during recovery.
request_stop() {
    STOP_REQUESTED=1
    if [[ -n "$ACTIVE_PID" ]]; then kill -USR1 "$ACTIVE_PID" 2>/dev/null || true; fi
}
inventory=$(bash scripts/pbs/chunks.sh)
tasks=0
while IFS=$'\t' read -r id kind phase item; do
    case "$phase" in
        00_07_infrastructure) bundle=1 ;;
        12_campaign_a|A) bundle=2 ;;
        13_campaign_b|B) bundle=3 ;;
        14_campaign_c|C) bundle=4 ;;
        15_campaign_d|D) bundle=5 ;;
        J|qwen-revision) bundle=6 ;;
        llama-legacy) bundle=7 ;;
        llama-revision) bundle=8 ;;
        *) echo "Unmapped chunk phase: $phase" >&2; exit 2 ;;
    esac
    [[ "$bundle" == "$RCG_PBS_BUNDLE" ]] || continue
    tasks=$((tasks + 1))
    if [[ "$DRY" == 1 ]]; then
        RCG_PBS_CHUNK="$id" bash scripts/pbs/run_chunk.sh --body
    else
        # run_step forwards stop signals and waits for the child campaign, which
        # forwards them in turn to its active Python worker before committing.
        run_step "chunk_$id" env RCG_PBS_CHUNK="$id" bash scripts/pbs/run_chunk.sh --body
    fi
done <<< "$inventory"
(( tasks > 0 )) || { echo "Empty PBS bundle" >&2; exit 2; }
finish_report
