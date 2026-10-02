#!/usr/bin/env bash
# PBS adapter; the existing Bash campaign bodies remain the single run inventory.
set -euo pipefail
phase="${1:?Pass a campaign name}"
case "$phase" in
    00_07_infrastructure|12_campaign_a|13_campaign_b|14_campaign_c|15_campaign_d|16_campaign_j|split_work|bundle) ;;
    *) echo "Unknown campaign: $phase" >&2; exit 2 ;;
esac
export RCG_ROOT="${RCG_ROOT:-${PBS_O_WORKDIR:-$PWD}}"
cd "$RCG_ROOT"
script="scripts/slurm/$phase.sbatch"
[[ "$phase" != split_work ]] || script=scripts/pbs/split_work.sh
[[ "$phase" != bundle ]] || script=scripts/pbs/run_bundle.sh
[[ -f "$script" ]] || { echo "Set RCG_ROOT to the server repository root" >&2; exit 2; }
if [[ "${DRY:-0}" == 1 ]]; then
    exec bash "$script"
fi
[[ -n "${PBS_JOBID:-}" ]] || { echo "Run real jobs through PBS qsub on sutd_mega" >&2; exit 2; }
# PBS has no Slurm --signal=B:USR1@600 directive. Read the actual allocation,
# including qsub -l overrides, and send USR1 ten minutes before its deadline.
job_info=$(qstat -f "$PBS_JOBID")
walltime=$(awk '$1 == "Resource_List.walltime" && $2 == "=" {print $3}' <<< "$job_info")
[[ "$walltime" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]] || {
    echo "Cannot read PBS Resource_List.walltime for $PBS_JOBID" >&2; exit 2;
}
seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
(( seconds > 600 )) || { echo "PBS walltime must exceed the 600-second recovery margin" >&2; exit 2; }
# GNU timeout runs on the Linux compute node, never on the macOS bridge.
# Signal only the campaign shell: group-wide USR1 would also kill its tee logger.
# Incomplete work must not release an afterok dependency.
rc=0
timeout --foreground --signal=USR1 --kill-after=600s "$((seconds - 600))s" bash "$script" || rc=$?
case "$rc" in 124|137) exit 75 ;; *) exit "$rc" ;; esac
