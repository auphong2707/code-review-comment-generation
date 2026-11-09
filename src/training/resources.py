"""Assemble the additive `resources` block for complete_results.json
(phase 03; R1.6/R1.7). Torch-free: callers pass the torch-derived values.
"""

import os
import subprocess


def artifact_size_bytes(results_dir):
    """Size of the saved weights (adapter for PEFT, full model for FMFT)."""
    total = 0
    for name in os.listdir(results_dir):
        if name.endswith((".safetensors", ".bin")):
            total += os.path.getsize(os.path.join(results_dir, name))
    return total


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def collect_resources(train_output, time_summary, *, test_generation_time_s,
                      trainable_params, total_params, hardware, experiment_name,
                      dataset, seed, protocol, protocol_overrides,
                      eval_max_samples, gen_batch_size, results_dir):
    metrics = getattr(train_output, "metrics", None) or {}
    return {
        "train_runtime_s": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_step_time_s": time_summary.get("train_step_time_s", []),
        "eval_time_s": time_summary.get("eval_time_s", []),
        "test_generation_time_s": test_generation_time_s,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100.0 * trainable_params / total_params if total_params else None,
        "peak_vram_allocated_gb": time_summary.get("peak_vram_allocated_gb"),
        "peak_vram_reserved_gb": time_summary.get("peak_vram_reserved_gb"),
        "adapter_size_bytes": artifact_size_bytes(results_dir),
        "hardware": hardware,
        "seed": seed,
        "protocol": protocol,
        "protocol_overrides": protocol_overrides or {},
        "eval_max_samples": eval_max_samples,
        "gen_batch_size": gen_batch_size,
        "git_commit": git_commit(),
        "experiment_name": experiment_name,
        "dataset": dataset,
    }
