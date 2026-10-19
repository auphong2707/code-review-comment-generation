#!/usr/bin/env python3
"""Qualify conservative training settings with a persisted interruption pilot.

All paths are user-selected scratch. A passing report is evidence for this bounded
pilot, not a guarantee for arbitrary lengths/configs or different software revisions.
"""
import argparse
import json
import math
import importlib.metadata as metadata
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.recovery import (atomic_json, file_digest, fingerprint, install_stop_handlers,
                                   latest_recovery_checkpoint, stop_requested, INCOMPLETE)


def hardware_policy():
    """The A100 80 GB policy: GPU name, capacity window (GiB) and reserved-memory limit."""
    target = os.environ.get("RCG_TARGET_GPU", "a100_80gb")
    policies = {"a100_80gb": ("A100", 79, 81, 76)}
    if target not in policies:
        raise ValueError(f"Unsupported RCG_TARGET_GPU: {target}")
    return policies[target]


def environment_identity():
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets",
                 "evaluate", "bert-score", "galore-torch", "rank-bm25", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        cuda_build = file_digest(metadata.distribution("torch").locate_file("torch/version.py"))
    except (metadata.PackageNotFoundError, OSError):
        cuda_build = None
    try:
        driver = sorted(set(subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                          "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).splitlines()))
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return {"packages": packages, "python": sys.version, "cuda_build": cuda_build, "driver": driver,
            "target_gpu": os.environ.get("RCG_TARGET_GPU", "a100_80gb")}


def runtime_digest():
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py"))
    return fingerprint({"files": {str(p.relative_to(ROOT)): file_digest(p) for p in files},
                        "environment": environment_identity()})


def lifecycle_peaks(resources, run_dir):
    measurements = [resources]
    for path in Path(run_dir).glob("checkpoint-*/recovery_complete.json"):
        measurements.append(json.loads(path.read_text()))
    return {key: max(row[key] for row in measurements)
            for key in ("peak_vram_allocated_gb", "peak_vram_reserved_gb")}


def check_report(report, digest, expected=None):
    if report.get("runtime_digest") != digest or report.get("admitted") is not True:
        raise ValueError("Missing or stale admission; rerun the infrastructure smoke job")
    peak = report.get("peak_reserved_gib", float("inf"))
    allocated = report.get("peak_allocated_gib", float("inf"))
    name, minimum, maximum, limit = hardware_policy()
    capacity = report.get("total_memory_gib", 0)
    if (name not in report.get("hardware", "") or "MIG" in report.get("hardware", "").upper()
            or not minimum <= capacity <= maximum
            or not math.isfinite(peak) or not 0 < allocated <= peak <= limit):
        raise ValueError(f"Pilot did not meet the {name} {limit} GiB reserved-memory gate")
    if expected and any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Admission belongs to a different workload")
    return report


def require_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight requires exactly one allocated GPU")
    info = torch.cuda.get_device_properties(0)
    name, minimum, maximum, _ = hardware_policy()
    if (name not in info.name or "MIG" in info.name.upper()
            or not minimum <= info.total_memory / 2**30 <= maximum):
        raise RuntimeError(f"Expected {name} ({minimum}–{maximum} GiB); got {info.name}, "
                           f"{info.total_memory / 2**30:.2f} GiB")
    return info.name


def training_probe(args, digest, hardware):
    import torch
    from src.training.registry import get_spec, resolve_run
    # The generation batch is part of the pilot identity: a changed GEN_BS gets a fresh pilot.
    root = (Path(args.out_root) / "pilots" / digest[:12] / args.dataset / args.model / args.method
            / f"gen{args.gen_batch_size}")
    _, _, spec = get_spec(args.model, args.method)
    run_dir = Path(resolve_run(spec, learning_rate=5e-5, results_root=str(root))["experiment_results_dir"])
    result_path = run_dir / "complete_results.json"
    command = [sys.executable, "-m", "src.train", "--model", args.model, "--method", args.method,
               "--dataset", args.dataset, "--memory-profile", "a100", "--results-root", str(root),
               "--gen-batch-size", str(args.gen_batch_size),
               "--lifecycle-probe", "--learning_rate", "5e-5", "--recovery-steps", "1"]
    install_stop_handlers()
    if not result_path.exists():
        if latest_recovery_checkpoint(run_dir) is None:
            rc = subprocess.run(command + ["--probe-stop-after-step", "1"], cwd=ROOT).returncode
            if stop_requested():
                raise SystemExit(INCOMPLETE)
            if rc != INCOMPLETE:
                raise RuntimeError(f"Interruption pilot must exit 75 after its first save, got {rc}")
        rc = subprocess.run(command, cwd=ROOT).returncode
        if rc == INCOMPLETE or stop_requested():
            raise SystemExit(INCOMPLETE)
        if rc:
            raise RuntimeError(f"Lifecycle resume pilot failed with exit {rc}")
    result = json.loads(result_path.read_text())
    resources = result["resources"]
    if (not resources.get("lifecycle_probe") or resources.get("global_step") != 2
            or resources.get("resumed_from_step", 0) < 1
            or resources.get("trainable_params", 0) <= 0
            or result.get("expected_test_rows") != len(result.get("raw_data", []))):
        raise RuntimeError("Pilot did not establish training, resume and test completion")
    peaks = lifecycle_peaks(resources, run_dir)
    peak = peaks["peak_vram_reserved_gb"]
    report = {"runtime_digest": digest, "hardware": hardware, "model": args.model,
              "method": args.method, "dataset": args.dataset, "gen_batch_size": args.gen_batch_size,
              "peak_reserved_gib": peak,
              "peak_allocated_gib": peaks["peak_vram_allocated_gb"],
              "environment": environment_identity(),
              "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "admitted": peak <= hardware_policy()[3], "result": str(result_path),
              "note": "Two-step lifecycle and full-length generation pilot; target corpus campaign still required"}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True, choices=["trans_review", "acra_by_ls_pt"])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--gen-batch-size", type=int, default=1,
                        help="Campaign generation batch size the pilot must cover")
    parser.add_argument("--check", action="store_true", help="Read-only admission check, no GPU imports")
    args = parser.parse_args(argv)
    digest = runtime_digest()
    path = Path(args.out_root) / "admissions" / f"train_{args.model}_{args.method}_{args.dataset}.json"
    expected = {"model": args.model, "method": args.method, "dataset": args.dataset,
                "gen_batch_size": args.gen_batch_size}
    if args.check:
        check_report(json.loads(path.read_text()), digest, expected)
        print(f"Qualified: {args.model}/{args.method}/{args.dataset}")
        return
    hardware = require_gpu()
    report = training_probe(args, digest, hardware)
    atomic_json(path, report)
    check_report(report, digest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
