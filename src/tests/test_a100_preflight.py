import pytest
from tools.a100_preflight import check_report


def test_admission_rejects_stale_wrong_or_excess_memory(monkeypatch):
    monkeypatch.setenv("RCG_TARGET_GPU", "a100_80gb")
    valid = {"runtime_digest": "current", "admitted": True, "hardware": "NVIDIA A100-SXM4-80GB",
             "total_memory_gib": 79.15, "peak_reserved_gib": 60, "peak_allocated_gib": 59,
             "model": "qwen25", "method": "lora", "dataset": "trans_review", "gen_batch_size": 16}
    expected = {"model": "qwen25", "method": "lora", "dataset": "trans_review", "gen_batch_size": 16}
    assert check_report(valid, "current", expected) is valid
    for change in ({"runtime_digest": "old"}, {"admitted": False},
                   {"peak_reserved_gib": 77}, {"peak_reserved_gib": float("nan")},
                   {"hardware": "NVIDIA H100"}, {"peak_allocated_gib": 61},
                   {"method": "qlora"}, {"gen_batch_size": 8}):
        with pytest.raises(ValueError):
            check_report({**valid, **change}, "current", expected)


def test_peak_in_interrupted_process_is_retained(tmp_path):
    import json
    from tools.a100_preflight import lifecycle_peaks
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "recovery_complete.json").write_text(json.dumps({
        "peak_vram_allocated_gb": 22, "peak_vram_reserved_gb": 23}))
    peaks = lifecycle_peaks({"peak_vram_allocated_gb": 19, "peak_vram_reserved_gb": 20}, tmp_path)
    assert peaks["peak_vram_reserved_gb"] == 23
    assert peaks["peak_vram_allocated_gb"] == 22


def test_a100_admission_requires_full_capacity_and_matching_target(monkeypatch):
    monkeypatch.setenv("RCG_TARGET_GPU", "a100_80gb")
    valid = {"runtime_digest": "current", "admitted": True,
             "hardware": "NVIDIA A100-SXM4-80GB", "total_memory_gib": 79.15,
             "peak_reserved_gib": 60, "peak_allocated_gib": 59}
    assert check_report(valid, "current") is valid
    for change in ({"total_memory_gib": 40}, {"total_memory_gib": 20},
                   {"hardware": "NVIDIA RTX 3090"}, {"hardware": "NVIDIA H100 80GB"},
                   {"hardware": "NVIDIA A100-SXM4-80GB MIG 7g.80gb"},
                   {"peak_reserved_gib": 77}, {"peak_reserved_gib": float("nan")},
                   {"peak_allocated_gib": 61}):
        with pytest.raises(ValueError):
            check_report({**valid, **change}, "current")
    without_capacity = {key: value for key, value in valid.items() if key != "total_memory_gib"}
    with pytest.raises(ValueError):
        check_report(without_capacity, "current")
    monkeypatch.setenv("RCG_TARGET_GPU", "unsupported")
    with pytest.raises(ValueError):
        check_report(valid, "current")


@pytest.mark.parametrize("name,capacity,count,accepted", [
    ("NVIDIA A100-SXM4-80GB", 79.15, 1, True),
    ("NVIDIA A100-PCIE-40GB", 40, 1, False),
    ("NVIDIA A100-SXM4-80GB MIG", 20, 1, False),
    ("NVIDIA A100-SXM4-80GB MIG 7g.80gb", 79.15, 1, False),
    ("NVIDIA RTX 3090", 24, 1, False),
    ("NVIDIA A100-SXM4-80GB", 79.15, 2, False),
])
def test_allocated_gpu_policy_without_cuda(monkeypatch, name, capacity, count, accepted):
    import sys
    from types import SimpleNamespace
    from tools.a100_preflight import require_gpu
    monkeypatch.setenv("RCG_TARGET_GPU", "a100_80gb")
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: count,
                           get_device_properties=lambda _: SimpleNamespace(
                               name=name, total_memory=capacity * 2**30))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    if accepted:
        assert require_gpu() == name
    else:
        with pytest.raises(RuntimeError):
            require_gpu()
