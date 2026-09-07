import pytest
from tools.a5000_preflight import check_report


def test_admission_rejects_stale_wrong_or_excess_memory():
    valid = {"runtime_digest": "current", "admitted": True, "hardware": "NVIDIA RTX A5000",
             "peak_reserved_gib": 21, "peak_allocated_gib": 20,
             "model": "qwen25", "method": "lora", "dataset": "trans_review"}
    expected = {"model": "qwen25", "method": "lora", "dataset": "trans_review"}
    assert check_report(valid, "current", expected) is valid
    for change in ({"runtime_digest": "old"}, {"admitted": False},
                   {"peak_reserved_gib": 23}, {"peak_reserved_gib": float("nan")},
                   {"hardware": "NVIDIA H100"}, {"peak_allocated_gib": 22},
                   {"method": "qlora"}):
        with pytest.raises(ValueError):
            check_report({**valid, **change}, "current", expected)


def test_peak_in_interrupted_process_is_retained(tmp_path):
    import json
    from tools.a5000_preflight import lifecycle_peaks
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "recovery_complete.json").write_text(json.dumps({
        "peak_vram_allocated_gb": 22, "peak_vram_reserved_gb": 23}))
    peaks = lifecycle_peaks({"peak_vram_allocated_gb": 19, "peak_vram_reserved_gb": 20}, tmp_path)
    assert peaks["peak_vram_reserved_gb"] == 23
    assert peaks["peak_vram_allocated_gb"] == 22
