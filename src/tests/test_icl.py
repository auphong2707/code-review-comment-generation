"""ICL v2 infrastructure tests (phase 04, D-023).

The golden-prompt test rebuilds legacy v1 prompts byte-for-byte from the raw
JSONLs and compares against a produced ICL artifact; it needs assets/ and the
Qwen tokenizer, so it only runs with RUN_SLOW=1 (the phase gate sets it).
"""

import json
import os
import random
from pathlib import Path

import pytest

from src.icl.naming import icl_result_filename
from src.icl.prompts import build_icl_prompt_causal, build_icl_prompt_seq2seq
from src.icl.selection import BM25Selection, demo_seed_select, legacy_select

ACRA_POOL = [
    {"patch": f"patch body {i}", "msg": f"review {i}"} for i in range(10)
]


def test_v2_fixes_the_test_turn_header():
    v1 = build_icl_prompt_causal(ACRA_POOL[:2], "test code", "acra_by_ls_pt", "v1")
    v2 = build_icl_prompt_causal(ACRA_POOL[:2], "test code", "acra_by_ls_pt", "v2")
    # demos always carry the dataset suffix
    assert v1[1]["content"].startswith("Code changes:")
    assert v2[1]["content"].startswith("Code changes:")
    # v1 keeps the legacy bug on the test turn (KI-014); v2 matches paper Fig. 4
    assert v1[-1]["content"].startswith("Code:\n")
    assert v2[-1]["content"].startswith("Code changes:\n")
    # trans_review has an empty suffix: both versions render identically
    tr_pool = [{"code_tokens": ["a", "b"], "docstring_tokens": ["ok"]}]
    assert (build_icl_prompt_causal(tr_pool, "x", "trans_review", "v1")
            == build_icl_prompt_causal(tr_pool, "x", "trans_review", "v2"))


def test_zero_shot_message_structure():
    messages = build_icl_prompt_causal([], "test code", "acra_by_ls_pt", "v2")
    assert [m["role"] for m in messages] == ["system", "user"]


def test_seq2seq_prompt_identical_across_versions():
    for version in ("v1", "v2"):
        prompt = build_icl_prompt_seq2seq(ACRA_POOL[:2], "test code", "acra_by_ls_pt", version)
        assert prompt.endswith("Code: test code\nReview:")
        assert prompt.count("Review:") == 3


def test_demo_seed_isolated_from_global_rng():
    random.seed(0)
    first = demo_seed_select(ACRA_POOL, 3, demo_seed=1, sample_index=5)
    random.seed(999)
    random.random()
    assert demo_seed_select(ACRA_POOL, 3, demo_seed=1, sample_index=5) == first
    assert demo_seed_select(ACRA_POOL, 3, demo_seed=1, sample_index=6) != first
    assert demo_seed_select(ACRA_POOL, 0, demo_seed=1, sample_index=5) == []


def test_legacy_select_unchanged():
    random.seed(42)
    expected = random.sample(ACRA_POOL, 3)
    assert legacy_select(ACRA_POOL, 3, "random", 42) == expected
    assert legacy_select(ACRA_POOL, 2, "first", 42) == ACRA_POOL[:2]
    assert legacy_select(ACRA_POOL, 0, "random", 42) == []
    assert legacy_select(ACRA_POOL, 99, "random", 42) == ACRA_POOL


def test_bm25_determinism_and_exclusion(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(row) for row in ACRA_POOL))
    sel = BM25Selection(ACRA_POOL, "acra_by_ls_pt", str(train_file), cache_dir=str(tmp_path))
    top = sel.select("patch body 3", 2)
    # the exact-match candidate is excluded; results are deterministic
    assert all(row["patch"] != "patch body 3" for row in top)
    assert sel.excluded_count == 1
    assert sel.select("patch body 3", 2) == top
    # cache round-trip yields the same ranking
    sel2 = BM25Selection(ACRA_POOL, "acra_by_ls_pt", str(train_file), cache_dir=str(tmp_path))
    assert sel2.select("patch body 3", 2) == top
    # --allow-exact-match keeps the duplicate as the top hit
    allowed = sel.select("patch body 3", 2, allow_exact_match=True)
    assert allowed[0]["patch"] == "patch body 3"


def test_filename_builder_table():
    cases = [
        (dict(), "qwen25_icl_3ex_random.json"),
        (dict(demo_seed=1), "qwen25_icl_3ex_random_s1_v1.json"),
        (dict(prompt_version="v2"), "qwen25_icl_3ex_random_v2.json"),
        (dict(demo_seed=2, prompt_version="v2"), "qwen25_icl_3ex_random_s2_v2.json"),
        (dict(non_default=True), "qwen25_icl_3ex_random_v1.json"),
    ]
    for kwargs, expected in cases:
        assert icl_result_filename("qwen25", 3, "random", **kwargs) == expected
    assert icl_result_filename("qwen25-0.5b", 5, "bm25") == "qwen25-0.5b_icl_5ex_bm25.json"


GOLDEN_ARTIFACT = Path("assets/models/review-code-generation-code-changes/qwen25_icl_3ex_random.json")


@pytest.mark.skipif(not os.getenv("RUN_SLOW"), reason="slow golden test; set RUN_SLOW=1")
@pytest.mark.skipif(not GOLDEN_ARTIFACT.is_file(), reason="assets/ not present")
def test_golden_v1_prompt_byte_identical():
    from transformers import AutoTokenizer

    from src.utils.helper import extract_input_target, load_jsonl

    artifact = json.loads(GOLDEN_ARTIFACT.read_text())
    train_data = load_jsonl("assets/data/Code_Review/msg-train.jsonl")
    test_data = load_jsonl("assets/data/Code_Review/msg-test.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct",
                                              trust_remote_code=True)
    seed = artifact["seed"]
    for i in range(3):
        examples = legacy_select(train_data, artifact["num_icl_examples"], "random", seed + i)
        test_code, _ = extract_input_target(test_data[i], "acra_by_ls_pt")
        messages = build_icl_prompt_causal(examples, test_code, "acra_by_ls_pt", "v1")
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert prompt == artifact["prompts"][i], f"prompt {i} diverged from the produced artifact"
