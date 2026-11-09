"""ICL example selection (phase 04): pure selection extracted from
icl_inference.py plus the revision additions — per-sample demo seeds that do
not touch the global RNG, and BM25 retrieval with exact-match exclusion
(D-023). Context-length optimization stays in icl_inference (needs the model).
"""

import hashlib
import os
import pickle
import random

from src.utils.helper import extract_input_target

BM25_CACHE_DIR = "assets/cache/icl"


def legacy_select(train_data, num_examples, selection_method, seed):
    """Byte-identical legacy selection: seeds the *global* RNG (v1 behaviour)."""
    if num_examples <= 0:
        return []
    if num_examples >= len(train_data):
        return train_data
    if selection_method == "random":
        random.seed(seed)
        return random.sample(train_data, num_examples)
    if selection_method == "first":
        return train_data[:num_examples]
    if selection_method == "diverse":
        step = len(train_data) // num_examples
        return [train_data[i * step] for i in range(num_examples)]
    raise ValueError(f"Unknown selection method: {selection_method}")


def demo_seed_select(train_data, num_examples, demo_seed, sample_index):
    """Repeatable per-sample random demos from an isolated RNG (D-023)."""
    if num_examples <= 0:
        return []
    if num_examples >= len(train_data):
        return train_data
    rng = random.Random(demo_seed * 1_000_003 + sample_index)
    return rng.sample(train_data, num_examples)


class BM25Selection:
    """BM25 retrieval over whitespace-tokenised train code strings.

    The index is built once per (dataset, train-file sha256) and pickled under
    assets/cache/icl/. Candidates whose code string equals the test code are
    skipped by default (exact-match exclusion; --allow-exact-match disables).
    """

    def __init__(self, train_data, dataset_name, train_file_path,
                 cache_dir=BM25_CACHE_DIR):
        from rank_bm25 import BM25Okapi

        self.train_data = train_data
        self.codes = [extract_input_target(row, dataset_name)[0] for row in train_data]
        digest = hashlib.sha256(open(train_file_path, "rb").read()).hexdigest()[:16]
        cache_path = os.path.join(cache_dir, f"bm25_{dataset_name}_{digest}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fh:
                self.bm25 = pickle.load(fh)
        else:
            self.bm25 = BM25Okapi([code.split() for code in self.codes])
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(self.bm25, fh)
        self.excluded_count = 0

    def select(self, test_code, num_examples, allow_exact_match=False):
        if num_examples <= 0:
            return []
        scores = self.bm25.get_scores(test_code.split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = []
        for i in ranked:
            if not allow_exact_match and self.codes[i] == test_code:
                self.excluded_count += 1
                continue
            selected.append(self.train_data[i])
            if len(selected) == num_examples:
                break
        return selected
