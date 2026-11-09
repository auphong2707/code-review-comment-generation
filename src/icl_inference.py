#!/usr/bin/env python3
# ------------------------------------------------------------------
# In-Context Learning (ICL) Inference Script
# ------------------------------------------------------------------
import sys
import os
from pathlib import Path
import argparse
import json
import random
from typing import List, Dict, Any

import huggingface_hub

# Add project root to Python path (keeps `python src/icl_inference.py` working)
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helper import set_seed
set_seed()

from src.utils.constants import *
from src.utils.helper import load_jsonl, extract_input_target
from src.evaluation.metrics import EMPTY_PLACEHOLDER, as_icl_results, compute_generation_metrics

import torch
import warnings
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


from src.llm_as_judge.integration import (
    add_judge_arguments,
    append_judge_results_to_icl,
    build_judge_config,
    save_json_result,
)
from src.icl import prompts as icl_prompts
from src.icl.naming import icl_result_filename
from src.icl.selection import BM25Selection, demo_seed_select, legacy_select


def get_max_context_length(model) -> int:
    """Get the maximum context length from the model's configuration.

    Args:
        model: The loaded model

    Returns:
        Maximum context length for the model
    """
    if hasattr(model.config, 'max_position_embeddings'):
        return model.config.max_position_embeddings
    elif hasattr(model.config, 'n_positions'):
        return model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        return model.config.max_sequence_length
    else:
        # Fallback to default if model doesn't specify
        print(f"Warning: Could not determine max context length for model. Using default: {INPUT_MAX_LENGTH}")
        return INPUT_MAX_LENGTH


def check_context_length_warning(text: str, tokenizer, model, context_type: str = "input") -> bool:
    """Check if text exceeds model's context length and issue warning.

    Args:
        text: Text to check
        tokenizer: Tokenizer to use for encoding
        model: The loaded model
        context_type: Type of context ("input", "prompt")

    Returns:
        True if context was truncated, False otherwise
    """
    max_length = get_max_context_length(model)
    tokens = tokenizer.encode(text, add_special_tokens=True)

    if len(tokens) > max_length:
        warnings.warn(
            f"Context length ({len(tokens)} tokens) exceeds model "
            f"maximum ({max_length} tokens) for {context_type}. Content will be truncated.",
            UserWarning
        )
        return True
    return False


def load_base_model_and_tokenizer(model_name: str, attn_implementation: str = None):
    """Load the base model and tokenizer for ICL inference.

    Args:
        model_name: Name of the model ("codet5", "plbart", "qwen15", "qwen25", "qwen25-0.5b")
        attn_implementation: override for the causal models (default: legacy
            flash_attention_2 on CUDA; pass "sdpa" on machines without flash-attn)

    Returns:
        tuple: (model, tokenizer)
    """
    model_mapping = {
        "codet5": (MODEL_CT5, TOKENIZER_CT5),
        "plbart": (MODEL_PLBART, TOKENIZER_PLBART),
        "qwen15": (MODEL_QWEN15, TOKENIZER_QWEN15),
        "qwen25": (MODEL_QWEN25, TOKENIZER_QWEN25),
        "qwen25-0.5b": (MODEL_QWEN25_0_5B, TOKENIZER_QWEN25_0_5B),
        "qwen25-14b": (MODEL_QWEN25_14B, TOKENIZER_QWEN25_14B),
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_mapping.keys())}")

    model_path, tokenizer_path = model_mapping[model_name]

    print(f"Loading model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load model based on type
    if model_name in ["codet5", "plbart"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_name in ["qwen15", "qwen25", "qwen25-0.5b", "qwen25-14b"]:
        attn = attn_implementation or ("flash_attention_2" if torch.cuda.is_available() else None)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn
        )
        # Configure tokenizer for left-padding for causal models
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Set generation config
    if hasattr(model, 'generation_config'):
        model.generation_config.num_beams = NUM_BEAMS
        model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH

    return model, tokenizer


def select_icl_examples(train_data: List[Dict], num_examples: int, selection_method: str = "random",
                       current_sample: Dict = None, seed: int = SEED, model_name: str = None,
                       tokenizer=None, model=None) -> List[Dict]:
    """Select examples for in-context learning.

    Args:
        train_data: List of training examples
        num_examples: Number of examples to select
        selection_method: Method for selecting examples ("random", "first", "diverse")
        current_sample: Current test sample (for future similarity-based selection)
        seed: Random seed for reproducibility
        model_name: Name of the model (for prompt building)
        tokenizer: Tokenizer (for context length checking)
        model: The loaded model (for context length checking)

    Returns:
        List of selected examples
    """
    selected = legacy_select(train_data, num_examples, selection_method, seed)

    # If model and tokenizer provided, check if we can fit more examples
    if model and tokenizer and current_sample:
        selected = optimize_icl_examples_for_context(selected, current_sample, model_name, tokenizer, model)

    return selected


def optimize_icl_examples_for_context(selected_examples: List[Dict], current_sample: Dict,
                                     model_name: str, tokenizer, model,
                                     dataset_name: str = None,
                                     prompt_version: str = "v1") -> List[Dict]:
    """Greedy legacy fit: keep prefix demos while the rendered prompt fits the
    model context (causal models reserve OUTPUT_MAX_LENGTH for the output)."""
    dataset_name = dataset_name or DATASET_NAME
    test_code, _ = extract_input_target(current_sample, dataset_name)
    max_context = get_max_context_length(model)
    seq2seq = model_name in ["codet5", "plbart"]
    available_context = max_context if seq2seq else max_context - OUTPUT_MAX_LENGTH

    def prompt_tokens(examples):
        if seq2seq:
            prompt = icl_prompts.build_icl_prompt_seq2seq(examples, test_code, dataset_name, prompt_version)
        else:
            messages = icl_prompts.build_icl_prompt_causal(examples, test_code, dataset_name, prompt_version)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(tokenizer.encode(prompt, add_special_tokens=True))

    if prompt_tokens(selected_examples) <= available_context:
        return selected_examples

    optimized_examples = []
    for example in selected_examples:
        if prompt_tokens(optimized_examples + [example]) <= available_context:
            optimized_examples.append(example)
        else:
            break

    if len(optimized_examples) < len(selected_examples):
        warnings.warn(
            f"Reduced ICL examples from {len(selected_examples)} to {len(optimized_examples)} "
            f"to fit within model context limit ({available_context} tokens available for input)",
            UserWarning
        )

    return optimized_examples


def build_icl_prompt_seq2seq(examples: List[Dict], test_code: str) -> str:
    """Thin legacy wrapper over src.icl.prompts (module-global dataset, v1)."""
    return icl_prompts.build_icl_prompt_seq2seq(examples, test_code, DATASET_NAME)


def build_icl_prompt_causal(examples: List[Dict], test_code: str, tokenizer) -> List[Dict]:
    """Thin legacy wrapper over src.icl.prompts (module-global dataset, v1)."""
    return icl_prompts.build_icl_prompt_causal(examples, test_code, DATASET_NAME)


def generate_review_seq2seq(model, tokenizer, prompt: str, max_new_tokens: int = OUTPUT_MAX_LENGTH) -> str:
    """Generate review comment using seq2seq model.

    Args:
        model: Seq2seq model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated review comment
    """
    # Check context length and warn if truncated
    max_input_length = get_max_context_length(model)
    check_context_length_warning(prompt, tokenizer, model, "seq2seq prompt")

    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=OUTPUT_MIN_LENGTH,
            num_beams=NUM_BEAMS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text.strip()


def generate_review_causal(model, tokenizer, messages: List[Dict], max_new_tokens: int = OUTPUT_MAX_LENGTH) -> str:
    """Generate review comment using causal LM model.

    Args:
        model: Causal LM model
        tokenizer: Tokenizer
        messages: Messages for chat template
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated review comment
    """
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Check context length and warn if truncated
    max_input_length = get_max_context_length(model) - max_new_tokens  # Reserve space for output
    check_context_length_warning(prompt, tokenizer, model, "causal LM prompt")

    inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=OUTPUT_MIN_LENGTH,
                num_beams=NUM_BEAMS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Extract only the generated part
    generated_ids = generated_ids[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text.strip()


def generate_batched(model, tokenizer, prompts_chunk: List[str], seq2seq: bool,
                     max_prompt_tokens=None) -> List[str]:
    """Batched generation with the same beam settings as the per-sample paths.

    Causal batches are left-padded (tokenizer.padding_side="left"); note that
    bf16 batching may change rare beam ties, so one batch size per model is
    pinned in the campaign manifest (D-028c).
    """
    max_input_length = get_max_context_length(model) - (0 if seq2seq else OUTPUT_MAX_LENGTH)
    if max_prompt_tokens is not None:
        max_input_length = min(max_input_length, max_prompt_tokens)
    inputs = tokenizer(prompts_chunk, max_length=max_input_length, truncation=True,
                       padding=True, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=OUTPUT_MAX_LENGTH,
        min_new_tokens=OUTPUT_MIN_LENGTH,
        num_beams=NUM_BEAMS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        if seq2seq:
            generated = model.generate(
                decoder_start_token_id=tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") else None,
                **gen_kwargs)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**gen_kwargs)
            generated = generated[:, inputs["input_ids"].shape[1]:]
    texts = []
    for row in generated:
        text = tokenizer.decode(row, skip_special_tokens=True).strip()
        texts.append(text if text else EMPTY_PLACEHOLDER)
    return texts


# Dataset keys used by subset files (the judge driver's names) for each ICL dataset.
SUBSET_DATASET_KEYS = {"trans_review": "trans_review", "acra_by_ls_pt": "code_reviewer"}


def load_sample_indices(path, dataset_name, row_count):
    """Sorted original test-row indices from a {"dataset", "indices"} subset file."""
    with open(path, encoding="utf-8") as stream:
        subset = json.load(stream)
    expected = SUBSET_DATASET_KEYS.get(dataset_name, dataset_name)
    if subset.get("dataset") != expected:
        raise ValueError(f"{path} is a subset of {subset.get('dataset')!r}, not {expected!r}")
    indices = sorted(int(index) for index in subset["indices"])
    if not indices or len(set(indices)) != len(indices) or indices[0] < 0 or indices[-1] >= row_count:
        raise ValueError(f"{path}: indices must be unique and within the {row_count}-row test set")
    return indices


def run_icl_inference(model_name: str, num_examples: int, selection_method: str = "random",
                     test_data_path: str = TEST_DATA_DIR, train_data_path: str = TRAIN_DATA_DIR,
                     output_path: str = None, max_samples: int = None, seed: int = SEED,
                     max_context_utilization: bool = False, verbose: bool = False,
                     dataset_name: str = None, prompt_version: str = "v1",
                     demo_seed: int = None, batch_size: int = 1,
                     attn_implementation: str = None, max_prompt_tokens: int = None,
                     allow_exact_match: bool = False, memory_profile=None,
                     run_fingerprint=None, sample_index_file: str = None) -> Dict[str, Any]:
    """Run ICL inference on the test dataset.

    Every new argument defaults to the legacy behaviour: prompt v1, global-RNG
    demo selection seeded seed+i, greedy context fit, per-sample generation.
    `sample_index_file` restricts the test set to fixed original row indices (the
    judge's {"dataset", "indices"} subset file); demos are then seeded by the
    original index, so a row draws the same demos as in a full-set run.
    """

    from src.training.recovery import PredictionJournal, stop_if_requested, defer_if_short, metric_seconds
    from src.evaluation.metrics import configure_metric_resources
    from src.training.memory_profile import metric_resources
    metric_policy = metric_resources(memory_profile) or {}
    configure_metric_resources(**metric_policy)

    # Configure warnings based on max_context_utilization flag
    if not max_context_utilization:
        warnings.filterwarnings("ignore", category=UserWarning, module=__name__)

    print(f"Context utilization mode: {'Aggressive' if max_context_utilization else 'Standard'}")
    dataset_name = dataset_name or DATASET_NAME

    # Load model and tokenizer
    print(f"Loading {model_name} model...")
    model, tokenizer = load_base_model_and_tokenizer(model_name, attn_implementation)
    if memory_profile and any(device in ("cpu", "disk")
                              for device in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("A100 ICL cannot silently offload the subject model")
    if prompt_version == "v2":
        # an overflowing prompt keeps the test turn, not the oldest demos
        tokenizer.truncation_side = "left"

    print(f"Max context length for {model_name}: {get_max_context_length(model)} tokens")

    # Load datasets
    print("Loading datasets...")
    train_data = load_jsonl(train_data_path)
    test_data = load_jsonl(test_data_path)

    sample_indices = None
    if sample_index_file:
        sample_indices = load_sample_indices(sample_index_file, dataset_name, len(test_data))
        test_data = [test_data[index] for index in sample_indices]
        print(f"Processing {len(test_data)} test samples from {sample_index_file}")
    if max_samples:
        test_data = test_data[:max_samples]
        if sample_indices is not None:
            sample_indices = sample_indices[:max_samples]
        print(f"Processing {len(test_data)} test samples")

    seq2seq = model_name in ["codet5", "plbart"]
    bm25 = (BM25Selection(train_data, dataset_name, train_data_path)
            if selection_method == "bm25" else None)

    references = []
    prompts = []
    messages_per_sample = []
    demo_counts = []
    prompt_token_counts = []
    context_truncation_count = 0
    example_reduction_count = 0

    def render(examples, test_code):
        if seq2seq:
            return icl_prompts.build_icl_prompt_seq2seq(
                examples, test_code, dataset_name, prompt_version), None
        messages = icl_prompts.build_icl_prompt_causal(
            examples, test_code, dataset_name, prompt_version)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, messages

    # Pass A: select demos and render every prompt
    for i, test_sample in enumerate(tqdm(test_data, desc="Building prompts")):
        try:
            test_code, reference = extract_input_target(test_sample, dataset_name)
            row_index = i if sample_indices is None else sample_indices[i]

            if bm25 is not None:
                icl_examples = bm25.select(test_code, num_examples, allow_exact_match)
            elif demo_seed is not None:
                icl_examples = demo_seed_select(train_data, num_examples, demo_seed, row_index)
            else:
                icl_examples = legacy_select(train_data, num_examples, selection_method, seed + row_index)

            if max_prompt_tokens is None:
                # legacy greedy fit against the model context
                icl_examples = optimize_icl_examples_for_context(
                    icl_examples, test_sample, model_name, tokenizer, model,
                    dataset_name, prompt_version)
                prompt, messages = render(icl_examples, test_code)
            else:
                # explicit budget: drop demos from the end until the prompt fits
                prompt, messages = render(icl_examples, test_code)
                while icl_examples and len(tokenizer.encode(prompt, add_special_tokens=True)) > max_prompt_tokens:
                    icl_examples = icl_examples[:-1]
                    prompt, messages = render(icl_examples, test_code)

            if len(icl_examples) < num_examples:
                example_reduction_count += 1
            if check_context_length_warning(prompt, tokenizer, model, f"sample {i+1} prompt"):
                context_truncation_count += 1

            prompts.append(prompt)
            messages_per_sample.append(messages)
            references.append(reference)
            demo_counts.append(len(icl_examples))
            prompt_token_counts.append(len(tokenizer.encode(prompt, add_special_tokens=True)))

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            prompts.append("<ERROR>")
            messages_per_sample.append(None)
            try:
                _, error_reference = extract_input_target(test_sample, dataset_name)
            except Exception:
                error_reference = ""
            references.append(error_reference)
            demo_counts.append(0)
            prompt_token_counts.append(0)

    journal = (PredictionJournal(str(output_path) + ".progress",
               {"run": run_fingerprint, "model_revision": getattr(model.config, "_commit_hash", None),
                "prompts": prompts, "references": references}, len(prompts)) if memory_profile else None)

    # Pass B: generate
    print(f"Running ICL inference with {num_examples} examples...")
    predictions = ["<ERROR>"] * len(prompts)
    valid = [i for i, p in enumerate(prompts) if p != "<ERROR>"]

    if journal:
        if len(valid) != len(prompts):
            raise RuntimeError("A100 ICL prompt construction failed; refusing complete output")
        for start in tqdm(range(0, len(valid), batch_size),
                          desc=f"Generating reviews (resumable, batch {batch_size})"):
            stop_if_requested()
            idx = valid[start:start + batch_size]
            saved = [journal.get(i) for i in idx]
            if any(row is None for row in saved):
                # The whole batch is regenerated so its composition matches a fresh run;
                # only missing rows are written. No catch-all: an OOM/error must fail,
                # preserving earlier rows.
                texts = generate_batched(model, tokenizer, [prompts[i] for i in idx], seq2seq,
                                         max_prompt_tokens=max_prompt_tokens)
                for i, row, text in zip(idx, saved, texts):
                    if row is None:
                        journal.put(i, {"prediction": text, "reference": references[i]})
                saved = [journal.get(i) for i in idx]
            for i, row in zip(idx, saved):
                predictions[i] = row["prediction"]
        journal.values()
        stop_if_requested()
        defer_if_short(metric_seconds(len(predictions), metric_policy.get("device")), "ICL corpus metrics")
    elif batch_size <= 1:
        for i in tqdm(valid, desc="Generating reviews"):
            try:
                if seq2seq:
                    generated_review = generate_review_seq2seq(model, tokenizer, prompts[i])
                else:
                    generated_review = generate_review_causal(model, tokenizer, messages_per_sample[i])
                if not generated_review.strip():
                    print(f"WARNING: Empty generation at sample {i}")
                    generated_review = EMPTY_PLACEHOLDER
                predictions[i] = generated_review
                if verbose:
                    print(f"\nExample {i+1}:")
                    print(f"  ICL Examples Used: {demo_counts[i]}/{num_examples}")
                    print(f"  Prompt: {prompts[i][:200]}...")
                    print(f"  Generated: {generated_review}")
                    print(f"  Reference: {references[i]}")
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
    else:
        for start in tqdm(range(0, len(valid), batch_size),
                          desc=f"Generating reviews (batched x{batch_size})"):
            idx = valid[start:start + batch_size]
            for i, text in zip(idx, generate_batched(model, tokenizer,
                                                     [prompts[i] for i in idx], seq2seq,
                                                     max_prompt_tokens=max_prompt_tokens)):
                predictions[i] = text

    # Compute the frozen metric suite (single shared implementation)
    print("Computing evaluation metrics...")
    metric_values = compute_generation_metrics(predictions, references)

    # Prepare results (legacy ICL schema + additive revision keys)
    results = {
        "model_name": model_name,
        "num_icl_examples": num_examples,
        "selection_method": selection_method,
        "num_test_samples": len(test_data),
        **as_icl_results(metric_values),
        "predictions": predictions,
        "references": references,
        "prompts": prompts,
        "seed": seed,
        "max_context_length": get_max_context_length(model),
        "context_truncation_count": context_truncation_count,
        "example_reduction_count": example_reduction_count,
        "max_context_utilization": max_context_utilization,
        "prompt_version": prompt_version,
        "demo_seed": demo_seed,
        "retrieval_method": selection_method,
        "exact_match_excluded": bm25.excluded_count if bm25 else 0,
        "demo_count_effective": sum(demo_counts) / len(demo_counts) if demo_counts else 0.0,
        "prompt_tokens_mean": sum(prompt_token_counts) / len(prompt_token_counts) if prompt_token_counts else 0.0,
        "prompt_tokens_max": max(prompt_token_counts, default=0),
        "batch_size": batch_size,
        "attn_implementation": attn_implementation,
        "max_prompt_tokens": max_prompt_tokens,
    }
    if sample_indices is not None:
        # Original test-set row of each prediction (the judge pairs inputs by it).
        results.update(sample_index_file=sample_index_file, sample_indices=sample_indices)

    if memory_profile:
        results.update(
            memory_profile=memory_profile, run_fingerprint=run_fingerprint,
            model_revision=getattr(model.config, "_commit_hash", None),
            metric_device=metric_policy.get("device"), metric_batch_size=metric_policy.get("batch_size"),
            prompt_truncated_count=sum(n > max_prompt_tokens for n in prompt_token_counts),
            effective_prompt_tokens_max=min(max(prompt_token_counts, default=0), max_prompt_tokens),
        )

    print(f"\nResults:")
    print(f"  Model: {model_name}")
    print(f"  ICL Examples: {num_examples}")
    print(f"  Selection Method: {selection_method}")
    print(f"  Test Samples: {len(test_data)}")
    print(f"  Max Context Length: {get_max_context_length(model)}")
    print(f"  Context Truncations: {context_truncation_count}/{len(test_data)}")
    print(f"  Example Reductions: {example_reduction_count}/{len(test_data)}")
    print(f"  BLEU-4: {metric_values['bleu_4']:.4f}")
    print(f"  Exact Match: {metric_values['exact_match']:.4f}")

    return results


def main():
    # The prompt builders and example selectors read the module-level
    # DATASET_NAME (star-imported default); main() rebinds it from the CLI.
    global DATASET_NAME

    parser = argparse.ArgumentParser(description="Run In-Context Learning (ICL) inference for code review generation")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["codet5", "plbart", "qwen15", "qwen25", "qwen25-0.5b", "qwen25-14b"],
                       help="Name of the model to use")

    # ICL arguments
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of ICL examples to use (default: 3)")
    parser.add_argument("--selection_method", type=str, default="random",
                       choices=["random", "first", "diverse", "bm25"],
                       help="Method for selecting ICL examples (default: random)")
    parser.add_argument("--max_context_utilization", action="store_true",
                       help="Enable aggressive context length maximization with warnings")
    parser.add_argument("--prompt-version", dest="prompt_version", default="v1",
                       choices=["v1", "v2"],
                       help="v1 = legacy (CodeReviewer test turn says 'Code:', KI-014); "
                            "v2 = test turn uses the dataset prompt_suffix (paper Fig. 4)")
    parser.add_argument("--demo-seed", dest="demo_seed", type=int, default=None,
                       help="Per-sample demo seed via an isolated RNG (default: legacy "
                            "global random.seed(seed+i))")
    parser.add_argument("--allow-exact-match", dest="allow_exact_match", action="store_true",
                       help="bm25 only: keep candidates whose code equals the test code "
                            "(excluded by default; excluded count is reported)")
    parser.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None,
                       help="Explicit prompt budget: drop demos from the end until the prompt "
                            "fits (default: legacy greedy fit against the model context)")
    parser.add_argument("--memory-profile", choices=["a100"], default=None,
                        help="A100 80 GB: batched SDPA generation, bounded context, durable row recovery")
    parser.add_argument("--sample-index-file", dest="sample_index_file", default=None,
                        help="Restrict the test set to the original row indices of a "
                             "{dataset, indices} subset file (e.g. paper/generated/cr_judge_subset.json)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
                       help="Batched left-padded generation (1 = legacy per-sample; bf16 "
                            "batching may change rare beam ties - pin one size per model)")
    parser.add_argument("--attn-implementation", dest="attn_implementation", default=None,
                       help="Override attn implementation (default: legacy flash_attention_2 "
                            "on CUDA; use sdpa locally)")

    # Data arguments
    parser.add_argument("--test_data_path", type=str, default=None,
                       help="Path to test dataset JSONL file (default: from the --dataset registry)")
    parser.add_argument("--train_data_path", type=str, default=None,
                       help="Path to training dataset JSONL file (default: from the --dataset registry)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of test samples to process")

    # Output arguments
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument("--results_dir", type=str, default="assets/results/icl/",
                       help="Directory to save results (used if output_path not specified)")
    parser.add_argument("--skip-if-complete", dest="skip_if_complete", action="store_true",
                       help="Exit 0 immediately when the result JSON already exists "
                            "(idempotent campaign re-submission)")

    # General arguments
    parser.add_argument("--seed", type=int, default=SEED,
                       help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                       choices=sorted(DATASET_CONFIGS),
                       help="Dataset the test/train files belong to (default: constants.DATASET_NAME)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print per-sample prompts and generations")
    parser.add_argument("--upload", action="store_true",
                       help="Upload the results directory to the Hugging Face Hub after the run")
    add_judge_arguments(parser)

    args = parser.parse_args()
    judge_config = build_judge_config(args) if args.llm_judge else None

    DATASET_NAME = args.dataset

    # Resolve default data paths from the dataset registry (identical to the
    # legacy TRAIN/TEST_DATA_DIR constants for the default dataset)
    dataset_cfg = get_dataset_config(args.dataset)
    if args.test_data_path is None:
        args.test_data_path = str(dataset_cfg["data_files"]["test"])
    if args.train_data_path is None:
        args.train_data_path = str(dataset_cfg["data_files"]["train"])

    if args.sample_index_file and judge_config is not None:
        raise SystemExit("--sample-index-file outputs are judged by the separate judge phase")
    if args.memory_profile:
        if args.output_path is None and args.results_dir == "assets/results/icl/":
            raise SystemExit("A100 ICL requires an explicit isolated --results_dir or --output_path")
        if args.batch_size < 1 or args.model_name not in ("qwen25", "qwen15", "qwen25-0.5b"):
            raise SystemExit("A100 ICL supports a positive batch on the three original causal models")
        if args.attn_implementation not in (None, "sdpa"):
            raise SystemExit("A100 ICL requires SDPA")
        if judge_config is not None:
            raise SystemExit("Run the separate judge phase after A100 inference")
        args.attn_implementation = "sdpa"
        args.prompt_version = "v2"
        args.max_prompt_tokens = 2048 if args.max_prompt_tokens is None else args.max_prompt_tokens
        if not 1 <= args.max_prompt_tokens <= 2048:
            raise SystemExit("A100 ICL prompt budget must be in [1, 2048]")

    # Generate output path if not provided (legacy name when every new flag is
    # at its default, so revision outputs can never overwrite legacy artifacts)
    if args.output_path is None:
        filename = icl_result_filename(
            args.model_name, args.num_examples, args.selection_method,
            demo_seed=args.demo_seed, prompt_version=args.prompt_version,
            non_default=(args.max_prompt_tokens is not None or args.allow_exact_match
                         or args.batch_size != 1 or args.attn_implementation is not None
                         or args.sample_index_file is not None))
        args.output_path = os.path.join(args.results_dir, filename)

    from contextlib import nullcontext
    from src.training.recovery import (RunLock, bind_identity, file_digest, atomic_json,
                                       install_stop_handlers, stop_if_requested)
    run_fingerprint = None
    with (RunLock(args.output_path + ".lock") if args.memory_profile else nullcontext()):
        if args.memory_profile:
            if Path(args.output_path).exists() and not Path(args.output_path + ".identity.json").exists():
                raise SystemExit("Refusing to overwrite a result not bound to this A100 profile")
            identity = {key: value for key, value in vars(args).items()
                        if key not in ("skip_if_complete", "verbose", "upload")}
            identity["train_data_sha256"] = file_digest(args.train_data_path)
            identity["test_data_sha256"] = file_digest(args.test_data_path)
            if args.sample_index_file:
                identity["sample_index_sha256"] = file_digest(args.sample_index_file)
            run_fingerprint = bind_identity(args.output_path + ".identity.json", identity)
            install_stop_handlers()
        if args.skip_if_complete:
            targets = [args.output_path]
            if args.judge_output_path:
                targets.append(args.judge_output_path)
            if all(os.path.exists(path) for path in targets):
                if args.memory_profile:
                    from src.training.recovery import validate_icl_result
                    validate_icl_result(args.output_path, run_fingerprint)
                print(f"skip-if-complete: {', '.join(targets)} already present; nothing to do")
                return

        # Run ICL inference
        results = run_icl_inference(
            model_name=args.model_name,
            num_examples=args.num_examples,
            selection_method=args.selection_method,
            test_data_path=args.test_data_path,
            train_data_path=args.train_data_path,
            output_path=args.output_path,
            max_samples=args.max_samples,
            seed=args.seed,
            max_context_utilization=args.max_context_utilization,
            verbose=args.verbose,
            dataset_name=args.dataset,
            prompt_version=args.prompt_version,
            demo_seed=args.demo_seed,
            batch_size=args.batch_size,
            attn_implementation=args.attn_implementation,
            max_prompt_tokens=args.max_prompt_tokens,
            allow_exact_match=args.allow_exact_match,
            memory_profile=args.memory_profile, run_fingerprint=run_fingerprint,
            sample_index_file=args.sample_index_file,
        )

        if judge_config is None:
            # Save results to JSON
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            if args.memory_profile:
                stop_if_requested()
                atomic_json(args.output_path, results)
            else:
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output_path}")
        elif args.judge_output_path:
            # Save the normal artifact first, then write the judged copy separately.
            save_json_result(args.output_path, results)
            print(f"Results saved to: {args.output_path}")
            judged_results = dict(results)
            append_judge_results_to_icl(
                judged_results,
                args.test_data_path,
                args.max_samples,
                DATASET_NAME,
                judge_config,
                args
            )
            save_json_result(args.judge_output_path, judged_results)
            print(f"Judged results saved to: {args.judge_output_path}")
        else:
            append_judge_results_to_icl(
                results,
                args.test_data_path,
                args.max_samples,
                DATASET_NAME,
                judge_config,
                args
            )
            # Save results to JSON
            save_json_result(args.output_path, results)
            print(f"Judged results saved to: {args.output_path}")

        # Upload to Hugging Face Hub (opt-in; was unconditional before)
        if args.upload:
            if is_placeholder_repo(HUGGINGFACE_REPO_ID):
                print(f"WARNING: HUGGINGFACE_REPO_ID is a placeholder ({HUGGINGFACE_REPO_ID}); skipping upload.")
            else:
                huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))

                api = huggingface_hub.HfApi()
                api.upload_large_folder(
                    folder_path=os.path.dirname(args.output_path),
                    repo_id=HUGGINGFACE_REPO_ID,
                    repo_type="model"
                )

        print("ICL inference completed successfully!")

if __name__ == "__main__":
    main()
