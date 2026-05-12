#!/usr/bin/env python3
# ------------------------------------------------------------------
# In-Context Learning (ICL) Inference Script
# ------------------------------------------------------------------
from utils.helper import set_seed
set_seed()

import sys
import os
from pathlib import Path
import argparse
import json
import random
from typing import List, Dict, Any

import huggingface_hub

# Add project root to Python path
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.constants import *
from src.utils.helper import load_jsonl

import evaluate
import torch
import warnings
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


JUDGE_CODE_COLUMNS = ("code_or_diff", "code", "patch", "diff", "source", "input", "function", "code_tokens")


def add_judge_arguments(parser):
    parser.add_argument("--llm-judge", dest="llm_judge", action="store_true",
                       help="Enable optional LLM-as-Judge scoring")
    parser.add_argument("--judge-model-path", dest="judge_model_path",
                       default="assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct",
                       help="Path to local Hugging Face-compatible judge model")
    parser.add_argument("--judge-k", dest="judge_k", type=int, default=5,
                       help="Number of judge runs per sample")
    parser.add_argument("--judge-cache-path", dest="judge_cache_path", default=None,
                       help="Optional JSONL cache path for judge runs")
    parser.add_argument("--judge-output-path", dest="judge_output_path", default=None,
                       help="Optional output path for judged results")
    parser.add_argument("--judge-local-files-only", dest="judge_local_files_only",
                       action="store_true", help="Load judge model with local_files_only=True")
    parser.add_argument("--judge-debug", dest="judge_debug", action="store_true",
                       help="Print per-sample judge errors")
    parser.add_argument("--judge-code-column", dest="judge_code_column", default=None,
                       help="Optional raw dataset column to use as judge code/diff input")
    parser.add_argument("--judge-prediction-column", dest="judge_prediction_column", default=None,
                       help="Optional result/raw column to use as judge prediction")
    parser.add_argument("--judge-reference-column", dest="judge_reference_column", default=None,
                       help="Optional result/raw column to use as judge reference")


def build_judge_config(args):
    from llm_as_judge import JudgeConfig

    return JudgeConfig(
        model_path=args.judge_model_path,
        k=args.judge_k,
        cache_path=args.judge_cache_path,
        local_files_only=args.judge_local_files_only,
    )


def value_as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def get_sample_code_for_judge(sample, explicit_column=None):
    if explicit_column:
        if explicit_column not in sample:
            raise ValueError(
                f"missing judge code column '{explicit_column}'; available columns: {sorted(sample.keys())}"
            )
        return value_as_text(sample[explicit_column])

    if DATASET_NAME == "trans_review" and "code_tokens" in sample:
        return value_as_text(sample["code_tokens"])
    if DATASET_NAME == "acra_by_ls_pt" and "patch" in sample:
        return value_as_text(sample["patch"])

    for column in JUDGE_CODE_COLUMNS:
        if column in sample:
            return value_as_text(sample[column])
    raise ValueError(f"could not detect judge code/diff column; available columns: {sorted(sample.keys())}")


def get_judge_text(row, sample, explicit_column, default_column, label):
    if explicit_column:
        if explicit_column in row:
            return value_as_text(row[explicit_column])
        if explicit_column in sample:
            return value_as_text(sample[explicit_column])
        raise ValueError(
            f"missing judge {label} column '{explicit_column}'; "
            f"available result columns: {sorted(row.keys())}; available raw columns: {sorted(sample.keys())}"
        )
    return value_as_text(row[default_column])


def failed_judge_result(judge_config):
    return {
        "qwen_judge_score": float("nan"),
        "qwen_judge_mean": float("nan"),
        "qwen_judge_median": float("nan"),
        "qwen_judge_std": float("nan"),
        "qwen_judge_agreement": float("nan"),
        "qwen_judge_norm": float("nan"),
        "qwen_judge_k": judge_config.k,
        "qwen_judge_valid_runs": 0,
        "qwen_judge_parse_errors": judge_config.k,
        "qwen_judge_model": judge_config.model_path,
        "qwen_judge_prompt_version": judge_config.prompt_version,
    }


def append_judge_results_to_icl(results, test_data_path, max_samples, judge_config, args):
    print("LLM-as-Judge enabled. Scoring generated ICL predictions...")
    from llm_as_judge.qwen_judge import QwenJudge

    raw_samples = load_jsonl(test_data_path)
    if max_samples:
        raw_samples = raw_samples[:max_samples]

    judge = QwenJudge(judge_config)
    predictions = results.get("predictions", [])
    references = results.get("references", [])
    raw_data = []

    for index, prediction in enumerate(tqdm(predictions, desc="LLM-as-Judge scoring")):
        sample = raw_samples[index] if index < len(raw_samples) else {}
        reference = references[index] if index < len(references) else ""
        row = {"prediction": prediction, "reference": reference}
        try:
            record = {
                "code_or_diff": get_sample_code_for_judge(sample, args.judge_code_column),
                "prediction": get_judge_text(row, sample, args.judge_prediction_column, "prediction", "prediction"),
                "reference": get_judge_text(row, sample, args.judge_reference_column, "reference", "reference"),
            }
            row.update(judge.score_record(record))
        except Exception as exc:
            if args.judge_debug:
                print(f"WARNING: LLM-as-Judge failed for sample {index}: {exc}")
            row.update(failed_judge_result(judge_config))
        raw_data.append(row)

    results["raw_data"] = raw_data
    return results


def save_json_result(path, results):
    results_dir = os.path.dirname(path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


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


def load_base_model_and_tokenizer(model_name: str):
    """Load the base model and tokenizer for ICL inference.
    
    Args:
        model_name: Name of the model ("codet5", "plbart", "qwen15", "qwen25", "qwen25-0.5b")
    
    Returns:
        tuple: (model, tokenizer)
    """
    model_mapping = {
        "codet5": (MODEL_CT5, TOKENIZER_CT5),
        "plbart": (MODEL_PLBART, TOKENIZER_PLBART),
        "qwen15": (MODEL_QWEN15, TOKENIZER_QWEN15),
        "qwen25": (MODEL_QWEN25, TOKENIZER_QWEN25),
        "qwen25-0.5b": (MODEL_QWEN25_0_5B, TOKENIZER_QWEN25_0_5B)
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
    elif model_name in ["qwen15", "qwen25", "qwen25-0.5b"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
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
    if num_examples <= 0:
        return []
    
    if num_examples >= len(train_data):
        selected = train_data
    elif selection_method == "random":
        random.seed(seed)
        selected = random.sample(train_data, num_examples)
    elif selection_method == "first":
        selected = train_data[:num_examples]
    elif selection_method == "diverse":
        # Simple diversity: select examples spread across the dataset
        step = len(train_data) // num_examples
        selected = [train_data[i * step] for i in range(num_examples)]
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # If model and tokenizer provided, check if we can fit more examples
    if model and tokenizer and current_sample:
        selected = optimize_icl_examples_for_context(selected, current_sample, model_name, tokenizer, model)
    
    return selected


def optimize_icl_examples_for_context(selected_examples: List[Dict], current_sample: Dict, 
                                     model_name: str, tokenizer, model) -> List[Dict]:
    """Optimize ICL examples to maximize context usage without exceeding limits.
    
    Args:
        selected_examples: Initially selected examples
        current_sample: Current test sample
        model_name: Name of the model (for prompt building)
        tokenizer: Tokenizer for length calculation
        model: The loaded model
    
    Returns:
        Optimized list of examples
    """
    if DATASET_NAME == "trans_review":
        test_code = " ".join(current_sample["code_tokens"])
    elif DATASET_NAME == "acra_by_ls_pt":
        test_code = current_sample['patch']
    max_context = get_max_context_length(model)
    
    if model_name in ["codet5", "plbart"]:
        # For seq2seq models, build prompt and check length
        prompt = build_icl_prompt_seq2seq(selected_examples, test_code)
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        
        if len(tokens) <= max_context:
            return selected_examples
        
        # If too long, try to fit as many examples as possible
        optimized_examples = []
        for example in selected_examples:
            temp_examples = optimized_examples + [example]
            temp_prompt = build_icl_prompt_seq2seq(temp_examples, test_code)
            temp_tokens = tokenizer.encode(temp_prompt, add_special_tokens=True)
            
            if len(temp_tokens) <= max_context:
                optimized_examples.append(example)
            else:
                break
        
        if len(optimized_examples) < len(selected_examples):
            warnings.warn(
                f"Reduced ICL examples from {len(selected_examples)} to {len(optimized_examples)} "
                f"to fit within model context limit ({max_context} tokens)",
                UserWarning
            )
        
        return optimized_examples
    
    else:  # Causal LM models
        # For causal models, build messages and check length
        messages = build_icl_prompt_causal(selected_examples, test_code, tokenizer)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        
        # Reserve space for output
        available_context = max_context - OUTPUT_MAX_LENGTH
        
        if len(tokens) <= available_context:
            return selected_examples
        
        # If too long, try to fit as many examples as possible
        optimized_examples = []
        for example in selected_examples:
            temp_examples = optimized_examples + [example]
            temp_messages = build_icl_prompt_causal(temp_examples, test_code, tokenizer)
            temp_prompt = tokenizer.apply_chat_template(temp_messages, tokenize=False, add_generation_prompt=True)
            temp_tokens = tokenizer.encode(temp_prompt, add_special_tokens=True)
            
            if len(temp_tokens) <= available_context:
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
    """Build ICL prompt for seq2seq models (CodeT5, PLBART).
    
    Args:
        examples: List of ICL examples
        test_code: Test code to generate review for
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add examples
    for example in examples:
        if DATASET_NAME == "trans_review":
            code = " ".join(example["code_tokens"])
            review = " ".join(example["docstring_tokens"])
        elif DATASET_NAME == "acra_by_ls_pt":
            code = example['patch']
            review = example['msg']
        prompt_parts.append(f"Code: {code}\nReview: {review}")
    
    # Add test case
    prompt_parts.append(f"Code: {test_code}\nReview:")
    
    return "\n\n".join(prompt_parts)


def build_icl_prompt_causal(examples: List[Dict], test_code: str, tokenizer) -> List[Dict]:
    """Build ICL prompt for causal LM models (Qwen1.5, Qwen2.5).
    
    Args:
        examples: List of ICL examples
        test_code: Test code to generate review for
        tokenizer: Tokenizer for chat template
    
    Returns:
        Messages list for chat template
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a code reviewer. Output exactly ONE concise, actionable comment "
                "on the code. No restating, no extra context, no pleasantries. "
                "Recommend to use up to 60 words. (longer if needed)"
            ),
        }
    ]
    
    # Add ICL examples as conversation history
    for example in examples:
        if DATASET_NAME == "trans_review":
            code = " ".join(example["code_tokens"])
            review = " ".join(example["docstring_tokens"])
        elif DATASET_NAME == "acra_by_ls_pt":
            code = example['patch']
            review = example['msg']
        
        suffix = " changes" if DATASET_NAME == "acra_by_ls_pt" else ""
        messages.extend([
            {
                "role": "user",
                "content": f"Code{suffix}:\n```{code}```\nReview comment?",
            },
            {
                "role": "assistant",
                "content": review.strip(),
            }
        ])
    
    # Add current test case
    messages.append({
        "role": "user",
        "content": f"Code:\n```{test_code}```\nReview comment?",
    })
    
    return messages


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
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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


def run_icl_inference(model_name: str, num_examples: int, selection_method: str = "random",
                     test_data_path: str = TEST_DATA_DIR, train_data_path: str = TRAIN_DATA_DIR,
                     output_path: str = None, max_samples: int = None, seed: int = SEED,
                     max_context_utilization: bool = False) -> Dict[str, Any]:
    """Run ICL inference on test dataset.
    
    Args:
        model_name: Name of the model to use
        num_examples: Number of ICL examples to use
        selection_method: Method for selecting ICL examples
        test_data_path: Path to test dataset
        train_data_path: Path to training dataset
        output_path: Path to save results
        max_samples: Maximum number of test samples to process
        seed: Random seed
        max_context_utilization: Enable aggressive context maximization
    
    Returns:
        Dictionary with evaluation results including predictions, references, and prompts
    """
    
    # Configure warnings based on max_context_utilization flag
    if not max_context_utilization:
        warnings.filterwarnings("ignore", category=UserWarning, module=__name__)
    
    print(f"Context utilization mode: {'Aggressive' if max_context_utilization else 'Standard'}")
    
    # Load model and tokenizer
    print(f"Loading {model_name} model...")
    model, tokenizer = load_base_model_and_tokenizer(model_name)
    
    print(f"Max context length for {model_name}: {get_max_context_length(model)} tokens")
    
    # Load datasets
    print("Loading datasets...")
    train_data = load_jsonl(train_data_path)
    test_data = load_jsonl(test_data_path)
    
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"Processing {len(test_data)} test samples")
    
    predictions = []
    references = []
    prompts = []
    context_truncation_count = 0
    example_reduction_count = 0
    
    # Process each test sample
    print(f"Running ICL inference with {num_examples} examples...")
    for i, test_sample in enumerate(tqdm(test_data, desc="Generating reviews")):
        try:
            # Select ICL examples
            original_num_examples = num_examples
            icl_examples = select_icl_examples(
                train_data, num_examples, selection_method, test_sample, seed + i, 
                model_name, tokenizer, model
            )
            
            # Track if examples were reduced due to context limits
            if len(icl_examples) < original_num_examples:
                example_reduction_count += 1
            
            # Prepare input
            if DATASET_NAME == "trans_review":
                test_code = " ".join(test_sample["code_tokens"])
                reference = " ".join(test_sample["docstring_tokens"])
            elif DATASET_NAME == "acra_by_ls_pt":
                test_code = test_sample['patch']
                reference = test_sample['msg']

            # Generate based on model type
            if model_name in ["codet5", "plbart"]:
                prompt = build_icl_prompt_seq2seq(icl_examples, test_code)
                prompts.append(prompt)
                # Check if this specific prompt was truncated
                if check_context_length_warning(prompt, tokenizer, model, f"sample {i+1} prompt"):
                    context_truncation_count += 1
                generated_review = generate_review_seq2seq(model, tokenizer, prompt)
            else:  # qwen15, qwen25
                messages = build_icl_prompt_causal(icl_examples, test_code, tokenizer)
                # Convert messages to string format for storage
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
                # Check if this specific prompt was truncated
                if check_context_length_warning(prompt, tokenizer, model, f"sample {i+1} prompt"):
                    context_truncation_count += 1
                generated_review = generate_review_causal(model, tokenizer, messages)
            
            # Handle empty generations
            if not generated_review.strip():
                print(f"WARNING: Empty generation at sample {i}")
                generated_review = "<EMPTY>"
            
            predictions.append(generated_review)
            references.append(reference)

            # Print example details
            print(f"\nExample {i+1}:")
            print(f"  ICL Examples Used: {len(icl_examples)}/{original_num_examples}")
            print(f"  Code: {test_code[:100]}...")
            print(f"  Prompt: {prompt[:200]}...")
            print(f"  Generated: {generated_review}")
            print(f"  Reference: {reference}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            predictions.append("<ERROR>")
            references.append(" ".join(test_sample["docstring_tokens"]))
            prompts.append("<ERROR>")
    
    # Load evaluation metrics
    print("Loading evaluation metrics...")
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    rouge_metric = evaluate.load("rouge")
    bert_score_metric = evaluate.load("bertscore")
    meteor_metric = evaluate.load("meteor")

    # Compute BLEU score
    print("Computing evaluation metrics...")
    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )["bleu"]
    
    exact_match_score = exact_match_metric.compute(
        predictions=predictions,
        references=references
    )["exact_match"]

    rouge_score = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    bertscore = bert_score_metric.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    bertscore = {
        "precision": sum(bertscore["precision"]) / len(bertscore["precision"]),
        "recall": sum(bertscore["recall"]) / len(bertscore["recall"]),
        "f1": sum(bertscore["f1"]) / len(bertscore["f1"])
    }
    
    meteor_score = meteor_metric.compute(
        predictions=predictions,
        references=references
    )["meteor"]
    
    # Prepare results
    results = {
        "model_name": model_name,
        "num_icl_examples": num_examples,
        "selection_method": selection_method,
        "num_test_samples": len(test_data),
        "bleu_score": bleu_score,
        "exact_match_score": exact_match_score,
        "rouge_score": rouge_score,
        "bertscore": bertscore,
        "meteor_score": meteor_score,
        "predictions": predictions,
        "references": references,
        "prompts": prompts,
        "seed": seed,
        "max_context_length": get_max_context_length(model),
        "context_truncation_count": context_truncation_count,
        "example_reduction_count": example_reduction_count,
        "max_context_utilization": max_context_utilization
    }
    
    print(f"\nResults:")
    print(f"  Model: {model_name}")
    print(f"  ICL Examples: {num_examples}")
    print(f"  Selection Method: {selection_method}")
    print(f"  Test Samples: {len(test_data)}")
    print(f"  Max Context Length: {get_max_context_length(model)}")
    print(f"  Context Truncations: {context_truncation_count}/{len(test_data)}")
    print(f"  Example Reductions: {example_reduction_count}/{len(test_data)}")
    print(f"  BLEU-4: {bleu_score:.4f}")
    print(f"  Exact Match: {exact_match_score:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run In-Context Learning (ICL) inference for code review generation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["codet5", "plbart", "qwen15", "qwen25", "qwen25-0.5b"],
                       help="Name of the model to use")
    
    # ICL arguments
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of ICL examples to use (default: 3)")
    parser.add_argument("--selection_method", type=str, default="random",
                       choices=["random", "first", "diverse"],
                       help="Method for selecting ICL examples (default: random)")
    parser.add_argument("--max_context_utilization", action="store_true",
                       help="Enable aggressive context length maximization with warnings")
    
    # Data arguments
    parser.add_argument("--test_data_path", type=str, default=TEST_DATA_DIR,
                       help="Path to test dataset JSONL file")
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_DIR,
                       help="Path to training dataset JSONL file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of test samples to process")
    
    # Output arguments
    parser.add_argument("--output_path", type=str, default=None,
                       help="Path to save results JSON file")
    parser.add_argument("--results_dir", type=str, default="./results/icl/",
                       help="Directory to save results (used if output_path not specified)")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=SEED,
                       help="Random seed for reproducibility")
    add_judge_arguments(parser)
    
    args = parser.parse_args()
    judge_config = build_judge_config(args) if args.llm_judge else None
    
    # Generate output path if not provided
    if args.output_path is None:
        filename = f"{args.model_name}_icl_{args.num_examples}ex_{args.selection_method}.json"
        args.output_path = os.path.join(args.results_dir, filename)
    
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
        max_context_utilization=args.max_context_utilization
    )
    
    if judge_config is None:
        # Save results to JSON
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
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
            judge_config,
            args
        )
        # Save results to JSON
        save_json_result(args.output_path, results)
        print(f"Judged results saved to: {args.output_path}")
    
    # Upload to Hugging Face Hub
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
