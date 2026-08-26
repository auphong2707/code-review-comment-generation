# python -m src.re_eval_model --model_directory assets/results/code-t5/codet5_fmft_official_0.0001 --model_name codet5 --model_method fmft
import sys
from pathlib import Path

# Keep direct-script invocation (python src/re_eval_model.py) working
sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from src.utils.helper import set_seed
set_seed()

from src.utils.helper import load_model_and_tokenizer, load_dataset_for_eval, normalize_model_name
from src.utils.constants import *
from src.evaluation.metrics import EMPTY_PLACEHOLDER, as_test_results, compute_generation_metrics
import torch
import json
import os
from tqdm import tqdm

"""
Read trained models in results directory infer the dataset and evaluate them
"""


from src.llm_as_judge.integration import (
    add_judge_arguments,
    append_judge_results,
    build_judge_config,
    save_json_result,
)


def main(model_directory, model_name, model_method, test_dataset_directory,
         judge_config=None, judge_args=None, dataset_name=None, verbose=False):
    dataset_name = dataset_name or DATASET_NAME
    model_name = normalize_model_name(model_name)

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_directory, model_name, model_method)

    # Load the test dataset
    test_dataset = load_dataset_for_eval(test_dataset_directory, model_name, tokenizer, dataset_name=dataset_name)

    # Evaluate the model on the test dataset
    print("Starting test set evaluation with text generation...")
    
    # Set model to evaluation mode
    model.eval()
    
    predictions = []
    references = []
    
    # Generate text for each test sample
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataset, desc="Evaluating test set")):
            # Prepare input
            input_ids = torch.tensor([batch['input_ids']]).to(model.device)
            attention_mask = torch.tensor([batch['attention_mask']]).to(model.device)
            
            # For causal LM models (Qwen), we need to handle differently
            if model_name in ["qwen15", "qwen25", "qwen25-0.5b-instruct"]:
                # Find the length of the input to extract only the generated part
                input_length = len(batch['input_ids'])

                # Generate text
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        input_ids,
                        max_new_tokens=OUTPUT_MAX_LENGTH,
                        min_new_tokens=OUTPUT_MIN_LENGTH,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Extract only the generated part (remove input)
                generated_ids = generated_ids[0][input_length:]

                # Debugging line
                if verbose:
                    print(f"Generated IDs: {generated_ids}")

                # Decode generated text
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Get reference text (assuming labels in the batch)
                if 'labels' in batch:
                    # Remove -100 tokens and decode reference
                    labels = batch['labels']
                    labels = [token_id for token_id in labels if token_id != -100]
                    reference_text = tokenizer.decode(labels, skip_special_tokens=True)
                else:
                    # Fallback: assume the target is the part after input in input_ids
                    reference_text = tokenizer.decode(batch['input_ids'][input_length:], skip_special_tokens=True)
            
            else:  # For seq2seq models (CodeT5, PLBART)
                # Generate text
                with torch.autocast("cuda"):
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=OUTPUT_MAX_LENGTH,
                        min_new_tokens=OUTPUT_MIN_LENGTH,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        decoder_start_token_id=tokenizer.bos_token_id if model_name == "plbart" else model.config.decoder_start_token_id,
                    )
                
                # Decode generated text
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Get reference text from labels
                labels = batch['labels']
                labels = [token_id for token_id in labels if token_id != -100]
                reference_text = tokenizer.decode(labels, skip_special_tokens=True)
            
            # After decoding
            if len(generated_text.strip()) == 0:
                print(f"WARNING: Empty generation at sample {i}")
                generated_text = EMPTY_PLACEHOLDER  # Placeholder to avoid BLEU crash
            
            predictions.append(generated_text)
            references.append(reference_text)

            # Print examples only when asked (thousands of lines on full runs)
            if verbose:
                print(f"Example {i+1}:")
                print(f"  Generated: {generated_text}")
                print(f"  Reference: {reference_text}")
                print()
    
    # Compute the frozen metric suite (single shared implementation)
    metrics = compute_generation_metrics(predictions, references)

    # Print final results

    print(f"Test Results:")
    print(f"  BLEU-4: {metrics['bleu_4']}")
    print(f"  Exact Match: {metrics['exact_match']}")
    print(f"  ROUGE: {metrics['rouge']}")
    print(f"  BERTScore: {metrics['bertscore']}")
    print(f"  METEOR: {metrics['meteor']}")

    # Prepare raw data
    raw_data = [{"prediction": pred, "reference": ref} for pred, ref in zip(predictions, references)]

    # Combine all results into one structure (legacy test_* schema)
    complete_results = {
        **as_test_results(metrics),
        "raw_data": raw_data
    }
    
    results_file = os.path.join(model_directory, "re_evaluate_test_results.json")

    if judge_config is None:
        # Save everything to one JSON file
        with open(results_file, "w") as f:
            json.dump(complete_results, f, indent=2)

        print(f"Complete results saved to: {results_file}")

        return complete_results

    if judge_args.judge_output_path:
        save_json_result(results_file, complete_results)
        print(f"Unjudged results saved to: {results_file}")
        append_judge_results(raw_data, test_dataset_directory, dataset_name, judge_config, judge_args)
        save_json_result(judge_args.judge_output_path, complete_results)
        print(f"Judged results saved to: {judge_args.judge_output_path}")
        return complete_results

    # Save everything to one JSON file
    append_judge_results(raw_data, test_dataset_directory, dataset_name, judge_config, judge_args)
    save_json_result(results_file, complete_results)

    print(f"Judged results saved to: {results_file}")

    return complete_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained models on test dataset')
    parser.add_argument('--model_directory', type=str, required=True,
                       help='Directory containing the trained model files')
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['codet5', 'plbart', 'qwen15', 'qwen25', 'qwen25-0.5b-instruct', 'qwen25-0.5b'],
                       help='Name of the model architecture')
    parser.add_argument('--model_method', type=str, required=True,
                       choices=['fmft', 'lora', 'qlora', 'ia3'],
                       help='Training method used for the model')
    parser.add_argument('--test_dataset_directory', type=str, default=None,
                       help='Path to test dataset JSONL file (default: from the --dataset registry)')
    parser.add_argument('--dataset', type=str, default=DATASET_NAME,
                       choices=sorted(DATASET_CONFIGS),
                       help='Dataset the test file belongs to (default: constants.DATASET_NAME)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print per-sample generations during evaluation')
    add_judge_arguments(parser)

    args = parser.parse_args()
    judge_config = build_judge_config(args) if args.llm_judge else None

    # Resolve the default test path from the dataset registry (identical to
    # the legacy TEST_DATA_DIR constant for the default dataset)
    if args.test_dataset_directory is None:
        args.test_dataset_directory = str(get_dataset_config(args.dataset)["data_files"]["test"])

    # Run evaluation
    results = main(
        model_directory=args.model_directory,
        model_name=args.model_name,
        model_method=args.model_method,
        test_dataset_directory=args.test_dataset_directory,
        judge_config=judge_config,
        judge_args=args if args.llm_judge else None,
        dataset_name=args.dataset,
        verbose=args.verbose
    )
    
    print("Evaluation completed successfully!")
    print(f"Final Results: {results}")
