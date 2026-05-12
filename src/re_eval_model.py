# python src/re_eval_model.py --model_directory ./results-cc/code-t5/codet5_fmft_official_0.0001 --model_name codet5 --model_method fmft
from utils.helper import set_seed
set_seed()

from utils.helper import load_model_and_tokenizer, load_dataset_for_eval
from utils.constants import *
import evaluate
import torch
import json
import os
from tqdm import tqdm

"""
Read trained models in results directory infer the dataset and evaluate them
"""


JUDGE_CODE_COLUMNS = ("code_or_diff", "code", "patch", "diff", "source", "input", "function", "code_tokens")


def add_judge_arguments(parser):
    parser.add_argument('--llm-judge', dest='llm_judge', action='store_true',
                       help='Enable optional LLM-as-Judge scoring')
    parser.add_argument('--judge-model-path', dest='judge_model_path',
                       default='assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct',
                       help='Path to local Hugging Face-compatible judge model')
    parser.add_argument('--judge-k', dest='judge_k', type=int, default=5,
                       help='Number of judge runs per sample')
    parser.add_argument('--judge-cache-path', dest='judge_cache_path', default=None,
                       help='Optional JSONL cache path for judge runs')
    parser.add_argument('--judge-output-path', dest='judge_output_path', default=None,
                       help='Optional output path for judged results')
    parser.add_argument('--judge-local-files-only', dest='judge_local_files_only',
                       action='store_true', help='Load judge model with local_files_only=True')
    parser.add_argument('--judge-debug', dest='judge_debug', action='store_true',
                       help='Print per-sample judge errors')
    parser.add_argument('--judge-code-column', dest='judge_code_column', default=None,
                       help='Optional raw dataset column to use as judge code/diff input')
    parser.add_argument('--judge-prediction-column', dest='judge_prediction_column', default=None,
                       help='Optional result/raw column to use as judge prediction')
    parser.add_argument('--judge-reference-column', dest='judge_reference_column', default=None,
                       help='Optional result/raw column to use as judge reference')


def build_judge_config(args):
    from llm_as_judge import JudgeConfig

    return JudgeConfig(
        model_path=args.judge_model_path,
        k=args.judge_k,
        cache_path=args.judge_cache_path,
        local_files_only=args.judge_local_files_only,
    )


def load_raw_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def append_judge_results(raw_data, test_dataset_directory, judge_config, args):
    print("LLM-as-Judge enabled. Scoring generated predictions...")
    from llm_as_judge.qwen_judge import QwenJudge

    raw_samples = load_raw_jsonl(test_dataset_directory)
    judge = QwenJudge(judge_config)

    for index, row in enumerate(tqdm(raw_data, desc="LLM-as-Judge scoring")):
        sample = raw_samples[index] if index < len(raw_samples) else {}
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

    return raw_data


def save_json_result(path, results):
    results_dir = os.path.dirname(path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main(model_directory, model_name, model_method, test_dataset_directory,
         judge_config=None, judge_args=None):
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_directory, model_name, model_method)

    # Load the test dataset
    test_dataset = load_dataset_for_eval(test_dataset_directory, model_name, tokenizer)

    # Evaluate the model on the test dataset
    print("Starting test set evaluation with text generation...")
    
    # Load metrics
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    rouge_metric = evaluate.load("rouge")
    bert_score_metric = evaluate.load("bertscore")
    meteor_metric = evaluate.load("meteor")

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
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
                with torch.cuda.amp.autocast():
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
                generated_text = "<EMPTY>"  # Placeholder to avoid BLEU crash
            
            predictions.append(generated_text)
            references.append(reference_text)
            
            # Print first few examples
            print(f"Example {i+1}:")
            print(f"  Generated: {generated_text}")
            print(f"  Reference: {reference_text}")
            print()
    
    # Compute BLEU-4 score
    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )["bleu"]
    
    # Compute exact match using Hugging Face metric
    exact_match_score = exact_match_metric.compute(
        predictions=predictions,
        references=references
    )["exact_match"]

    # Compute ROUGE score (multiple scores)
    rouge_score = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    # Compute BERTScore (multiple scores)
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
    
    # Compute METEOR score
    meteor_score = meteor_metric.compute(
        predictions=predictions,
        references=references
    )["meteor"]

    # Print final results
    
    print(f"Test Results:")
    print(f"  BLEU-4: {bleu_score}")
    print(f"  Exact Match: {exact_match_score}")
    print(f"  ROUGE: {rouge_score}")
    print(f"  BERTScore: {bertscore}")
    print(f"  METEOR: {meteor_score}")
    
    # Prepare raw data
    raw_data = [{"prediction": pred, "reference": ref} for pred, ref in zip(predictions, references)]
    
    # Combine all results into one structure
    complete_results = {
        "test_bleu_4": bleu_score,
        "test_exact_match": exact_match_score,
        "test_rouge": rouge_score,
        "test_bertscore": bertscore,
        "test_meteor": meteor_score,
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
        append_judge_results(raw_data, test_dataset_directory, judge_config, judge_args)
        save_json_result(judge_args.judge_output_path, complete_results)
        print(f"Judged results saved to: {judge_args.judge_output_path}")
        return complete_results

    # Save everything to one JSON file
    append_judge_results(raw_data, test_dataset_directory, judge_config, judge_args)
    save_json_result(results_file, complete_results)

    print(f"Judged results saved to: {results_file}")

    return complete_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained models on test dataset')
    parser.add_argument('--model_directory', type=str, required=True,
                       help='Directory containing the trained model files')
    parser.add_argument('--model_name', type=str, required=True, 
                       choices=['codet5', 'plbart', 'qwen15', 'qwen25', 'qwen25-0.5b-instruct'],
                       help='Name of the model architecture')
    parser.add_argument('--model_method', type=str, required=True,
                       choices=['fmft', 'lora', 'qlora', 'ia3'],
                       help='Training method used for the model')
    parser.add_argument('--test_dataset_directory', type=str, default=TEST_DATA_DIR,
                       help='Path to test dataset JSONL file')
    add_judge_arguments(parser)
    
    args = parser.parse_args()
    judge_config = build_judge_config(args) if args.llm_judge else None
    
    # Run evaluation
    results = main(
        model_directory=args.model_directory,
        model_name=args.model_name,
        model_method=args.model_method,
        test_dataset_directory=args.test_dataset_directory,
        judge_config=judge_config,
        judge_args=args if args.llm_judge else None
    )
    
    print("Evaluation completed successfully!")
    print(f"Final Results: {results}")
