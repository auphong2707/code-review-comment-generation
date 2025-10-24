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

def main(model_directory, model_name, model_method, test_dataset_directory):
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
    
    # Save everything to one JSON file
    results_file = os.path.join(model_directory, "re_evaluate_test_results.json")
    with open(results_file, "w") as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"Complete results saved to: {results_file}")
    
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
    
    args = parser.parse_args()
    
    # Run evaluation
    results = main(
        model_directory=args.model_directory,
        model_name=args.model_name,
        model_method=args.model_method,
        test_dataset_directory=args.test_dataset_directory
    )
    
    print("Evaluation completed successfully!")
    print(f"Final Results: {results}")
    