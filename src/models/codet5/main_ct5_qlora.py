from src.utils.helper import set_seed
set_seed()

#!/usr/bin/env python3
# Setup Python path for imports
import sys
import os
from pathlib import Path
import argparse

# Add project root to Python path
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import wandb
import huggingface_hub
import evaluate
import json

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration, BitsAndBytesConfig, GenerationConfig, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
from tqdm import tqdm

from src.utils.constants import *
from src.utils.helper import *
from src.utils.custom_modified_api import TimeMemoryCallback


def main(experiment_id, learning_rate, subset):
    # [PREPARING CONSTANT PARAMETER]
    if learning_rate:
        LR_CT5 = learning_rate
        EXPERIMENT_NAME_CT5 = "codet5_qlora_official_" + str(learning_rate)
    else:
        LR_CT5 = LR[experiment_id]
        EXPERIMENT_NAME_CT5 = f"codet5_qlora_{experiment_id}"

    # Initialize experiment
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME_CT5)

    # Prepare dataset and tokenizer
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset_for_fedm(TOKENIZER_CT5, subset_percentage=subset)

    # Setup experiment directory
    EXPERIMENT_RESULTS_DIR = RESULTS_DIR_CT5 + EXPERIMENT_NAME_CT5
    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)

    # Check for existing checkpoints
    def get_last_checkpoint(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
            return os.path.join(output_dir, last_checkpoint)
        return None

    checkpoint = get_last_checkpoint(EXPERIMENT_RESULTS_DIR)

    # Create model configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load base model with quantization
    base_model = T5ForConditionalGeneration.from_pretrained(
        MODEL_CT5,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.generation_config.num_beams = NUM_BEAMS
    base_model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH

    def setup_lora_parameters(model):
        """Ensure LoRA parameters are properly configured for training."""
        for name, param in model.named_parameters():
            if 'lora_' in name or 'adapter' in name:
                param.requires_grad = True

    def count_trainable_params(model):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if checkpoint:
        # Resume from checkpoint
        model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)
        model.train()
        setup_lora_parameters(model)
        print(f"Resumed from checkpoint: {checkpoint}")
    else:
        # Prepare model for LoRA training
        model = prepare_model_for_kbit_training(base_model)

        # Create new LoRA model
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=RANK_QLORA,
            lora_alpha=ALPHA_QLORA,
            lora_dropout=0.05,
            target_modules=["q", "v", "k", "o", "wi", "wo"],
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(model, lora_cfg)
        print("Created new LoRA model")

    # Enable gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Count and log trainable parameters
    trainable_params = count_trainable_params(model)
    wandb.config.update({"system/trainable_params": trainable_params}, allow_val_change=True)
    print(f"Trainable parameters: {trainable_params/1e6:.2f} M")

    # Create training arguments
    training_args = Seq2SeqTrainingArguments(
        run_name=EXPERIMENT_NAME_CT5,
        report_to="wandb",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR_CT5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_CT5,
        per_device_eval_batch_size=EVAL_BATCH_SIZE_CT5,
        num_train_epochs=NUM_TRAIN_EPOCHS_CT5,
        weight_decay=WEIGHT_DECAY_CT5,
        output_dir=EXPERIMENT_RESULTS_DIR,
        logging_dir=f"{EXPERIMENT_RESULTS_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu_4",
        greater_is_better=True,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        seed=SEED,
        predict_with_generate=True,
        optim="paged_adamw_8bit",
        generation_config=GenerationConfig(
            max_new_tokens=OUTPUT_MAX_LENGTH,
            min_new_tokens=OUTPUT_MIN_LENGTH,
            num_beams=NUM_BEAMS,
            decoder_start_token_id=base_model.config.decoder_start_token_id,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    )

    # Create trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    

    # Load metrics for evaluation
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    rouge_metric = evaluate.load("rouge")
    bert_score_metric = evaluate.load("bertscore")
    meteor_metric = evaluate.load("meteor")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode predictions and labels
        preds = [[pred for pred in pred_seq if pred != -100] for pred_seq in preds]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = [[label for label in label_seq if label != -100] for label_seq in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Print details for debugging
        print("Result of first example in the eval batch:")
        print(f"Decoded Predictions Sample: {decoded_preds[0]}")
        print(f"Decoded Labels Sample: {decoded_labels[0]}")
        
        # Compute BLEU-4 score
        bleu_score = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )["bleu"]
        
        # Compute exact match
        exact_match_score = exact_match_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["exact_match"]
        
        # Compute ROUGE score (multiple scores)
        rouge_score = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Compute BERTScore (multiple scores)
        bertscore = bert_score_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            lang="en"
        )
        bertscore = {
            "precision": sum(bertscore["precision"]) / len(bertscore["precision"]),
            "recall": sum(bertscore["recall"]) / len(bertscore["recall"]),
            "f1": sum(bertscore["f1"]) / len(bertscore["f1"])
        }
        
        # Compute METEOR score
        meteor_score = meteor_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["meteor"]
        
        results = {
            "eval_bleu_4": bleu_score,
            "eval_exact_match": exact_match_score,
            "eval_rouge": rouge_score,
            "eval_bertscore": bertscore,
            "eval_meteor": meteor_score
        }
        
        return results

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[TimeMemoryCallback()],
    )

    # Train the model
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # ------------------------------------------------------------------
    # Manual test set evaluation with text generation
    # ------------------------------------------------------------------
    print("Starting test set evaluation with text generation...")
    
    # Load metrics
    bleu_metric = evaluate.load("bleu")
    exact_match_metric = evaluate.load("exact_match")
    
    # Set model to evaluation mode
    model.eval()
    
    predictions = []
    references = []
    
    # Generate text for each test sample
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
    )
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating test set")):
            labels = batch["labels"]
            
            # Move batch to the appropriate device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Generate text
            with torch.cuda.amp.autocast():
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=OUTPUT_MAX_LENGTH,
                    min_new_tokens=OUTPUT_MIN_LENGTH,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                )
            
            # Decode generated text
            print(f"Generated IDs: {generated_ids}")  # Debugging line
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # After decoding
            if len(generated_text.strip()) == 0:
                print(f"WARNING: Empty generation at sample {i}")
                generated_text = "<EMPTY>"  # Placeholder to avoid BLEU crash
            
            # Get reference text from labels
            reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

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
    
    # Log to wandb
    wandb.log({
        "test_bleu_4": bleu_score,
        "test_exact_match": exact_match_score,
        "test_rouge": rouge_score,
        "test_bertscore": bertscore,
        "test_meteor": meteor_score
    })

    # Evaluate on test set

    # Save model, tokenizer, and results
    model.save_pretrained(EXPERIMENT_RESULTS_DIR)
    tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR)

    with open(f"{EXPERIMENT_RESULTS_DIR}/training_args.txt", "w") as f:
        f.write(str(training_args))

    # Save everything to one JSON file
    with open(EXPERIMENT_RESULTS_DIR + "/complete_results.json", "w") as f:
        json.dump(complete_results, f, indent=2)

    # Upload to HuggingFace Hub
    if "official" in EXPERIMENT_NAME_CT5:
        api = huggingface_hub.HfApi()
        api.upload_large_folder(
            folder_path=RESULTS_DIR_CT5,
            repo_id=HUGGINGFACE_REPO_ID,
            repo_type="model",
            private=False,
            num_workers=NUM_WORKERS,
            allow_patterns="*official*/**",  # glob pattern
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QLoRA fine-tuning experiment")
    parser.add_argument("--experiment_id", type=int, default=None,
                        help="Experiment ID for tracking multiple runs")
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override the learning rate')
    parser.add_argument('--subset', type=float, default=1.0,
                       help='Fraction of training data to use (0 to 1)')
    args = parser.parse_args()

    main(args.experiment_id, args.learning_rate, args.subset)
