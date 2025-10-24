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

# ------------------------------------------------------------------
# IA3 PEFT for Qwen/CodeQwen1.5-7B-Chat
# ------------------------------------------------------------------
import os, evaluate, wandb, huggingface_hub, torch, json
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from peft import get_peft_model, PeftModel, IA3Config, TaskType
from src.utils.constants import *
from src.utils.helper import *
from src.utils.custom_modified_api import TimeMemoryCallback, LLMCustomTrainer
from tqdm import tqdm


def main(experiment_id, learning_rate, subset):
    # ------------------------------------------------------------------
    # 1. House-keeping
    # ------------------------------------------------------------------
    if learning_rate:
        LR_QWEN15 = learning_rate
        EXPERIMENT_NAME_QWEN15 = "qwen15_ia3_official_" + str(learning_rate)
    else:
        LR_QWEN15 = LR[experiment_id]
        EXPERIMENT_NAME_QWEN15 = f"qwen15_ia3_{experiment_id}"
    EXPERIMENT_RESULTS_DIR = RESULTS_DIR_QWEN15 + EXPERIMENT_NAME_QWEN15

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME_QWEN15)

    # ------------------------------------------------------------------
    # 2. Dataset & tokenizer
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds, tokenizer = prepare_dataset_for_qwen_llms(TOKENIZER_QWEN15, subset_percentage=subset)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    def latest_ckpt(outdir):
        ckpts = [d for d in os.listdir(outdir) if d.startswith("checkpoint")]
        return os.path.join(outdir, sorted(ckpts, key=lambda x: int(x.split('-')[-1]))[-1]) if ckpts else None

    ckpt_path = latest_ckpt(EXPERIMENT_RESULTS_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_QWEN15,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    base_model.generation_config.num_beams = NUM_BEAMS
    base_model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH
    base_model.resize_token_embeddings(len(tokenizer))

    if hasattr(base_model, 'gradient_checkpointing_disable'):
        base_model.gradient_checkpointing_disable()
    if hasattr(base_model, 'config'):
        base_model.config.use_cache = True

    ia3_cfg = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        feedforward_modules=["gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        init_ia3_weights=True,
    )

    model = get_peft_model(base_model, ia3_cfg)
    model.train()

    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    if hasattr(model, 'config'):
        model.config.use_cache = True
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'gradient_checkpointing_disable'):
        model.base_model.gradient_checkpointing_disable()
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
        model.base_model.config.use_cache = True

    for name, param in model.named_parameters():
        if 'ia3' in name.lower():
            param.requires_grad_(True)
            with torch.no_grad():
                if torch.all(param.data == 0.0):
                    param.data.fill_(1.0)

    trainable_params = count_trainable(model)
    wandb.config.update({"trainable_params": trainable_params}, allow_val_change=True)
    print(f"Trainable parameters: {trainable_params/1e6:.2f} M")

    # ------------------------------------------------------------------
    # 4. TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        run_name=EXPERIMENT_NAME_QWEN15,
        report_to="wandb",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR_QWEN15,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_QWEN15,
        per_device_eval_batch_size=EVAL_BATCH_SIZE_QWEN15,
        num_train_epochs=NUM_TRAIN_EPOCHS_QWEN15,
        weight_decay=WEIGHT_DECAY_QWEN15,
        output_dir=EXPERIMENT_RESULTS_DIR,
        logging_dir=EXPERIMENT_RESULTS_DIR + "/logs",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu_4",
        greater_is_better=True,
        bf16=True,
        fp16=False,
        save_total_limit=2,
        seed=SEED,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        deepspeed=None  # No deepspeed for IA3
    )

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    generation_config = GenerationConfig(
        max_new_tokens=OUTPUT_MAX_LENGTH,
        min_new_tokens=OUTPUT_MIN_LENGTH,
        num_beams=NUM_BEAMS,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    trainer = LLMCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        # Custom arguments for LLMTrainer
        tokenizer_gen=tokenizer,
        generation_config=generation_config,
        debug=True,
        callbacks=[TimeMemoryCallback()],
    )

    trainer.train(resume_from_checkpoint=ckpt_path) if ckpt_path else trainer.train()

    # ------------------------------------------------------------------
    # 6. Evaluation on test set with text generation
    # ------------------------------------------------------------------
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
        for i, batch in enumerate(tqdm(test_ds, desc="Evaluating test set")):

            
            # Prepare input (assuming the dataset has input_ids)
            input_ids = torch.tensor([batch['input_ids']]).to(model.device)
            
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
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Extract only the generated part (remove input)
            generated_ids = generated_ids[0][input_length:]
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # After decoding
            if len(generated_text.strip()) == 0:
                print(f"WARNING: Empty generation at sample {i}")
                generated_text = "<EMPTY>"  # Placeholder to avoid BLEU crash
            
            # Get reference text (assuming labels in the batch)
            if 'labels' in batch:
                # Remove -100 tokens and decode reference
                labels = batch['labels']
                labels = [token_id for token_id in labels if token_id != -100]
                reference_text = tokenizer.decode(labels, skip_special_tokens=True)
            else:
                # Fallback: assume the target is the part after input in input_ids
                reference_text = tokenizer.decode(batch['input_ids'][input_length:], skip_special_tokens=True)
            
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

    # ------------------------------------------------------------------
    # 7. Save, log, push
    # ------------------------------------------------------------------
    model.save_pretrained(EXPERIMENT_RESULTS_DIR)
    tokenizer.save_pretrained(EXPERIMENT_RESULTS_DIR)
    with open(f"{EXPERIMENT_RESULTS_DIR}/training_args.txt", "w") as f:
        f.write(str(training_args))
    
    # Save everything to one JSON file
    with open(EXPERIMENT_RESULTS_DIR + "/complete_results.json", "w") as f:
        json.dump(complete_results, f, indent=2)

    if "official" in EXPERIMENT_NAME_QWEN15:
        api = huggingface_hub.HfApi()
        api.upload_large_folder(
            folder_path=RESULTS_DIR_QWEN15,
            repo_id=HUGGINGFACE_REPO_ID,
            repo_type="model",
            private=False,
            num_workers=NUM_WORKERS,
            allow_patterns="*official*/**",  # glob pattern
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen/CodeQwen1.5-7B-Chat IA3 fine-tuning")
    parser.add_argument("--experiment_id", type=int, default=None,
                        help="Experiment ID for tracking multiple runs")
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override the learning rate')
    parser.add_argument('--subset', type=float, default=1.0,
                       help='Fraction of training data to use (0 to 1)')
    args = parser.parse_args()

    main(args.experiment_id, args.learning_rate, args.subset)