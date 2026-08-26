"""Training pipeline for the causal family (qwen15, qwen25, qwen25-0.5b),
reproducing the legacy per-script behavior from the spec table.

Accepted deviations from the legacy scripts (cosmetic / bug fixes, logged in
plans/refactor-source/equivalence-report.md):
- p-tuning test eval now trims the prompt from generations (legacy kept it);
- wandb trainable-params key unified to "trainable_params";
- only the first 3 test examples are printed;
- torch.autocast replaces deprecated torch.cuda.amp.autocast;
- HF upload is skipped for placeholder repo ids instead of failing.
"""

import json
import os


def run(model_name, method, spec, run_info, subset=1.0, dataset_name=None,
        smoke=False, attn_implementation=None):
    import huggingface_hub
    import torch
    import wandb
    from tqdm import tqdm
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForLanguageModeling

    from src.evaluation.metrics import EMPTY_PLACEHOLDER, as_test_results, compute_generation_metrics
    from src.utils import helper
    from src.utils.constants import (
        HUGGINGFACE_REPO_ID,
        NUM_BEAMS,
        NUM_WORKERS,
        OUTPUT_MAX_LENGTH,
        OUTPUT_MIN_LENGTH,
        WANDB_PROJECT,
        WANDB_DIR,
        is_placeholder_repo,
    )
    from src.utils.custom_modified_api import LLMCustomTrainer, TimeMemoryCallback
    from .peft_setup import (
        add_tokenizer_markers,
        build_generation_config,
        build_peft_config,
        build_training_args,
        latest_checkpoint,
        load_base_model,
        static_marker_context,
    )

    experiment_name = run_info["experiment_name"]
    experiment_results_dir = run_info["experiment_results_dir"]
    ctx = static_marker_context(run_info, spec)

    # ------------------------------------------------------------------
    # 1. House-keeping
    # ------------------------------------------------------------------
    if not smoke:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    os.makedirs(experiment_results_dir, exist_ok=True)
    os.makedirs(WANDB_DIR, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=experiment_name, dir=WANDB_DIR)

    # ------------------------------------------------------------------
    # 2. Dataset & tokenizer
    # ------------------------------------------------------------------
    prepare = getattr(helper, spec["dataset_prep"])
    train_ds, val_ds, test_ds, tokenizer = prepare(
        spec["tokenizer_id"], subset_percentage=subset, dataset_name=dataset_name
    )
    if smoke:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(2, len(val_ds))))
        test_ds = test_ds.select(range(min(2, len(test_ds))))
    add_tokenizer_markers(ctx, tokenizer)

    # ------------------------------------------------------------------
    # 3. Model (resume if checkpoint exists)
    # ------------------------------------------------------------------
    ckpt_path = latest_checkpoint(experiment_results_dir)

    def set_generation_defaults(m):
        m.generation_config.num_beams = NUM_BEAMS
        m.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH

    if method == "fmft":
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
            model = load_base_model({**spec, "model_id": ckpt_path}, ctx,
                                    attn_implementation=attn_implementation)
        else:
            model = load_base_model(spec, ctx, attn_implementation=attn_implementation)
            set_generation_defaults(model)
        model.resize_token_embeddings(len(tokenizer))
        if spec["grad_checkpoint_enable"]:
            model.gradient_checkpointing_enable()
    else:
        base_model = load_base_model(spec, ctx, attn_implementation=attn_implementation)
        set_generation_defaults(base_model)
        if method != "qlora":
            base_model.resize_token_embeddings(len(tokenizer))

        if method == "lora":
            lora_cfg = build_peft_config(spec, ctx)
            model = get_peft_model(base_model, lora_cfg)
            if ckpt_path:
                print(f"Resuming from checkpoint: {ckpt_path}")
                try:
                    model.load_adapter(ckpt_path, adapter_name="default")
                    model.set_adapter("default")
                    print("Successfully loaded LoRA adapter from checkpoint")
                except Exception as e:
                    print(f"Warning: Could not load adapter from checkpoint: {e}")
                    print("Continuing with fresh LoRA model...")
            for name, param in model.named_parameters():
                if "lora_" in name.lower() and param.requires_grad:
                    param.requires_grad_(True)

        elif method == "qlora":
            if ckpt_path:
                print(f"Resuming from checkpoint: {ckpt_path}")
                model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)
            else:
                model = prepare_model_for_kbit_training(base_model)
                model = get_peft_model(model, build_peft_config(spec, ctx))
            if spec["grad_checkpoint_enable"]:
                model.gradient_checkpointing_enable()

        elif method == "ia3":
            # Legacy IA3 scripts disable gradient checkpointing / re-enable the
            # cache on both the base and wrapped model, and re-init zeroed IA3
            # weights to 1.0
            if hasattr(base_model, "gradient_checkpointing_disable"):
                base_model.gradient_checkpointing_disable()
            if hasattr(base_model, "config"):
                base_model.config.use_cache = True
            model = get_peft_model(base_model, build_peft_config(spec, ctx))
            model.train()
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model, "config"):
                model.config.use_cache = True
            if hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_disable"):
                model.base_model.gradient_checkpointing_disable()
            if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
                model.base_model.config.use_cache = True
            for name, param in model.named_parameters():
                if "ia3" in name.lower():
                    param.requires_grad_(True)
                    with torch.no_grad():
                        if torch.all(param.data == 0.0):
                            param.data.fill_(1.0)

        elif method == "ptuning":
            if ckpt_path:
                print(f"Resuming from checkpoint: {ckpt_path}")
                model = PeftModel.from_pretrained(base_model, ckpt_path)
            else:
                ctx["<model.config.hidden_size>"] = base_model.config.hidden_size
                model = get_peft_model(base_model, build_peft_config(spec, ctx))
        else:
            raise ValueError(f"unknown causal method: {method}")

        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_count == 0:
            raise ValueError("No trainable parameters found! This will cause the gradient error.")
        wandb.config.update({"trainable_params": trainable_count}, allow_val_change=True)
        print(f"Trainable parameters: {trainable_count/1e6:.2f} M")

    # ------------------------------------------------------------------
    # 4-5. TrainingArguments + Trainer
    # ------------------------------------------------------------------
    training_args = build_training_args(spec, ctx, smoke=smoke)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    generation_config = build_generation_config(spec, ctx)

    trainer = LLMCustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer_gen=tokenizer,
        generation_config=generation_config,
        debug=not smoke,
        callbacks=[TimeMemoryCallback()],
    )

    trainer.train(resume_from_checkpoint=ckpt_path) if ckpt_path else trainer.train()

    # ------------------------------------------------------------------
    # 6. Evaluation on test set with text generation
    # ------------------------------------------------------------------
    print("Starting test set evaluation with text generation...")
    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_ds, desc="Evaluating test set")):
            input_ids = torch.tensor([batch["input_ids"]]).to(model.device)
            input_length = len(batch["input_ids"])

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=OUTPUT_MAX_LENGTH,
                    min_new_tokens=OUTPUT_MIN_LENGTH,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = generated_ids[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if len(generated_text.strip()) == 0:
                print(f"WARNING: Empty generation at sample {i}")
                generated_text = EMPTY_PLACEHOLDER

            if "labels" in batch:
                labels = [t for t in batch["labels"] if t != -100]
                reference_text = tokenizer.decode(labels, skip_special_tokens=True)
            else:
                reference_text = tokenizer.decode(batch["input_ids"][input_length:], skip_special_tokens=True)

            predictions.append(generated_text)
            references.append(reference_text)

            if i < 3:
                print(f"Example {i+1}:")
                print(f"  Generated: {generated_text}")
                print(f"  Reference: {reference_text}")
                print()

    metrics = compute_generation_metrics(predictions, references)

    print(f"Test Results:")
    print(f"  BLEU-4: {metrics['bleu_4']}")
    print(f"  Exact Match: {metrics['exact_match']}")
    print(f"  ROUGE: {metrics['rouge']}")
    print(f"  BERTScore: {metrics['bertscore']}")
    print(f"  METEOR: {metrics['meteor']}")

    raw_data = [{"prediction": pred, "reference": ref} for pred, ref in zip(predictions, references)]
    complete_results = {**as_test_results(metrics), "raw_data": raw_data}

    wandb.log(as_test_results(metrics))

    # ------------------------------------------------------------------
    # 7. Save, log, push
    # ------------------------------------------------------------------
    model.save_pretrained(experiment_results_dir)
    tokenizer.save_pretrained(experiment_results_dir)
    with open(f"{experiment_results_dir}/training_args.txt", "w") as f:
        f.write(str(training_args))
    with open(experiment_results_dir + "/complete_results.json", "w") as f:
        json.dump(complete_results, f, indent=2)

    if "official" in experiment_name and not smoke:
        if is_placeholder_repo(HUGGINGFACE_REPO_ID):
            print(f"WARNING: HUGGINGFACE_REPO_ID is a placeholder ({HUGGINGFACE_REPO_ID}); skipping upload.")
        else:
            api = huggingface_hub.HfApi()
            api.upload_large_folder(
                folder_path=spec["results_dir"],
                repo_id=HUGGINGFACE_REPO_ID,
                repo_type="model",
                private=False,
                num_workers=NUM_WORKERS,
                allow_patterns="*official*/**",
            )

    return complete_results
