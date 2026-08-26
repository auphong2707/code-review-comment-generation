"""Training pipeline for the seq2seq family (codet5, plbart), reproducing the
legacy per-script behavior from the spec table.

Accepted deviations (cosmetic / bug fixes, logged in
plans/refactor-source/equivalence-report.md):
- wandb trainable-params key unified to "trainable_params";
- only the first 3 test examples (and no raw generated ids) are printed;
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
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

    from src.evaluation.metrics import EMPTY_PLACEHOLDER, as_eval_results, as_test_results, compute_generation_metrics
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
    from src.utils.custom_modified_api import TimeMemoryCallback
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

    # [PREPARING DATASET AND FUNCTIONS]
    if not smoke:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    os.makedirs(WANDB_DIR, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=experiment_name, dir=WANDB_DIR)

    prepare = getattr(helper, spec["dataset_prep"])
    train_dataset, val_dataset, test_dataset, tokenizer = prepare(
        spec["tokenizer_id"], subset_percentage=subset, dataset_name=dataset_name
    )
    if smoke:
        train_dataset = train_dataset.select(range(min(8, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(2, len(val_dataset))))
        test_dataset = test_dataset.select(range(min(2, len(test_dataset))))
    add_tokenizer_markers(ctx, tokenizer)

    os.makedirs(experiment_results_dir, exist_ok=True)
    checkpoint = latest_checkpoint(experiment_results_dir)

    # Model assembly (per-method, mirroring the legacy scripts)
    def load(model_id_override=None):
        s = spec if model_id_override is None else {**spec, "model_id": model_id_override}
        return load_base_model(s, ctx, attn_implementation=attn_implementation)

    if method == "fmft":
        model = load(checkpoint) if checkpoint else load()
        if checkpoint:
            print(f"Resumed from checkpoint: {checkpoint}")
    elif method == "lora":
        base_model = load()
        base_model.generation_config.num_beams = NUM_BEAMS
        base_model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH
        if checkpoint:
            model = PeftModel.from_pretrained(base_model, checkpoint)
            print(f"Resumed from checkpoint: {checkpoint}")
        else:
            model = get_peft_model(base_model, build_peft_config(spec, ctx))
            print("Created new LoRA model")
    elif method == "qlora":
        base_model = load()
        base_model.generation_config.num_beams = NUM_BEAMS
        base_model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH
        if model_name == "plbart":
            # legacy plbart_qlora runs kbit prep before the resume branch
            base_model = prepare_model_for_kbit_training(base_model)
        if checkpoint:
            model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)
            print(f"Resumed from checkpoint: {checkpoint}")
        else:
            prepared = base_model if model_name == "plbart" else prepare_model_for_kbit_training(base_model)
            model = get_peft_model(prepared, build_peft_config(spec, ctx))
        if spec["grad_checkpoint_enable"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    elif method in ("ia3", "ptuning"):
        # legacy ia3/ptuning seq2seq scripts reload the base from the
        # checkpoint dir when resuming, then wrap a fresh PEFT config
        model = load(checkpoint) if checkpoint else load()
        if method == "ptuning":
            ctx["<model.config.hidden_size>"] = model.config.hidden_size
        model = get_peft_model(model, build_peft_config(spec, ctx))
    else:
        raise ValueError(f"unknown seq2seq method: {method}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update({"trainable_params": trainable_params}, allow_val_change=True)
    print(f"Trainable parameters: {trainable_params/1e6:.2f} M")

    # Training arguments (Seq2SeqTrainingArguments with nested generation_config)
    ctx["<GENERATION_CONFIG>"] = _nested_generation_config(spec, ctx, model, tokenizer)
    training_args = build_training_args(spec, ctx, smoke=smoke)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = [[p for p in pred_seq if p != -100] for pred_seq in preds]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = [[l for l in label_seq if l != -100] for label_seq in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print("Result of first example in the eval batch:")
        print(f"Decoded Predictions Sample: {decoded_preds[0]}")
        print(f"Decoded Labels Sample: {decoded_labels[0]}")

        return as_eval_results(compute_generation_metrics(decoded_preds, decoded_labels))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[TimeMemoryCallback()],
    )

    trainer.train(resume_from_checkpoint=checkpoint) if checkpoint else trainer.train()

    # ------------------------------------------------------------------
    # Manual test set evaluation with text generation
    # ------------------------------------------------------------------
    print("Starting test set evaluation with text generation...")
    model.eval()

    predictions = []
    references = []

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
    )

    decoder_start = (
        tokenizer.bos_token_id
        if spec["test_decoder_start"] == "tokenizer.bos"
        else model.config.decoder_start_token_id
    )

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating test set")):
            labels = batch["labels"]
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.autocast("cuda"):
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=OUTPUT_MAX_LENGTH,
                    min_new_tokens=OUTPUT_MIN_LENGTH,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=decoder_start,
                )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            if len(generated_text.strip()) == 0:
                print(f"WARNING: Empty generation at sample {i}")
                generated_text = EMPTY_PLACEHOLDER

            reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

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


def _nested_generation_config(spec, ctx, model, tokenizer):
    """The seq2seq TrainingArguments embed a GenerationConfig whose
    decoder_start_token_id source differs per model (codet5: model config,
    plbart: tokenizer.bos)."""
    from .peft_setup import build_generation_config

    ctx["<decoder_start>"] = (
        tokenizer.bos_token_id
        if spec["test_decoder_start"] == "tokenizer.bos"
        else model.config.decoder_start_token_id
    )
    return build_generation_config(spec, ctx)
