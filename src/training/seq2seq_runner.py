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
import time


def run(model_name, method, spec, run_info, subset=1.0, dataset_name=None,
        smoke=False, attn_implementation=None, seed=None, protocol="archived",
        protocol_overrides=None, gen_batch_size=1, eval_max_samples=None,
        peft_overrides=None, probe=False, memory_profile=None,
        recovery_steps=100, upload=False):
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
        DATASET_NAME,
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
    from .resources import collect_resources
    from .peft_setup import (
        add_tokenizer_markers,
        build_generation_config,
        build_peft_config,
        build_training_args,
        latest_checkpoint,
        load_base_model,
        static_marker_context,
    )

    from src.evaluation.metrics import configure_metric_resources
    configure_metric_resources(device="cpu" if memory_profile else None,
                               batch_size=1 if memory_profile else None)

    from .recovery import (latest_recovery_checkpoint, recovery_trainer_class,
                           attach_recovery, PredictionJournal, stop_if_requested, atomic_json, generation_identity, model_revision)
    recover = run_info.get("recovery", False)
    if recover:
        latest_checkpoint = latest_recovery_checkpoint

    experiment_name = run_info["experiment_name"]
    experiment_results_dir = run_info["experiment_results_dir"]
    if peft_overrides:
        # pinned adapter kwargs for reruns (e.g. archived codet5-IA3
        # resolution, KI-015); applied to a copy, never to the frozen spec
        spec = {**spec, "peft_config": {**spec["peft_config"],
                "kwargs": {**spec["peft_config"]["kwargs"], **peft_overrides}}}
    ctx = static_marker_context(run_info, spec)

    # [PREPARING DATASET AND FUNCTIONS]
    if not smoke and not probe:
        if os.getenv("WANDB_MODE", "offline") == "online":
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        if upload:
            huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    os.makedirs(WANDB_DIR, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=experiment_name, dir=WANDB_DIR,
               group=os.getenv("WANDB_GROUP") or None,
               tags=[t for t in os.getenv("WANDB_TAGS", "").split(",") if t] or None)

    prepare = getattr(helper, spec["dataset_prep"])
    train_dataset, val_dataset, test_dataset, tokenizer = prepare(
        spec["tokenizer_id"], subset_percentage=subset, dataset_name=dataset_name
    )
    if smoke or probe:
        train_dataset = train_dataset.select(range(min(8, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(2, len(val_dataset))))
        test_dataset = test_dataset.select(range(min(eval_max_samples or 2, len(test_dataset))))
    elif eval_max_samples is not None:
        val_dataset = val_dataset.select(range(min(eval_max_samples, len(val_dataset))))
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
    elif method in ("lora", "dora"):
        base_model = load()
        base_model.generation_config.num_beams = NUM_BEAMS
        base_model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH
        if checkpoint:
            model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)
            print(f"Resumed from checkpoint: {checkpoint}")
        else:
            model = get_peft_model(base_model, build_peft_config(spec, ctx))
            print("Created new LoRA model")
    elif method == "qlora":
        base_model = load()
        base_model.generation_config.num_beams = NUM_BEAMS
        base_model.generation_config.max_new_tokens = OUTPUT_MAX_LENGTH
        base_model = prepare_model_for_kbit_training(base_model)
        if checkpoint:
            model = PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)
            print(f"Resumed from checkpoint: {checkpoint}")
        else:
            model = get_peft_model(base_model, build_peft_config(spec, ctx))
        if spec["grad_checkpoint_enable"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    elif method == "ia3":
        base_model = load()
        model = (PeftModel.from_pretrained(base_model, checkpoint, is_trainable=True)
                 if checkpoint else get_peft_model(base_model, build_peft_config(spec, ctx)))
    elif method == "ptuning":
        # Preserve this experimental legacy path; not admitted by the A5000 profile.
        model = load(checkpoint) if checkpoint else load()
        ctx["<model.config.hidden_size>"] = model.config.hidden_size
        model = get_peft_model(model, build_peft_config(spec, ctx))
    else:
        raise ValueError(f"unknown seq2seq method: {method}")

    if memory_profile:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"trainable_params": trainable_params}, allow_val_change=True)
    print(f"Trainable parameters: {trainable_params/1e6:.2f} M")

    # Training arguments (Seq2SeqTrainingArguments with nested generation_config)
    ctx["<GENERATION_CONFIG>"] = _nested_generation_config(spec, ctx, model, tokenizer)
    overrides = dict(protocol_overrides or {})
    if seed is not None:
        overrides.update(seed=seed, data_seed=seed)
    if probe:
        # one real-batch-size step, nothing persisted (--probe, phase 07)
        overrides.update(max_steps=1, eval_strategy="no", save_strategy="no",
                         load_best_model_at_end=False, report_to="none")
    training_args = build_training_args(spec, ctx, smoke=smoke, overrides=overrides or None)

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

    time_cb = TimeMemoryCallback()
    trainer_cls = recovery_trainer_class(Seq2SeqTrainer) if recover else Seq2SeqTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[time_cb],
    )

    if recover:
        trainer.recovery_model_revision = model_revision(
            experiment_results_dir, spec["model_id"], getattr(model.config, "_commit_hash", None),
            resumed=bool(checkpoint))
        attach_recovery(trainer, recovery_steps)
        stop_if_requested()

    train_result = (trainer.train(resume_from_checkpoint=checkpoint) if checkpoint
                    else trainer.train())

    if probe:
        summary = time_cb.summary()
        print("PROBE RESULT: " + json.dumps({
            "peak_vram_allocated_gb": summary["peak_vram_allocated_gb"],
            "peak_vram_reserved_gb": summary["peak_vram_reserved_gb"],
            "trainable_params": trainable_params,
            "total_params": total_params,
            "train_runtime_s": train_result.metrics.get("train_runtime"),
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "optim": str(training_args.optim),
        }))
        return {}

    # ------------------------------------------------------------------
    # Manual test set evaluation with text generation
    # ------------------------------------------------------------------
    print("Starting test set evaluation with text generation...")
    model.eval()

    journal = (PredictionJournal(os.path.join(experiment_results_dir, "test_progress"),
                                 generation_identity(trainer, run_info["fingerprint"]), len(test_dataset)) if recover else None)
    predictions = []
    references = []

    test_loader = DataLoader(
        test_dataset,
        batch_size=gen_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    decoder_start = (
        tokenizer.bos_token_id
        if spec["test_decoder_start"] == "tokenizer.bos"
        else model.config.decoder_start_token_id
    )

    test_gen_start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating test set")):
            if journal:
                stop_if_requested()
                previous = journal.get(i)
                if previous is not None:
                    predictions.append(previous["prediction"])
                    references.append(previous["reference"])
                    continue
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

            if gen_batch_size == 1:
                # Legacy path kept verbatim: a single-sample batch is unpadded,
                # so labels decode directly
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                if len(generated_text.strip()) == 0:
                    print(f"WARNING: Empty generation at sample {i}")
                    generated_text = EMPTY_PLACEHOLDER

                reference_text = tokenizer.decode(labels[0], skip_special_tokens=True)

                predictions.append(generated_text)
                references.append(reference_text)
                if journal:
                    journal.put(i, {"prediction": generated_text, "reference": reference_text})

                if i < 3:
                    print(f"Example {i+1}:")
                    print(f"  Generated: {generated_text}")
                    print(f"  Reference: {reference_text}")
                    print()
            else:
                # Batched path (D-028c): the collator pads labels with -100,
                # which must be dropped before decoding
                for j in range(generated_ids.shape[0]):
                    generated_text = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                    if len(generated_text.strip()) == 0:
                        generated_text = EMPTY_PLACEHOLDER
                    ref_ids = [t for t in labels[j].tolist() if t != -100]
                    predictions.append(generated_text)
                    references.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

    test_generation_time_s = time.time() - test_gen_start
    if journal:
        journal.values()  # Require full coverage before corpus metrics/final publication.
        stop_if_requested()
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

    complete_results["resources"] = collect_resources(
        train_result, time_cb.summary(),
        test_generation_time_s=test_generation_time_s,
        trainable_params=trainable_params, total_params=total_params,
        hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        experiment_name=experiment_name, dataset=dataset_name or DATASET_NAME,
        seed=seed if seed is not None else 42,
        protocol=protocol, protocol_overrides=protocol_overrides,
        eval_max_samples=eval_max_samples, gen_batch_size=gen_batch_size,
        results_dir=experiment_results_dir,
    )

    if memory_profile:
        complete_results["resources"].update(
            memory_profile=memory_profile,
            metric_device="cpu", metric_batch_size=1,
            train_microbatch=training_args.per_device_train_batch_size,
            eval_microbatch=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
            attn_implementation=attn_implementation,
        )

    with open(f"{experiment_results_dir}/training_args.txt", "w") as f:
        f.write(str(training_args))
    if recover:
        complete_results["resources"]["model_revision"] = trainer.recovery_model_revision
        complete_results["run_fingerprint"] = run_info["fingerprint"]
        complete_results["expected_test_rows"] = len(test_dataset)
        stop_if_requested()
        atomic_json(os.path.join(experiment_results_dir, "complete_results.json"), complete_results)
    else:
        with open(experiment_results_dir + "/complete_results.json", "w") as f:
            json.dump(complete_results, f, indent=2)

    if upload and "official" in experiment_name and not smoke:
        # legacy pairing: upload the spec's results root; a --results-root run
        # lives elsewhere, so upload that root instead (KI-012 follow-up)
        upload_folder = spec["results_dir"]
        if not experiment_results_dir.startswith(spec["results_dir"]):
            upload_folder = os.path.dirname(experiment_results_dir) or "."
        if is_placeholder_repo(HUGGINGFACE_REPO_ID):
            print(f"WARNING: HUGGINGFACE_REPO_ID is a placeholder ({HUGGINGFACE_REPO_ID}); skipping upload.")
        else:
            api = huggingface_hub.HfApi()
            api.upload_large_folder(
                folder_path=upload_folder,
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
