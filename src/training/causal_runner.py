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
import time


def run(model_name, method, spec, run_info, subset=1.0, dataset_name=None,
        smoke=False, attn_implementation=None, seed=None, protocol="archived",
        protocol_overrides=None, gen_batch_size=1, eval_max_samples=None,
        peft_overrides=None, probe=False, memory_profile=None,
        recovery_steps=100, upload=False):
    import huggingface_hub
    import torch
    import wandb
    from tqdm import tqdm
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForLanguageModeling

    from src.evaluation.metrics import EMPTY_PLACEHOLDER, as_test_results, compute_generation_metrics
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
    from src.utils.custom_modified_api import LLMCustomTrainer, TimeMemoryCallback
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
    lifecycle = run_info.get("lifecycle_probe", False)
    if lifecycle:
        OUTPUT_MIN_LENGTH = OUTPUT_MAX_LENGTH  # worst output-length pilot, probe only
    if recover:
        latest_checkpoint = latest_recovery_checkpoint

    experiment_name = run_info["experiment_name"]
    experiment_results_dir = run_info["experiment_results_dir"]
    if peft_overrides:
        # pinned adapter kwargs for reruns (e.g. archived IA3 resolution);
        # applied to a copy, never to the frozen spec dict (D-026)
        spec = {**spec, "peft_config": {**spec["peft_config"],
                "kwargs": {**spec["peft_config"]["kwargs"], **peft_overrides}}}
    ctx = static_marker_context(run_info, spec)

    # ------------------------------------------------------------------
    # 1. House-keeping
    # ------------------------------------------------------------------
    if not smoke and not probe:
        if os.getenv("WANDB_MODE", "offline") == "online":
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        if upload:
            huggingface_hub.login(token=os.getenv("HUGGINGFACE_TOKEN"))
    os.makedirs(experiment_results_dir, exist_ok=True)
    os.makedirs(WANDB_DIR, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=experiment_name, dir=WANDB_DIR,
               group=os.getenv("WANDB_GROUP") or None,
               tags=[t for t in os.getenv("WANDB_TAGS", "").split(",") if t] or None)

    # ------------------------------------------------------------------
    # 2. Dataset & tokenizer
    # ------------------------------------------------------------------
    prepare = getattr(helper, spec["dataset_prep"])
    train_ds, val_ds, test_ds, tokenizer = prepare(
        spec["tokenizer_id"], subset_percentage=subset, dataset_name=dataset_name
    )
    if smoke or probe:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(2, len(val_ds))))
        test_ds = test_ds.select(range(min(eval_max_samples or 2, len(test_ds))))
    elif eval_max_samples is not None:
        val_ds = val_ds.select(range(min(eval_max_samples, len(val_ds))))
    if lifecycle:
        # Dataset preparation uses fixed padded training lengths; retain enough
        # microbatches to exercise two full accumulated optimizer updates.
        pilot_train_rows = max(8, 2 * (protocol_overrides or {}).get("gradient_accumulation_steps", 1))
        train_ds = train_ds.select(range(min(pilot_train_rows, len(train_ds))))
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

    if method in ("fmft", "galore"):
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

        if method in ("lora", "dora"):
            # the "lora_" requires_grad loop below also covers DoRA's
            # lora_magnitude_vector parameters
            lora_cfg = build_peft_config(spec, ctx)
            model = get_peft_model(base_model, lora_cfg)
            if ckpt_path:
                print(f"Resuming from checkpoint: {ckpt_path}")
                model.load_adapter(ckpt_path, adapter_name="default", is_trainable=True)
                model.set_adapter("default")
            for name, param in model.named_parameters():
                if "lora_" in name.lower() and param.requires_grad:
                    param.requires_grad_(True)

        elif method == "qlora":
            base_model = prepare_model_for_kbit_training(base_model)
            if ckpt_path:
                print(f"Resuming from checkpoint: {ckpt_path}")
                model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)
            else:
                model = get_peft_model(base_model, build_peft_config(spec, ctx))
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

    if memory_profile:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # 4-5. TrainingArguments + Trainer
    # ------------------------------------------------------------------
    overrides = dict(protocol_overrides or {})
    if seed is not None:
        overrides.update(seed=seed, data_seed=seed)
    if lifecycle:
        overrides.update(max_steps=2, eval_strategy="steps", eval_steps=1,
                         save_strategy="steps", save_steps=1, report_to="none",
                         load_best_model_at_end=True)
    if probe:
        # one real-batch-size step, nothing persisted (--probe, phase 07)
        overrides.update(max_steps=1, eval_strategy="no", save_strategy="no",
                         load_best_model_at_end=False, report_to="none")
    training_args = build_training_args(spec, ctx, smoke=smoke, overrides=overrides or None)
    if "galore" in str(training_args.optim).lower():
        try:
            import galore_torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "optim is a GaLore variant but galore-torch is not installed; "
                "pip install galore-torch (see requirements.txt)"
            ) from exc
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    generation_config = build_generation_config(spec, ctx)

    time_cb = TimeMemoryCallback()
    trainer_cls = recovery_trainer_class(LLMCustomTrainer) if recover else LLMCustomTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer_gen=tokenizer,
        generation_config=generation_config,
        debug=not smoke,
        gen_batch_size=gen_batch_size,
        callbacks=[time_cb],
    )

    if recover:
        trainer.recovery_model_revision = model_revision(
            experiment_results_dir, spec["model_id"], getattr(model.config, "_commit_hash", None),
            resumed=bool(ckpt_path))
        attach_recovery(trainer, recovery_steps, run_info.get("probe_stop_after_step"))
        stop_if_requested()

    resumed_from_step = 0
    if recover and ckpt_path:
        with open(os.path.join(ckpt_path, "trainer_state.json")) as state_file:
            resumed_from_step = json.load(state_file)["global_step"]
    train_result = (trainer.train(resume_from_checkpoint=ckpt_path) if ckpt_path
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
    # 6. Evaluation on test set with text generation
    # ------------------------------------------------------------------
    print("Starting test set evaluation with text generation...")
    model.eval()

    journal = (PredictionJournal(os.path.join(experiment_results_dir, "test_progress"),
                                 generation_identity(trainer, run_info["fingerprint"]), len(test_ds)) if recover else None)
    predictions = []
    references = []
    test_gen_start = time.time()

    if gen_batch_size > 1:
        # Batched path (D-028c): rows are already left-padded to a common
        # length by the tokenizer, so a batch is a plain stack; the prompt is
        # trimmed at the shared input length. The explicit pad mask matches
        # what the legacy per-sample loop got implicitly (generate() infers
        # input_ids != pad when pad != eos, which holds for the qwen models).
        pad_id = tokenizer.pad_token_id
        with torch.no_grad():
            for start in tqdm(range(0, len(test_ds), gen_batch_size),
                              desc=f"Evaluating test set (batched x{gen_batch_size})"):
                rows = test_ds[start:start + gen_batch_size]
                input_ids = torch.tensor(rows["input_ids"]).to(model.device)
                attention_mask = (input_ids != pad_id).long()
                input_length = input_ids.shape[1]

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=OUTPUT_MAX_LENGTH,
                        min_new_tokens=OUTPUT_MIN_LENGTH,
                        num_beams=NUM_BEAMS,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                for i in range(input_ids.shape[0]):
                    generated_text = tokenizer.decode(generated_ids[i][input_length:],
                                                      skip_special_tokens=True)
                    if len(generated_text.strip()) == 0:
                        generated_text = EMPTY_PLACEHOLDER
                    labels = [t for t in rows["labels"][i] if t != -100]
                    predictions.append(generated_text)
                    references.append(tokenizer.decode(labels, skip_special_tokens=True))

    else:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_ds, desc="Evaluating test set")):
                if journal:
                    stop_if_requested()
                    previous = journal.get(i)
                    if previous is not None:
                        predictions.append(previous["prediction"])
                        references.append(previous["reference"])
                        continue
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
                if journal:
                    journal.put(i, {"prediction": generated_text, "reference": reference_text})

                if i < 3:
                    print(f"Example {i+1}:")
                    print(f"  Generated: {generated_text}")
                    print(f"  Reference: {reference_text}")
                    print()

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

    # ------------------------------------------------------------------
    # 7. Save, log, push
    # ------------------------------------------------------------------
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
            memory_profile=memory_profile, lifecycle_probe=lifecycle,
            resumed_from_step=resumed_from_step, global_step=trainer.state.global_step,
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
        complete_results["expected_test_rows"] = len(test_ds)
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
