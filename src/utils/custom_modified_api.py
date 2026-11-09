import torch, time, wandb, logging, warnings
from transformers import TrainerCallback, Trainer, GenerationConfig
from tqdm import tqdm

from src.evaluation.metrics import (
    as_eval_results,
    compute_generation_metrics,
    guard_empty,
    preload_metrics,
)

_GIB = 1_073_741_824


class TimeMemoryCallback(TrainerCallback):
    """Step-time / eval-time / peak-VRAM tracking (phase 03).

    The previous version accumulated wall-clock between on_epoch_begin and
    on_epoch_end and relied on on_evaluate_begin/on_evaluate_end — hooks that
    do not exist on TrainerCallback and therefore never fired. Now training
    time is the sum of on_step_begin→on_step_end deltas per epoch, and eval
    wall-clock is on_epoch_end→on_evaluate (on_epoch_end precedes the epoch
    evaluation in transformers 4.47). Peaks are maxima over all CUDA devices
    and are never reset, so they describe the whole run.
    """

    def __init__(self):
        self.step_time_per_epoch = []
        self.eval_time_per_epoch = []
        self.peak_vram_allocated_gb = 0.0
        self.peak_vram_reserved_gb = 0.0
        self._step_start = None
        self._epoch_step_time = 0.0
        self._eval_anchor = None

    def _update_peaks(self):
        for device in range(torch.cuda.device_count()):
            self.peak_vram_allocated_gb = max(
                self.peak_vram_allocated_gb, torch.cuda.max_memory_allocated(device) / _GIB)
            self.peak_vram_reserved_gb = max(
                self.peak_vram_reserved_gb, torch.cuda.max_memory_reserved(device) / _GIB)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_step_time = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self._epoch_step_time += time.time() - self._step_start
            self._step_start = None

    def on_epoch_end(self, args, state, control, **kwargs):
        self._update_peaks()
        self.step_time_per_epoch.append(self._epoch_step_time)
        self._eval_anchor = time.time()
        wandb.log({
            "epoch": state.epoch,
            "train_time_h": self._epoch_step_time / 3600,
            "vram_used_GB": self.peak_vram_allocated_gb,
            "vram_reserved_GB": self.peak_vram_reserved_gb,
        }, step=state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        if self._eval_anchor is None:
            return
        elapsed = time.time() - self._eval_anchor
        self._eval_anchor = None
        self.eval_time_per_epoch.append(elapsed)
        self._update_peaks()
        wandb.log({"eval_time_m": elapsed / 60}, step=state.global_step)

    def summary(self):
        self._update_peaks()
        return {
            "train_step_time_s": self.step_time_per_epoch,
            "eval_time_s": self.eval_time_per_epoch,
            "peak_vram_allocated_gb": self.peak_vram_allocated_gb,
            "peak_vram_reserved_gb": self.peak_vram_reserved_gb,
        }




logger = logging.getLogger(__name__)

class LLMCustomTrainer(Trainer):
    """
    Custom Trainer for decoder-only LLMs that computes BLEU-4 and exact match.
    Handles ChatML format where input_ids contain the full conversation and 
    labels mask the parts we don't want to predict.
    """
    
    def __init__(self, tokenizer_gen, generation_config=None,
                 assistant_token="<|im_start|>assistant\n", debug=False,
                 gen_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer_gen = tokenizer_gen
        self.assistant_token = assistant_token
        self.debug = debug
        # 1 = legacy per-sample generation; >1 = left-padded batched generation
        # with identical generate kwargs (D-028c; bf16 batching may change rare
        # beam ties — one batch size per model, pinned in the manifest)
        self.gen_batch_size = gen_batch_size
        
        # Set up generation config
        if generation_config is None:
            # Create default generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=256,
                min_new_tokens=2,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )
        else:
            # Use provided generation config
            self.generation_config = generation_config.copy() if hasattr(generation_config, 'copy') else generation_config
        
        # Critical validations only
        if self.tokenizer_gen is None:
            raise ValueError("ERROR: tokenizer cannot be None for LLMCustomTrainer!")
        
        # Load metrics with error handling (fail fast at trainer construction)
        try:
            preload_metrics()
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to load evaluation metrics: {e}")
        
        if self.debug:
            logger.info(f"Debug mode enabled for LLMCustomTrainer")
            logger.info(f"Assistant token: '{assistant_token}'")
            logger.info(f"Generation config: {self.generation_config}")
    
    def evaluate(self, eval_dataset=None, **kwargs):
        """Override evaluate to add BLEU and exact match computation"""
        # Run normal evaluation first
        eval_result = super().evaluate(eval_dataset, **kwargs)
        
        # Add BLEU and exact match evaluation
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is not None:
            metrics = self._compute_generation_metrics(eval_dataset)
            log_metrics = {
                "eval/" + k.split('eval_')[-1]: v for k, v in metrics.items()
            }
            wandb.log(log_metrics, commit=False)
            eval_result.update(metrics)
        else:
            warnings.warn("WARNING: No eval_dataset available for sequence evaluation!")
        
        print("Total evaluation results:", eval_result)
        
        return eval_result
    
    def _extract_prompt_and_reference(self, input_ids, labels):
        """
        Extract the prompt (for generation) and reference (for evaluation) from input_ids and labels.
        """
        # Find where the assistant response should start
        try:
            assistant_token_ids = self.tokenizer_gen.encode(self.assistant_token, add_special_tokens=False)
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to encode assistant_token '{self.assistant_token}': {e}")
        
        # Convert to list for easier searching
        input_list = input_ids.tolist()
        
        # Find assistant token position
        assistant_pos = None
        if len(assistant_token_ids) > 0:
            for i in range(len(input_list) - len(assistant_token_ids) + 1):
                if input_list[i:i+len(assistant_token_ids)] == assistant_token_ids:
                    assistant_pos = i + len(assistant_token_ids)
                    break
        
        if assistant_pos is not None:
            # Prompt is everything up to and including the assistant token
            prompt_ids = input_ids[:assistant_pos]
            
            # Reference is the non-masked part of labels after assistant token
            ref_labels = labels[assistant_pos:]
            ref_labels = ref_labels[ref_labels != -100]
            
            if len(ref_labels) == 0:
                if self.debug:
                    warnings.warn("WARNING: No non-masked labels found after assistant token!")
                reference_text = ""
            else:
                reference_text = self.tokenizer_gen.decode(ref_labels, skip_special_tokens=True).strip()
                
        else:
            # Critical error - assistant token not found
            raise RuntimeError(f"ERROR: Assistant token '{self.assistant_token}' not found in input! Check your data format.")
        
        return prompt_ids, reference_text
    
    def _generate_batched(self, prompt_ids_list):
        """Generate for a chunk of prompts: left-pad to the chunk max length,
        run one generate call with the same config, trim the shared prompt
        length per row. Masking `!= pad` replicates the per-sample path
        exactly: it passes no mask, and generate() infers `input_ids.ne(pad)`
        because pad != eos for the qwen tokenizers — so each row's own left
        padding is masked out there too."""
        pad_id = self.tokenizer_gen.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer_gen.eos_token_id
        max_len = max(p.shape[0] for p in prompt_ids_list)
        # The Trainer's eval dataloader has already moved prompts to the GPU; pad on the
        # same device.
        batch = torch.stack([
            torch.cat([torch.full((max_len - p.shape[0],), pad_id, dtype=p.dtype, device=p.device), p])
            for p in prompt_ids_list
        ]).to(self.args.device)
        attention_mask = (batch != pad_id).long()
        generated = self.model.generate(
            input_ids=batch,
            attention_mask=attention_mask,
            **self.generation_config.to_dict()
        )
        texts = []
        for row in generated[:, max_len:]:
            texts.append(self.tokenizer_gen.decode(row, skip_special_tokens=True).strip())
        return texts

    def _compute_generation_metrics(self, dataset):
        """Compute BLEU-4 and exact match by generating text"""
        if self.debug:
            logger.info(f"Computing BLEU metrics for {len(dataset)} samples...")

        self.model.eval()
        predictions = []
        references = []
        failed_samples = 0

        if self.gen_batch_size > 1:
            return self._compute_generation_metrics_batched(dataset)

        eval_dataloader = self.get_eval_dataloader(dataset)
        
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Evaluating test set"):
                labels = batch.get("labels")
                input_ids = batch["input_ids"]
                
                if labels is None:
                    raise RuntimeError(f"ERROR: No 'labels' key found in batch! Check your data collator.")
                
                batch_size = input_ids.shape[0]
                
                for i in range(batch_size):
                    try:
                        sample_input_ids = input_ids[i]
                        sample_labels = labels[i]
                        
                        # Extract prompt and reference
                        prompt_ids, reference_text = self._extract_prompt_and_reference(
                            sample_input_ids, sample_labels
                        )
                        
                        # Generate response
                        prompt_tensor = prompt_ids.unsqueeze(0).to(self.args.device)
                        
                        # Use generation config for all parameters
                        generated_ids = self.model.generate(
                            input_ids=prompt_tensor,
                            **self.generation_config.to_dict()
                        )
                        
                        # Extract only the newly generated tokens
                        if generated_ids.shape[1] > prompt_tensor.shape[1]:
                            new_tokens = generated_ids[:, prompt_tensor.shape[1]:]
                            pred_text = self.tokenizer_gen.decode(new_tokens[0], skip_special_tokens=True).strip()
                        else:
                            pred_text = ""
                        
                        # Clean up
                        pred_text = guard_empty(pred_text)
                        
                        if self.debug:
                            print(f"Input text: '{self.tokenizer_gen.decode(prompt_ids, skip_special_tokens=True)}'")
                            print(f"Prediction text: '{pred_text}'")
                            print(f"Reference text: '{reference_text}'")
                            print()
                            
                        
                        predictions.append(pred_text)
                        references.append(reference_text)
                        
                    except Exception as e:
                        failed_samples += 1
                        if self.debug:
                            logger.warning(f"WARNING: Failed sample {batch_idx}-{i}: {e}")
                        predictions.append("<FAILED>")
                        references.append("")
        
        # Critical error if too many failures
        total_samples = len(predictions)
        if failed_samples > total_samples * 0.5:  # More than 50% failed
            raise RuntimeError(f"CRITICAL: {failed_samples}/{total_samples} samples failed! Check your data format.")
        
        # Compute metrics (single shared implementation, frozen semantics)
        try:
            metric_values = compute_generation_metrics(predictions, references)
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to compute metrics: {e}")

        logger.info(f"BLEU-4: {metric_values['bleu_4']:.4f}, Exact Match: {metric_values['exact_match']:.4f}")
        logger.info(f"ROUGE-L: {metric_values['rouge']['rougeL']:.4f}, BERTScore-F1: {metric_values['bertscore']['f1']:.4f}, METEOR: {metric_values['meteor']:.4f}")

        return as_eval_results(metric_values)

    def _compute_generation_metrics_batched(self, dataset):
        """Batched variant of the generation loop (gen_batch_size > 1): same
        prompt extraction and metrics, generation in left-padded chunks. A row
        whose assistant token was truncated away scores as "<FAILED>" against an
        empty reference, as in the per-sample loop, which also aborts above 50%."""
        prompts, references, failed = [], [], set()
        eval_dataloader = self.get_eval_dataloader(dataset)
        for batch in eval_dataloader:
            labels = batch.get("labels")
            if labels is None:
                raise RuntimeError("ERROR: No 'labels' key found in batch! Check your data collator.")
            for i in range(batch["input_ids"].shape[0]):
                try:
                    prompt_ids, reference_text = self._extract_prompt_and_reference(
                        batch["input_ids"][i], labels[i]
                    )
                except RuntimeError as e:
                    if self.debug:
                        logger.warning(f"WARNING: Failed sample {len(references)}: {e}")
                    failed.add(len(references))
                    prompt_ids, reference_text = None, ""
                prompts.append(prompt_ids)
                references.append(reference_text)
        if len(failed) > len(references) * 0.5:
            raise RuntimeError(f"CRITICAL: {len(failed)}/{len(references)} samples failed! Check your data format.")

        valid = [p for p in prompts if p is not None]
        generated = []
        with torch.no_grad():
            for start in tqdm(range(0, len(valid), self.gen_batch_size),
                              desc=f"Evaluating (batched x{self.gen_batch_size})"):
                chunk = valid[start:start + self.gen_batch_size]
                generated.extend(guard_empty(t) for t in self._generate_batched(chunk))
        generated = iter(generated)
        predictions = ["<FAILED>" if i in failed else next(generated) for i in range(len(prompts))]

        metric_values = compute_generation_metrics(predictions, references)
        logger.info(f"BLEU-4: {metric_values['bleu_4']:.4f}, Exact Match: {metric_values['exact_match']:.4f}")
        return as_eval_results(metric_values)
