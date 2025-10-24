import torch, time, wandb, evaluate, logging, warnings
from transformers import TrainerCallback, Trainer, GenerationConfig
from tqdm import tqdm

class TimeMemoryCallback(TrainerCallback):
    def __init__(self):
        self.train_time_accumulated = 0.0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.train_time_accumulated = 0.0

    def on_epoch_end(self, args, state, control, **kwargs):
        # Add any remaining training time (if no evaluation happened at the end)
        if hasattr(self, 'start_time'):
            self.train_time_accumulated += time.time() - self.start_time
            
        vram_gb = torch.cuda.max_memory_allocated() / 1_073_741_824
        wandb.log({
            "epoch": state.epoch,
            "train_time_h": self.train_time_accumulated / 3600,
            "vram_used_GB": vram_gb
        }, step=state.global_step)

    def on_evaluate_begin(self, args, state, control, **kwargs):
        # Pause training timer and accumulate training time so far
        if hasattr(self, 'start_time'):
            self.train_time_accumulated += time.time() - self.start_time
            
        torch.cuda.reset_peak_memory_stats()
        self.eval_start_time = time.time()

    def on_evaluate_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.eval_start_time     # s
        vram_gb = torch.cuda.max_memory_allocated() / 1_073_741_824
        wandb.log({
            "eval_time_m": elapsed / 60,
            "eval_vram_used_GB": vram_gb
        }, step=state.global_step)
        
        # Resume training timer
        self.start_time = time.time()




logger = logging.getLogger(__name__)

class LLMCustomTrainer(Trainer):
    """
    Custom Trainer for decoder-only LLMs that computes BLEU-4 and exact match.
    Handles ChatML format where input_ids contain the full conversation and 
    labels mask the parts we don't want to predict.
    """
    
    def __init__(self, tokenizer_gen, generation_config=None, 
                 assistant_token="<|im_start|>assistant\n", debug=False, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer_gen = tokenizer_gen
        self.assistant_token = assistant_token
        self.debug = debug
        
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
        
        # Load metrics with error handling
        try:
            self.bleu_metric = evaluate.load("bleu")
            self.exact_match_metric = evaluate.load("exact_match")
            self.rouge_metric = evaluate.load("rouge")
            self.bert_score_metric = evaluate.load("bertscore")
            self.meteor_metric = evaluate.load("meteor")
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
    
    def _compute_generation_metrics(self, dataset):
        """Compute BLEU-4 and exact match by generating text"""
        if self.debug:
            logger.info(f"Computing BLEU metrics for {len(dataset)} samples...")
            
        self.model.eval()
        predictions = []
        references = []
        failed_samples = 0
        
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
                        pred_text = pred_text if pred_text else "<EMPTY>"
                        
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
        
        # Compute metrics
        try:
            bleu_score = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )["bleu"]
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to compute BLEU: {e}")
        
        try:
            exact_match_score = self.exact_match_metric.compute(
                predictions=predictions,
                references=references
            )["exact_match"]
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to compute exact match: {e}")
        
        # Compute ROUGE score (multiple scores)
        try:
            rouge_score = self.rouge_metric.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to compute ROUGE: {e}")
        
        # Compute BERTScore (multiple scores)
        try:
            bertscore = self.bert_score_metric.compute(
                predictions=predictions,
                references=references,
                lang="en"
            )
            bertscore = {
                "precision": sum(bertscore["precision"]) / len(bertscore["precision"]),
                "recall": sum(bertscore["recall"]) / len(bertscore["recall"]),
                "f1": sum(bertscore["f1"]) / len(bertscore["f1"])
            }
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to compute BERTScore: {e}")
        
        # Compute METEOR score
        try:
            meteor_score = self.meteor_metric.compute(
                predictions=predictions,
                references=references
            )["meteor"]
        except Exception as e:
            raise RuntimeError(f"ERROR: Failed to compute METEOR: {e}")
        
        logger.info(f"BLEU-4: {bleu_score:.4f}, Exact Match: {exact_match_score:.4f}")
        logger.info(f"ROUGE-L: {rouge_score['rougeL']:.4f}, BERTScore-F1: {bertscore['f1']:.4f}, METEOR: {meteor_score:.4f}")
        
        return {
            "eval_bleu_4": bleu_score,
            "eval_exact_match": exact_match_score,
            "eval_rouge": rouge_score,
            "eval_bertscore": bertscore,
            "eval_meteor": meteor_score
        }