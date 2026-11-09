# ICL Inference for Code Review Generation

This directory contains scripts for running In-Context Learning (ICL) inference on code review generation models.

## Files

- `src/icl_inference.py` - Main ICL inference script

## Quick Start

Run ICL inference for a specific model:

```bash
# Basic usage - 3 examples with random selection
python -m src.icl_inference --model_name codet5 --num_examples 3

# Zero-shot inference (no ICL examples)
python -m src.icl_inference --model_name codet5 --num_examples 0

# Different selection methods
python -m src.icl_inference --model_name codet5 --num_examples 5 --selection_method diverse

# Limit test samples for quick testing
python -m src.icl_inference --model_name codet5 --num_examples 3 --max_samples 10
```

## Command Line Arguments

### Required Arguments
- `--model_name`: Model to use (`codet5`, `plbart`, `qwen15`, `qwen25`, `qwen25-0.5b`)

### ICL Configuration
- `--num_examples`: Number of ICL examples (default: 3, use 0 for zero-shot)
- `--selection_method`: How to select examples (`random`, `first`, `diverse`, `bm25`)
- `--prompt-version`: `v1` (legacy; the CodeReviewer test turn says "Code:" while
  demos say "Code changes:" — KI-014) or `v2` (test turn uses the dataset suffix,
  matching paper Fig. 4; also truncates prompts from the left so the test turn survives)
- `--demo-seed`: per-sample demo seed via an isolated RNG (default: legacy
  `random.seed(seed + i)` on the global RNG)
- `--allow-exact-match`: bm25 only — keep train candidates whose code equals the
  test code (excluded by default; the excluded count is reported)
- `--max-prompt-tokens`: explicit prompt budget — demos are dropped from the end
  until the prompt fits (default: legacy greedy fit against the model context)

### Data Configuration
- `--dataset`: Dataset to use (`trans_review` or `acra_by_ls_pt`; default from constants.py)
- `--test_data_path`: Path to test dataset (default: from the dataset registry)
- `--train_data_path`: Path to training dataset (default: from the dataset registry)
- `--max_samples`: Limit number of test samples (useful for quick testing)
- `--upload`: Opt in to uploading the results directory to the Hugging Face Hub
- `--verbose`: Print per-sample prompts and generations

### Output Configuration
- `--output_path`: Specific path to save results JSON file
- `--results_dir`: Directory to save results (default: `assets/results/icl/`)

### General
- `--seed`: Random seed for reproducibility (default: 42)
- `--batch-size`: batched left-padded generation (default 1 = legacy per-sample;
  bf16 batching may change rare beam ties, so pin one batch size per model)
- `--attn-implementation`: override the attention backend (default: legacy
  `flash_attention_2` on CUDA; use `sdpa` on machines without flash-attn)

## Examples

### Test Different Numbers of ICL Examples

```bash
# Zero-shot (no examples)
python src/icl_inference.py --model_name codet5 --num_examples 0

# Few-shot with different numbers of examples
python src/icl_inference.py --model_name codet5 --num_examples 1
python src/icl_inference.py --model_name codet5 --num_examples 3
python src/icl_inference.py --model_name codet5 --num_examples 5
python src/icl_inference.py --model_name codet5 --num_examples 10
```

### Test Different Selection Methods

```bash
# Random selection (default)
python src/icl_inference.py --model_name codet5 --num_examples 3 --selection_method random

# First N examples from training set
python src/icl_inference.py --model_name codet5 --num_examples 3 --selection_method first

# Diverse selection (spread across training set)
python src/icl_inference.py --model_name codet5 --num_examples 3 --selection_method diverse
```

### Test Different Models

```bash
# Seq2Seq models
python src/icl_inference.py --model_name codet5 --num_examples 3
python src/icl_inference.py --model_name plbart --num_examples 3

# Causal LM models (requires more GPU memory)
python src/icl_inference.py --model_name qwen15 --num_examples 3
python src/icl_inference.py --model_name qwen25 --num_examples 3
```

### Quick Testing

```bash
# Test with only 10 samples for quick feedback
python src/icl_inference.py --model_name codet5 --num_examples 3 --max_samples 10
```

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "model_name": "codet5",
  "num_icl_examples": 3,
  "selection_method": "random",
  "num_test_samples": 1719,
  "bleu_score": 0.1234,
  "exact_match_score": 0.0567,
  "rouge_scores": {"rouge1": 0.1, "rouge2": 0.05, "rougeL": 0.1, "rougeLsum": 0.1},
  "bertscore": {"precision": 0.8, "recall": 0.8, "f1": 0.8},
  "meteor_score": 0.1,
  "predictions": ["Generated review 1", "..."],
  "references": ["Reference review 1", "..."],
  "prompts": ["Full rendered prompt 1", "..."],
  "seed": 42,
  "max_context_length": 32768,
  "context_truncation_count": 0,
  "example_reduction_count": 0,
  "max_context_utilization": false,
  "prompt_version": "v1",
  "demo_seed": null,
  "retrieval_method": "random",
  "exact_match_excluded": 0,
  "demo_count_effective": 3.0,
  "prompt_tokens_mean": 512.4,
  "prompt_tokens_max": 2048,
  "batch_size": 1,
  "attn_implementation": null,
  "max_prompt_tokens": null
}
```

The `*_score` keys and list fields are frozen (paper reproducibility); the
revision keys from `prompt_version` down are additive.

## Default File Naming

When `--output_path` is not specified, legacy runs (every new flag at its
default) keep the exact legacy name
`{model_name}_icl_{num_examples}ex_{selection_method}.json`; any non-default
run gets `[_s{demo_seed}]_{prompt_version}` appended so revision outputs can
never overwrite legacy artifacts.

Examples:
- `codet5_icl_3ex_random.json`
- `qwen15_icl_0ex_random.json` (zero-shot)
- `plbart_icl_5ex_diverse.json`
- `qwen25_icl_3ex_random_s1_v2.json` (demo seed 1, prompt v2)

## Performance Notes

### Memory Requirements
- **CodeT5/PLBART**: ~4-8GB GPU memory
- **Qwen1.5/Qwen2.5**: ~14-20GB GPU memory (7B parameters)

### Speed Estimates (approximate)
- **CodeT5/PLBART**: ~2-5 samples/second
- **Qwen models**: ~1-2 samples/second

### Full Dataset Processing Time
- **Small models**: ~10-30 minutes for full test set (1,719 TransReview samples)
- **Large models**: ~30-60 minutes for full test set (longer for the 10,169
  CodeReviewer samples); `--batch-size` is the main lever

## Tips

1. **Start small**: Use `--max_samples 10` for initial testing
2. **GPU memory**: Use smaller models (CodeT5/PLBART) if you have limited GPU memory
3. **Reproducibility**: Keep the same `--seed` value for comparable results
4. **Batch processing**: Use the provided batch scripts to run multiple experiments efficiently

## Troubleshooting

### Out of Memory Errors
- Reduce `--max_samples` to process fewer samples
- Use smaller models (CodeT5/PLBART instead of Qwen)
- Reduce `--num_examples` to use fewer ICL examples

### Slow Performance
- Use `--max_samples` to limit test samples for quick experiments
- Consider using CPU-only mode for testing (though much slower)

### Empty Generations
- Some models might generate empty responses; these are marked as `<EMPTY>` in results
- Check model loading and tokenizer configuration if this happens frequently
