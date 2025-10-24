# ICL Inference for Code Review Generation

This directory contains scripts for running In-Context Learning (ICL) inference on code review generation models.

## Files

- `src/icl_inference.py` - Main ICL inference script
- `run_icl_batch.sh` - Bash script for running multiple ICL experiments (Linux/Mac)
- `run_icl_batch.ps1` - PowerShell script for running multiple ICL experiments (Windows)

## Quick Start

### Single Experiment

Run ICL inference for a specific model:

```bash
# Basic usage - 3 examples with random selection
python src/icl_inference.py --model_name codet5 --num_examples 3

# Zero-shot inference (no ICL examples)
python src/icl_inference.py --model_name codet5 --num_examples 0

# Different selection methods
python src/icl_inference.py --model_name codet5 --num_examples 5 --selection_method diverse

# Limit test samples for quick testing
python src/icl_inference.py --model_name codet5 --num_examples 3 --max_samples 10
```

### Batch Experiments

Run multiple experiments with different settings:

**Linux/Mac:**
```bash
chmod +x run_icl_batch.sh
./run_icl_batch.sh
```

**Windows:**
```powershell
.\run_icl_batch.ps1
```

## Command Line Arguments

### Required Arguments
- `--model_name`: Model to use (`codet5`, `plbart`, `qwen15`, `qwen25`)

### ICL Configuration
- `--num_examples`: Number of ICL examples (default: 3, use 0 for zero-shot)
- `--selection_method`: How to select examples (`random`, `first`, `diverse`)

### Data Configuration
- `--test_data_path`: Path to test dataset (default: from constants.py)
- `--train_data_path`: Path to training dataset (default: from constants.py)
- `--max_samples`: Limit number of test samples (useful for quick testing)

### Output Configuration
- `--output_path`: Specific path to save results JSON file
- `--results_dir`: Directory to save results (default: `./results/icl/`)

### General
- `--seed`: Random seed for reproducibility (default: 42)

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
  "num_test_samples": 1720,
  "bleu_score": 0.1234,
  "exact_match_score": 0.0567,
  "predictions": ["Generated review 1", "Generated review 2", ...],
  "references": ["Reference review 1", "Reference review 2", ...],
  "seed": 42
}
```

## Default File Naming

When `--output_path` is not specified, files are automatically named as:
`{model_name}_icl_{num_examples}ex_{selection_method}.json`

Examples:
- `codet5_icl_3ex_random.json`
- `qwen15_icl_0ex_random.json` (zero-shot)
- `plbart_icl_5ex_diverse.json`

## Performance Notes

### Memory Requirements
- **CodeT5/PLBART**: ~4-8GB GPU memory
- **Qwen1.5/Qwen2.5**: ~14-20GB GPU memory (7B parameters)

### Speed Estimates (approximate)
- **CodeT5/PLBART**: ~2-5 samples/second
- **Qwen models**: ~1-2 samples/second

### Full Dataset Processing Time
- **Small models**: ~10-30 minutes for full test set (1720 samples)
- **Large models**: ~30-60 minutes for full test set

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
