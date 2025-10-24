# Code Review Generation with Deep Learning Models

This repository contains implementations for training various deep learning models for automated code review generation.

## Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model-specific training scripts
│   │   ├── codet5/              # CodeT5-large model scripts
│   │   │   ├── main_ct5_fmft.py
│   │   │   ├── main_ct5_ia3.py
│   │   │   ├── main_ct5_lora.py
│   │   │   ├── main_ct5_ptuning.py
│   │   │   └── main_ct5_qlora.py
│   │   ├── plbart/              # PLBart-large model scripts
│   │   │   ├── main_plbart_fmft.py
│   │   │   ├── main_plbart_ia3.py
│   │   │   ├── main_plbart_lora.py
│   │   │   ├── main_plbart_ptuning.py
│   │   │   └── main_plbart_qlora.py
│   │   ├── qwen15/              # Qwen1.5 model scripts
│   │   │   ├── main_qwen15_fmft.py
│   │   │   ├── main_qwen15_ia3.py
│   │   │   ├── main_qwen15_lora.py
│   │   │   ├── main_qwen15_ptuning.py
│   │   │   └── main_qwen15_qlora.py
│   │   └── qwen25/              # Qwen2.5 model scripts
│   │       ├── main_qwen25_fmft.py
│   │       ├── main_qwen25_ia3.py
│   │       ├── main_qwen25_lora.py
│   │       ├── main_qwen25_ptuning.py
│   │       └── main_qwen25_qlora.py
│   └── utils/                   # Utility modules
│       ├── constants.py         # Configuration constants
│       └── helper.py            # Helper functions
├── config/                      # Configuration files
│   └── ds_zero3_config.json    # DeepSpeed configuration
├── data/                        # Dataset and analysis notebooks
├── documents/                   # Research papers and documentation
├── tests/                       # Test files
└── README.md                    # This file
```

## Models and Training Methods

### Supported Models
- **CodeT5-large**: Salesforce/codet5-large
- **PLBart-large**: uclanlp/plbart-large
- **Qwen1.5**: Qwen/CodeQwen1.5-7B-Chat
- **Qwen2.5**: Qwen/CodeQwen2.5-7B-Instruct

### Training Methods
- **FMFT**: Full Model Fine-Tuning
- **IA3**: (IA)³ - Infused Adapter by Inhibiting and Amplifying Inner Activations
- **LoRA**: Low-Rank Adaptation
- **QLoRA**: Quantized Low-Rank Adaptation
- **P-tuning**: Prompt Tuning

### Method Availability by Model
| Model | FMFT | IA3 | LoRA | QLoRA | P-tuning |
|-------|------|-----|------|-------|----------|
| CodeT5-large | ✅ | ✅ | ✅ | ✅ | ✅ |
| PLBart-large | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen1.5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5 | ✅ | ✅ | ✅ | ✅ | ✅ |

## Usage

### Quick Start with Training Script

The easiest way to run training is using the provided script:

```bash
# Make the script executable (first time only)
chmod +x run_training.sh

# Run training for different models and methods
./run_training.sh qwen15 fmft     # Qwen1.5 with full fine-tuning
./run_training.sh qwen25 lora     # Qwen2.5 with LoRA
./run_training.sh codet5 ia3      # CodeT5 with IA3
./run_training.sh codet5 lora     # CodeT5 with LoRA  
./run_training.sh codet5 qlora    # CodeT5 with QLoRA
./run_training.sh plbart ptuning  # PLBart with P-tuning
./run_training.sh plbart lora     # PLBart with LoRA
./run_training.sh plbart qlora    # PLBart with QLoRA
```

### Manual Execution

### Running Full Model Fine-Tuning (FMFT)

For Qwen models with DeepSpeed:
```bash
# Qwen1.5 FMFT
deepspeed --num_gpus 8 src/models/qwen15/main_qwen15_fmft.py

# Qwen2.5 FMFT
deepspeed --num_gpus 8 src/models/qwen25/main_qwen25_fmft.py
```

### Running Other Training Methods

```bash
# LoRA training
python src/models/qwen15/main_qwen15_lora.py
python src/models/qwen25/main_qwen25_lora.py

# IA3 training
python src/models/qwen15/main_qwen15_ia3.py
python src/models/qwen25/main_qwen25_ia3.py

# P-tuning
python src/models/qwen15/main_qwen15_ptuning.py
python src/models/qwen25/main_qwen25_ptuning.py

# QLoRA training
python src/models/qwen15/main_qwen15_qlora.py
python src/models/qwen25/main_qwen25_qlora.py
```

### Running CodeT5 and PLBart Models

```bash
# CodeT5 models
python src/models/codet5/main_ct5_fmft.py
python src/models/codet5/main_ct5_ia3.py
python src/models/codet5/main_ct5_lora.py
python src/models/codet5/main_ct5_ptuning.py
python src/models/codet5/main_ct5_qlora.py

# PLBart models
python src/models/plbart/main_plbart_fmft.py
python src/models/plbart/main_plbart_ia3.py
python src/models/plbart/main_plbart_lora.py
python src/models/plbart/main_plbart_ptuning.py
python src/models/plbart/main_plbart_qlora.py
```

## Configuration

All model hyperparameters and training configurations are defined in `src/utils/constants.py`. 
DeepSpeed configuration for distributed training is available in `config/ds_zero3_config.json`.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- DeepSpeed
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- Evaluate
- Weights & Biases (wandb)

## Environment Setup

Make sure to set the following environment variables:
```bash
export WANDB_API_KEY="your_wandb_api_key"
export HUGGINGFACE_TOKEN="your_huggingface_token"
```

## Installation

Install the required dependencies:
```bash
# Option 1: Install directly
pip install torch transformers==4.47.0 datasets evaluate wandb huggingface-hub deepspeed peft

# Option 2: Install from requirements file
pip install -r requirements.txt
```

## Quick Validation

To verify that the refactored codebase is working correctly:

```bash
python validate_setup.py
```

This will check:
- ✅ File structure integrity
- ✅ Import path functionality  
- ✅ Training script availability
- ✅ Path setup in all scripts

## Troubleshooting

### Import Errors
If you encounter import errors like `ModuleNotFoundError: No module named 'constants'`, this means the Python path is not set correctly. Each training script now includes automatic path setup, but if you still have issues:

1. **Use the training script** (recommended):
   ```bash
   ./run_training.sh qwen15 fmft
   ```

2. **Set PYTHONPATH manually**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"
   python src/models/qwen15/main_qwen15_fmft.py
   ```

3. **Run from project root** and ensure you're in the correct directory structure.

### Path Issues
- Make sure you're running commands from the project root directory
- Verify that the `src/utils/constants.py` and `src/utils/helper.py` files exist
- Each training script now includes automatic path detection and setup

## Project Structure Details

- **`src/models/`**: Contains model-specific training scripts organized by model type
- **`src/utils/`**: Shared utilities (constants, helper functions)
- **`config/`**: Configuration files (DeepSpeed, etc.)
- **`tests/`**: Test scripts and validation code
- **`data/`**: Dataset files and analysis notebooks
- **`documents/`**: Research papers and documentation