# Code Review Generation with Deep Learning Models

This repository contains implementations for training various deep learning models for automated code review generation.

## Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Model-specific training scripts
│   │   ├── codet5/              # CodeT5-base model scripts
│   │   │   ├── main_ct5_fmft.py
│   │   │   ├── main_ct5_ia3.py
│   │   │   ├── main_ct5_lora.py
│   │   │   └── main_ct5_qlora.py
│   │   ├── plbart/              # PLBart-base model scripts
│   │   │   ├── main_plbart_fmft.py
│   │   │   ├── main_plbart_ia3.py
│   │   │   ├── main_plbart_lora.py
│   │   │   └── main_plbart_qlora.py
│   │   ├── qwen15/              # Qwen1.5 model scripts
│   │   │   ├── main_qwen15_fmft.py
│   │   │   ├── main_qwen15_ia3.py
│   │   │   ├── main_qwen15_lora.py
│   │   │   └── main_qwen15_qlora.py
│   │   ├── qwen25/              # Qwen2.5 model scripts
│   │   │   ├── main_qwen25_fmft.py
│   │   │   ├── main_qwen25_ia3.py
│   │   │   ├── main_qwen25_lora.py
│   │   │   └── main_qwen25_qlora.py
│   │   └── qwen25-0.5b-instruct/ # Qwen2.5-0.5B model scripts
│   │       ├── main_qwen25_0.5b_fmft.py
│   │       ├── main_qwen25_0.5b_ia3.py
│   │       ├── main_qwen25_0.5b_lora.py
│   │       └── main_qwen25_0.5b_qlora.py
│   ├── utils/                   # Utility modules
│   │   ├── constants.py         # Configuration constants
│   │   └── helper.py            # Helper functions
│   ├── tests/                   # Test files
│   │   ├── download_model.py
│   │   └── test_model.py
│   ├── icl_inference.py         # In-Context Learning inference script
│   ├── manual_upload_to_huggingface.py
│   └── re_eval_model.py         # Model re-evaluation script
├── config/                      # Configuration files
├── data/                        # Dataset and analysis notebooks
│   ├── metrics.ipynb
│   ├── rcg_eda.ipynb
│   ├── Code-Revision-After-Review/
│   ├── Code-Revision-Before-Review/
│   ├── Comment_Generation/
│   └── Review-Comment-Generation/
├── documents/                   # Research papers and documentation
├── run_training.sh              # Training script
├── run_icl_batch.sh            # ICL batch execution script
├── ICL_README.md               # ICL-specific documentation
├── requirements.txt            # Python dependencies
├── LICENSE                     # License file
└── README.md                   # This file
```

## Models and Training Methods

### Supported Models
- **CodeT5-base**: Salesforce/codet5-base
- **PLBart-base**: uclanlp/plbart-base
- **CodeQwen1.5-7B-Chat**: Qwen/CodeQwen1.5-7B-Chat
- **Qwen2.5-Coder-7B-Instruct**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Qwen2.5-0.5B-Instruct**: Qwen/Qwen2.5-0.5B-Instruct

### Training Methods
- **FMFT**: Full Model Fine-Tuning
- **IA3**: (IA)³ - Infused Adapter by Inhibiting and Amplifying Inner Activations
- **LoRA**: Low-Rank Adaptation
- **QLoRA**: Quantized Low-Rank Adaptation
- **ICL**: In-Context Learning

### Method Availability by Model
| Model | FMFT | IA3 | LoRA | QLoRA | ICL |
|-------|------|-----|------|-------|-----|
| CodeT5-base | ✅ | ✅ | ✅ | ✅ | ❌ |
| PLBart-base | ✅ | ✅ | ✅ | ✅ | ❌ |
| CodeQwen1.5-7B-Chat | ❌ | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5-Coder-7B-Instruct | ❌ | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5-0.5B-Instruct | ✅ | ✅ | ✅ | ✅ | ✅ |

## Usage

### Quick Start with Training Script (Examples)

The easiest way to run training is using the provided script:

```bash
# Make the script executable (first time only)
chmod +x run_training.sh

# SWEEP mode - Run hyperparameter sweep (experiments 0-4)
./run_training.sh qwen15 ia3 sweep 0 4 0.2      # Qwen1.5 with IA3, subset=0.2
./run_training.sh codet5 qlora sweep 0 2 0.5    # CodeT5 with QLoRA, subset=0.5

# FIXED mode examples with tested learning rates:
./run_training.sh codet5 fmft fixed 1e-4        # CodeT5 with FMFT
./run_training.sh plbart lora fixed 1e-3        # PLBart with LoRA
./run_training.sh qwen15 ia3 fixed 1e-5         # Qwen1.5 with IA3
./run_training.sh qwen25 lora fixed 1e-5        # Qwen2.5 with LoRA
./run_training.sh qwen25-0.5b fmft fixed 5e-5   # Qwen2.5-0.5B with FMFT
```

### Running In-Context Learning (ICL)

For models that support ICL, you can run inference without fine-tuning:

```bash
# Qwen2.5-Coder-7B-Instruct ICL with different numbers of examples
python src/icl_inference.py --model_name qwen25 --num_examples 3 --selection_method random
python src/icl_inference.py --model_name qwen25 --num_examples 5 --selection_method random
python src/icl_inference.py --model_name qwen25 --num_examples 10 --selection_method random

# Qwen2.5-0.5B-Instruct ICL with different numbers of examples
python src/icl_inference.py --model_name qwen25-0.5b --num_examples 3 --selection_method random
python src/icl_inference.py --model_name qwen25-0.5b --num_examples 5 --selection_method random
python src/icl_inference.py --model_name qwen25-0.5b --num_examples 10 --selection_method random

# CodeQwen1.5-7B-Chat ICL with different numbers of examples
python src/icl_inference.py --model_name qwen15 --num_examples 3 --selection_method random
python src/icl_inference.py --model_name qwen15 --num_examples 5 --selection_method random
python src/icl_inference.py --model_name qwen15 --num_examples 10 --selection_method random
```

## Dataset Information

This project uses two main datasets for training and evaluation:

### TransReview Dataset
- **Source**: [Generated_Review_How_Far Repository](https://github.com/soarsmu/Generated_Review_How_Far/tree/main?tab=readme-ov-file)
- **Download Link**: [Google Drive](https://drive.google.com/file/d/13W5zisw9jGDXHIczyh2D7DI5kLUYYWHn/view?usp=share_link)

### CodeReviewer Dataset
- **Source**: [Zenodo Repository](https://zenodo.org/records/6900648)
- **Download Link**: [Comment_Generation.zip](https://zenodo.org/records/6900648/files/Comment_Generation.zip?download=1)

## Configuration

All model hyperparameters and training configurations are defined in `src/utils/constants.py`.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
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
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

Install the required dependencies:
```bash
# Option 1: Install directly
pip install torch transformers==4.47.0 datasets evaluate wandb huggingface-hub peft

# Option 2: Install from requirements file
pip install -r requirements.txt
```