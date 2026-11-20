# Code Review Generation with Deep Learning Models

This repository contains implementations for training various deep learning models for automated code review generation.

The revision campaign runs on Spartan HPC (University of Melbourne, Slurm, A100 80GB) as one
job file, `scripts/spartan/revision.slurm` (1–4 GPUs, default 2). Before running, the job
checks the public Hugging Face results dataset, then runs the campaign, writes a log per
section and pushes new results back. Setup and usage: [scripts/spartan/README.md](scripts/spartan/README.md).

## Project Structure

```
├── src/                          # Source code
│   ├── train.py                  # Unified training entry point (all models × methods)
│   ├── training/                 # Training internals
│   │   ├── registry.py           # (model, method) -> spec resolution + experiment naming
│   │   ├── specs.py              # Per-combination training specs (generated, frozen)
│   │   ├── peft_setup.py         # Config object builders
│   │   ├── causal_runner.py      # Qwen family pipeline
│   │   └── seq2seq_runner.py     # CodeT5/PLBart pipeline
│   ├── evaluation/
│   │   └── metrics.py            # Shared BLEU-4/EM/ROUGE/BERTScore/METEOR implementation
│   ├── llm_as_judge/             # Optional LLM-as-judge evaluator (see LLM_AS_JUDGE.md)
│   ├── utils/
│   │   ├── constants.py          # Hyperparameters, paths, dataset registry
│   │   ├── helper.py             # Dataset prep, model loading, seeding
│   │   └── custom_modified_api.py# Custom trainer + wandb callbacks
│   ├── tests/                    # Pytest suite (judge, metrics, integration)
│   ├── icl_inference.py          # In-Context Learning inference script
│   ├── manual_upload_to_huggingface.py
│   └── re_eval_model.py          # Model re-evaluation script
├── tools/                        # Operational scripts (judge table updates, manual checks)
├── data/                         # Analysis notebooks
│   └── rcg_eda.ipynb
├── assets/                       # Everything not in git: data/, models/, results/, logs/, wandb/, cache/
├── run_training.sh               # Training dispatch script
├── ICL_README.md                 # ICL-specific documentation
├── LLM_AS_JUDGE.md               # LLM-as-judge documentation
├── pyproject.toml                # Package + pytest configuration
├── environment.yml               # Local Conda environment with CUDA 11.7
├── requirements.txt              # Local Python dependencies
├── LICENSE                       # License file
└── README.md                     # This file
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
- **P-Tuning v2** *(experimental — implemented but not validated by published results)*
- **ICL**: In-Context Learning

### Method Availability by Model
| Model | FMFT | IA3 | LoRA | QLoRA | P-Tuning* | ICL |
|-------|------|-----|------|-------|-----------|-----|
| CodeT5-base | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| PLBart-base | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| CodeQwen1.5-7B-Chat | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5-Coder-7B-Instruct | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5-0.5B-Instruct | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |

*P-Tuning is experimental.

## Usage

### Quick Start with Training Script (Examples)

The easiest way to run training is using the provided script (it dispatches to
the unified entry point `python -m src.train`):

```bash
# Use eager attention for Qwen training with the local CUDA 11.7 environment.
export TRAIN_EXTRA_ARGS="--attn-implementation eager"

# SWEEP mode - Run hyperparameter sweep (experiments 0-4)
bash run_training.sh qwen15 ia3 sweep 0 4 0.2      # Qwen1.5 with IA3, subset=0.2
bash run_training.sh codet5 qlora sweep 0 2 0.5    # CodeT5 with QLoRA, subset=0.5

# FIXED mode examples with tested learning rates:
bash run_training.sh codet5 fmft fixed 1e-4        # CodeT5 with FMFT
bash run_training.sh plbart lora fixed 1e-3        # PLBart with LoRA
bash run_training.sh qwen15 ia3 fixed 1e-5         # Qwen1.5 with IA3
bash run_training.sh qwen25 lora fixed 1e-5        # Qwen2.5 with LoRA
bash run_training.sh qwen25-0.5b fmft fixed 5e-5   # Qwen2.5-0.5B with FMFT

# Print the resolved configuration without training:
TRAIN_DRY_RUN=1 bash run_training.sh qwen25 lora fixed 5e-5

# Direct invocation with extra options (--dataset, --smoke, --dry-run):
python -m src.train --model qwen25 --method lora --learning_rate 5e-5 --attn-implementation eager
python -m src.train --model codet5 --method lora --smoke --subset 0.01  # 2-step test run
```

### Running In-Context Learning (ICL)

For models that support ICL, you can run inference without fine-tuning:

```bash
# Qwen2.5-Coder-7B-Instruct ICL with different numbers of examples
python -m src.icl_inference --model_name qwen25 --num_examples 3 --selection_method random --attn-implementation eager
python -m src.icl_inference --model_name qwen25 --num_examples 5 --selection_method random --attn-implementation eager
python -m src.icl_inference --model_name qwen25 --num_examples 10 --selection_method random --attn-implementation eager

# Qwen2.5-0.5B-Instruct ICL with different numbers of examples
python -m src.icl_inference --model_name qwen25-0.5b --num_examples 3 --selection_method random --attn-implementation eager
python -m src.icl_inference --model_name qwen25-0.5b --num_examples 5 --selection_method random --attn-implementation eager
python -m src.icl_inference --model_name qwen25-0.5b --num_examples 10 --selection_method random --attn-implementation eager

# CodeQwen1.5-7B-Chat ICL with different numbers of examples
python -m src.icl_inference --model_name qwen15 --num_examples 3 --selection_method random --attn-implementation eager
python -m src.icl_inference --model_name qwen15 --num_examples 5 --selection_method random --attn-implementation eager
python -m src.icl_inference --model_name qwen15 --num_examples 10 --selection_method random --attn-implementation eager
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

## Local installation with Conda

The local environment targets Linux x86-64 with an NVIDIA GPU and a driver
that supports CUDA 11.7. Conda installs Python 3.11, PyTorch 2.0.1, and the
CUDA 11.7 runtime. For Spartan HPC, follow the [Spartan setup guide](scripts/spartan/README.md).

From the repository root, create `codereview` with Python 3.11 and IPython.
Then install the CUDA-enabled PyTorch build and the project dependencies.

```bash
conda create -n codereview python=3.11 ipython pip --override-channels -c conda-forge
conda activate codereview
conda install --override-channels -c pytorch -c nvidia -c conda-forge \
    'pytorch=2.0.1=*cuda11.7*' pytorch-cuda=11.7 'mkl<2024.1'
DS_BUILD_OPS=0 python -m pip install -r requirements.txt
```

Run the entry points directly from the checkout. Verify the dependencies and GPU before running the tests.

```bash
python -m pip check
python -c 'import torch; assert torch.__version__.split("+", 1)[0] == "2.0.1"; assert torch.version.cuda == "11.7"; print(torch.cuda.get_device_name(0), torch.ones(1, device="cuda").item())'
WANDB_MODE=offline python -m pytest
```

Use `--attn-implementation eager` for local Qwen training and ICL, as in the
examples above. The pinned Transformers release requires PyTorch 2.1.1 or newer
for SDPA. Causal checkpoint re-evaluation requires a separate, compatible
FlashAttention 2 installation. DeepSpeed workloads that compile CUDA extensions
also need a matching CUDA 11.7 toolkit with `nvcc`.

Set `WANDB_API_KEY` for online Weights & Biases logging and
`HUGGINGFACE_TOKEN` for authenticated Hugging Face access.
