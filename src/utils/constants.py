"""
This file contains all the constants used in the project.
"""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# [0. GENERAL CONSTANTS]
# -----------------------------------------------------------------------------------
# All run artifacts live under assets/ (gitignored): results, logs, wandb, HF cache.
ASSETS_DIR                  = "assets/"
DATA_DIR                    = ASSETS_DIR + "data/"
MODELS_DIR                  = ASSETS_DIR + "models/"      # produced HPC artifacts (read-only)
RESULTS_DIR                 = ASSETS_DIR + "results/"     # local training/ICL outputs
LOGS_DIR                    = ASSETS_DIR + "logs/"        # run_training.sh console logs
WANDB_DIR                   = ASSETS_DIR + "wandb/"       # wandb run dirs
CACHE_DIR                   = ASSETS_DIR + "cache/"       # HF_HOME default
LOGGING_STEPS               = 5
SEED                        = 42

DATASET_NAME                = "acra_by_ls_pt" # Can be: acra_by_ls_pt or trans_review
HUGGINGFACE_DATASET_ID      = "your-huggingface-username/acra-by-ls-pt-dataset" # Update with your actual dataset ID
TRAIN_DATA_DIR              = DATA_DIR + 'Code_Review/msg-train.jsonl'
VAL_DATA_DIR                = DATA_DIR + 'Code_Review/msg-valid.jsonl'
TEST_DATA_DIR               = DATA_DIR + 'Code_Review/msg-test.jsonl'

INPUT_MAX_LENGTH            = 512
OUTPUT_MAX_LENGTH           = 256
OUTPUT_MIN_LENGTH           = 2
NUM_BEAMS                   = 3
DEEPSPEED_CFG               = "config/ds_zero3_config.json"

NUM_WORKERS                 = 4

WANDB_PROJECT                = "code-review-comment-generation"
HUGGINGFACE_REPO_ID          = "your-huggingface-username/code-review-comment-generation" # Update with your actual repo ID

LR = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
# -----------------------------------------------------------------------------------



# [1. CODET5-BASE HYPERPARAMETERS]
# -----------------------------------------------------------------------------------
MODEL_CT5                  = "Salesforce/codet5-base"
TOKENIZER_CT5              = "Salesforce/codet5-base"
RESULTS_DIR_CT5            = RESULTS_DIR + "code-t5/"

TRAIN_BATCH_SIZE_CT5       = 8
EVAL_BATCH_SIZE_CT5        = 8
NUM_TRAIN_EPOCHS_CT5       = 8
WEIGHT_DECAY_CT5           = 0.01
# -----------------------------------------------------------------------------------



# [2. PLBART-BASE HYPERPARAMETERS]
# -----------------------------------------------------------------------------------
MODEL_PLBART               = "uclanlp/plbart-base"
TOKENIZER_PLBART           = "uclanlp/plbart-base"
RESULTS_DIR_PLBART         = RESULTS_DIR + "plbart/"

TRAIN_BATCH_SIZE_PLBART    = 8
EVAL_BATCH_SIZE_PLBART     = 8
NUM_TRAIN_EPOCHS_PLBART    = 8
WEIGHT_DECAY_PLBART        = 0.01
# -----------------------------------------------------------------------------------



# [3. QWEN CodeQwen-1.5-7B-Chat HYPERPARAMETERS]
# -----------------------------------------------------------------------------------
MODEL_QWEN15                = "Qwen/CodeQwen1.5-7B-Chat"
TOKENIZER_QWEN15            = MODEL_QWEN15
RESULTS_DIR_QWEN15          = RESULTS_DIR + "qwen15/"

TRAIN_BATCH_SIZE_QWEN15     = 1
EVAL_BATCH_SIZE_QWEN15      = 1
NUM_TRAIN_EPOCHS_QWEN15     = 3
WEIGHT_DECAY_QWEN15         = 0.01
# -----------------------------------------------------------------------------------



# [4. QWEN Qwen-2.5-Coder-7B-Instruct HYPERPARAMETERS]
#-------------------------------------------------------------------------
MODEL_QWEN25                = "Qwen/Qwen2.5-Coder-7B-Instruct"
TOKENIZER_QWEN25            = MODEL_QWEN25
RESULTS_DIR_QWEN25          = RESULTS_DIR + "qwen25/"

TRAIN_BATCH_SIZE_QWEN25     = 1
EVAL_BATCH_SIZE_QWEN25      = 1
NUM_TRAIN_EPOCHS_QWEN25     = 3
WEIGHT_DECAY_QWEN25         = 0.01
#-------------------------------------------------------------------------



# [5. QWEN Qwen-2.5-0.5B-Instruct HYPERPARAMETERS]
#-------------------------------------------------------------------------
MODEL_QWEN25_0_5B                = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_QWEN25_0_5B            = MODEL_QWEN25_0_5B
RESULTS_DIR_QWEN25_0_5B          = RESULTS_DIR + "qwen25-0.5b-instruct/"

TRAIN_BATCH_SIZE_QWEN25_0_5B     = 8
EVAL_BATCH_SIZE_QWEN25_0_5B      = 8
NUM_TRAIN_EPOCHS_QWEN25_0_5B     = 8
WEIGHT_DECAY_QWEN25_0_5B         = 0.01
#-------------------------------------------------------------------------



# [6. PEFT HYPERPARAMETERS]
#-------------------------------------------------------------------------
NUM_VIRTUAL_TOKENS          = 100
ENCODER_REPARAMETERIZATION_TYPE = "MLP"

RANK_LORA                   = 16
ALPHA_LORA                  = 16

RANK_QLORA                  = 16
ALPHA_QLORA                 = 16
#-------------------------------------------------------------------------



# [7. DATASET REGISTRY]
#-------------------------------------------------------------------------
# Per-dataset column/prompt/path configuration. DATASET_NAME above stays the
# default; scripts accept --dataset to select one explicitly.
DATASET_CONFIGS = {
    "trans_review": {
        "input_column": "code_tokens",
        "target_column": "docstring_tokens",
        "join_tokens": True,           # columns are token lists, joined with spaces
        "prompt_suffix": "",           # prompt header reads "Code:"
        "data_files": {
            "train": "assets/data/Trans_Review/train.jsonl",
            "validation": "assets/data/Trans_Review/val.jsonl",
            "test": "assets/data/Trans_Review/test.jsonl",
        },
    },
    "acra_by_ls_pt": {
        "input_column": "patch",
        "target_column": "msg",
        "join_tokens": False,
        "prompt_suffix": " changes",   # prompt header reads "Code changes:"
        "data_files": {
            "train": TRAIN_DATA_DIR,
            "validation": VAL_DATA_DIR,
            "test": TEST_DATA_DIR,
        },
    },
}


def get_dataset_config(name):
    try:
        return DATASET_CONFIGS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset: {name}; expected one of {sorted(DATASET_CONFIGS)}"
        ) from None


def is_placeholder_repo(repo_id):
    """True for the shipped 'your-huggingface-username/...' placeholder ids."""
    return isinstance(repo_id, str) and repo_id.startswith("your-huggingface-username/")
#-------------------------------------------------------------------------
