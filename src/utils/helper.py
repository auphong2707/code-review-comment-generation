import json
import os
import random
import numpy as np
from peft import PeftConfig, PeftModel
import torch

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset, load_dataset, Sequence, Features, Value, disable_caching
from huggingface_hub import snapshot_download, login
from typing import Tuple

disable_caching()

from .constants import *
TOKENIZER = None

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU training
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility in CNNs
    torch.backends.cudnn.benchmark = False  # Disables auto-tuning for reproducibility

def load_jsonl(file_path) -> list:
    """
    Load a JSONL (JSON Lines) file and return its contents as a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be loaded.

    Returns:
        list: A list of dictionaries, each representing a JSON object from a line in the file.
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def tokenize_for_fedm(data_point) -> dict:
    """
    Tokenizes the input data point into model inputs and labels.
    Let DataCollatorForSeq2Seq handle the label masking.
    """
    if DATASET_NAME == "trans_review":
        input_text = " ".join(data_point['code_tokens'])
        target_summary = " ".join(data_point['docstring_tokens'])  
    elif DATASET_NAME == "acra_by_ls_pt":
        input_text = data_point['patch']
        target_summary = data_point['msg']
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")
    
    model_inputs = TOKENIZER(
        input_text, 
        max_length=INPUT_MAX_LENGTH, 
        truncation=True, 
        padding='max_length',
        return_tensors=None
    )
        
    labels = TOKENIZER(
        target_summary, 
        max_length=OUTPUT_MAX_LENGTH, 
        truncation=True, 
        padding=False,
        return_tensors=None
    )
        
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

def prepare_dataset_for_fedm(tokenizer_name: str, subset_percentage: float = 1.0) -> Tuple[Dataset, Dataset, Dataset, AutoTokenizer]:
    """
    Prepares the dataset for full encoder-decoder model training.

    This function performs the following steps:
    1. Downloads the dataset from the Hugging Face repository.
    2. Loads the training and validation data from JSONL files.
    3. Converts the loaded data into Dataset objects.
    4. Tokenizes the data using a specified tokenizer.

    Returns:
        Tuple[Dataset, Dataset, Dataset, AutoTokenizer]: A tuple containing the training, validation, and test datasets,
        along with the tokenizer used for tokenization.
    """
    # Load the tokenizer
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    # Download the dataset
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    snapshot_download(repo_id=HUGGINGFACE_DATASET_ID, 
                      repo_type="dataset",
                      local_dir=".",
                      allow_patterns="*.jsonl"
                     )

    # For JSONL files, explicitly specify field_delimiter for proper parsing
    data_files = {
        'train': TRAIN_DATA_DIR,
        'validation': VAL_DATA_DIR, 
        'test': TEST_DATA_DIR
    }
    
    if DATASET_NAME == "trans_review":
        column_names = ['code_tokens', 'docstring_tokens']
        features = Features({
            'code_tokens': Sequence(Value('string')),
            'docstring_tokens': Sequence(Value('string'))
        })
    elif DATASET_NAME == "acra_by_ls_pt":
        column_names = ['patch', 'msg']
        features = Features({
            "oldf": Value("string"),
            "patch": Value("string"),
            "msg":   Value("string"),
            "id":    Value("int64"),
            "y":     Value("int64"),
            "idx":   Value("int64"),   # present in some files
            "proj":  Value("string"),  # present in some files
            "lang":  Value("string"),  # present in some files
        })
    

    # Load datasets
    dataset = load_dataset('json', data_files=data_files, features=features)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    train_dataset = train_dataset.select_columns(column_names)
    val_dataset = val_dataset.select_columns(column_names)
    test_dataset = test_dataset.select_columns(column_names)

    # Apply subset percentage
    if subset_percentage < 1.0:
        train_dataset = train_dataset.select(range(int(len(train_dataset) * subset_percentage)))
        val_dataset = val_dataset.select(range(int(len(val_dataset) * subset_percentage)))
        test_dataset = test_dataset.select(range(int(len(test_dataset) * subset_percentage)))


    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_for_fedm, batched=False, remove_columns=train_dataset.column_names, num_proc=NUM_WORKERS)
    val_dataset = val_dataset.map(tokenize_for_fedm, batched=False, remove_columns=val_dataset.column_names, num_proc=NUM_WORKERS)
    test_dataset = test_dataset.map(tokenize_for_fedm, batched=False, remove_columns=test_dataset.column_names, num_proc=NUM_WORKERS)

    return train_dataset, val_dataset, test_dataset, TOKENIZER


def build_prompt_messages(data_point):
    """Short, strict review prompt."""

    if DATASET_NAME == "trans_review":
        code = " ".join(data_point["code_tokens"])
        review_comment = " ".join(data_point["docstring_tokens"])
    elif DATASET_NAME == "acra_by_ls_pt":
        code = data_point["patch"]
        review_comment = data_point["msg"]
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")

    if not code.strip() or not review_comment.strip():
        raise ValueError("Invalid data")

    # Build optional suffix outside the f-string to avoid f-string parsing ambiguities
    suffix = " changes" if DATASET_NAME == "acra_by_ls_pt" else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a code reviewer. Output exactly ONE concise, actionable comment "
                "on the code. No restating, no extra context, no pleasantries. "
                "Recommend to use up to 60 words. (longer if needed)"
            ),
        },
        {
            "role": "user",
            "content": f"Code{suffix}:\n```{code}```\nReview comment?",
        },
        {
            "role": "assistant",
            "content": review_comment.strip(),
        },
    ]
    
    return messages

def tokenize_for_qwen_eval(data_point):
    messages = build_prompt_messages(data_point)

    label = messages[-1]['content']  # Assistant's response only
    prompt = TOKENIZER.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True
    )
    
    ids = TOKENIZER(prompt,
                    max_length=INPUT_MAX_LENGTH,
                    truncation=True,
                    padding='max_length')
    
    label_ids = TOKENIZER(label,
                        max_length=OUTPUT_MAX_LENGTH,
                        truncation=True,
                        padding='max_length')

    ids["labels"] = label_ids['input_ids']

    return ids

def prepare_dataset_for_qwen_llms(tokenizer_name: str, mask_code_loss: bool = True, subset_percentage: float = 1.0):
    """Normalize JSONL code/review -> prompt Qwen and add label. Returns train/val/test Dataset & AutoTokenizer.
     
    Args:
        tokenizer_name (str): The name of the tokenizer to be used for tokenization.
        mask_code_loss (bool, optional): Whether to mask the code loss during training. Defaults to True.
    """
    
    # Download the dataset
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    snapshot_download(repo_id=HUGGINGFACE_DATASET_ID, 
                      repo_type="dataset",
                      local_dir=".",
                      allow_patterns="*.jsonl"
                     )

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Configure tokenizer for left-padding
    TOKENIZER.padding_side = "left"
    
    def tokenize_for_qwen_train(data_point):
        messages = build_prompt_messages(data_point)
        
        # Apply chat template to get the full prompt
        full_prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # False because we include the assistant response
        )
        
        # Tokenize the full prompt with left padding
        ids = TOKENIZER(full_prompt,
                       max_length=INPUT_MAX_LENGTH + OUTPUT_MAX_LENGTH,
                       truncation=True,
                       padding='max_length')
        
        labels = ids['input_ids'].copy()
        
        if mask_code_loss:
            # We need to find where the assistant's response starts to only compute loss on that part
            # First, get the prompt without the assistant's response
            user_messages = messages[:-1]  # All messages except the assistant's response
            user_prompt = TOKENIZER.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True  # True to add the assistant prompt
            )
            
            # Tokenize just the user part with left padding
            user_ids = TOKENIZER(user_prompt, 
                               max_length=INPUT_MAX_LENGTH + OUTPUT_MAX_LENGTH,
                               truncation=True,
                               add_special_tokens=False)
            
            # With left padding, we need to adjust our masking strategy
            # Find the actual start of content (after padding tokens)
            pad_token_id = TOKENIZER.pad_token_id
            
            # Find where padding ends in the full sequence
            full_ids = ids['input_ids']
            try:
                # Find first non-pad token
                content_start_idx = next(i for i, token_id in enumerate(full_ids) if token_id != pad_token_id)
            except StopIteration:
                # If all tokens are padding (shouldn't happen with real data)
                content_start_idx = 0
            
            # Find where padding ends in user sequence
            user_ids_list = user_ids['input_ids']
            try:
                user_content_start_idx = next(i for i, token_id in enumerate(user_ids_list) if token_id != pad_token_id)
                user_content_length = len(user_ids_list) - user_content_start_idx
            except StopIteration:
                user_content_length = len(user_ids_list)
            
            # Mask padding tokens and user content
            # First mask all padding tokens
            for i in range(content_start_idx):
                labels[i] = -100
            
            # Then mask user content length from where actual content starts
            mask_end_idx = min(content_start_idx + user_content_length, len(labels))
            for i in range(content_start_idx, mask_end_idx):
                labels[i] = -100
        
        # Also mask padding tokens in labels regardless of mask_code_loss setting
        pad_token_id = TOKENIZER.pad_token_id
        labels = [(-100 if token_id == pad_token_id else token_id) for token_id in labels]
        
        ids["labels"] = labels
        return ids

    # Download the dataset
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    snapshot_download(repo_id=HUGGINGFACE_DATASET_ID,
                     repo_type="dataset",
                     local_dir=".",
                     allow_patterns="*.jsonl"
                    )
    
    # For JSONL files, explicitly specify field_delimiter for proper parsing
    data_files = {
        'train': TRAIN_DATA_DIR,
        'validation': VAL_DATA_DIR, 
        'test': TEST_DATA_DIR
    }
    
    if DATASET_NAME == "trans_review":
        column_names = ['code_tokens', 'docstring_tokens']
        features = Features({
            'code_tokens': Sequence(Value('string')),
            'docstring_tokens': Sequence(Value('string'))
        })
    elif DATASET_NAME == "acra_by_ls_pt":
        column_names = ['patch', 'msg']
        features = Features({
            "oldf": Value("string"),
            "patch": Value("string"),
            "msg":   Value("string"),
            "id":    Value("int64"),
            "y":     Value("int64"),
            "idx":   Value("int64"),   # present in some files
            "proj":  Value("string"),  # present in some files
            "lang":  Value("string"),  # present in some files
        })
    
    # Load datasets
    dataset = load_dataset('json', data_files=data_files, features=features)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    train_dataset = train_dataset.select_columns(column_names)
    val_dataset = val_dataset.select_columns(column_names)
    test_dataset = test_dataset.select_columns(column_names)

    # Apply subset percentage
    if subset_percentage < 1.0:
        train_dataset = train_dataset.select(range(int(len(train_dataset) * subset_percentage)))
        val_dataset = val_dataset.select(range(int(len(val_dataset) * subset_percentage)))
        test_dataset = test_dataset.select(range(int(len(test_dataset) * subset_percentage)))

    # Tokenize the data
    train_dataset = train_dataset.map(tokenize_for_qwen_train, batched=False, remove_columns=train_dataset.column_names, num_proc=NUM_WORKERS)
    val_dataset = val_dataset.map(tokenize_for_qwen_train, batched=False, remove_columns=val_dataset.column_names, num_proc=NUM_WORKERS)
    test_dataset = test_dataset.map(tokenize_for_qwen_eval, batched=False, remove_columns=test_dataset.column_names, num_proc=NUM_WORKERS)

    print(f"Sample tokenized train data point: {train_dataset[0]}")
    print(f"Sample tokenized val data point: {val_dataset[0]}")
    print(f"Sample tokenized test data point: {test_dataset[0]}")
    
    return train_dataset, val_dataset, test_dataset, TOKENIZER

def load_model_and_tokenizer(model_directory, model_name, model_method):
    """Load a model for inference.

    Args:
        model_directory (str): The directory containing the model files.
        model_name (str): The name of the model to load ("codet5", "plbart", "qwen15", "qwen25")
        model_method (str): The PEFT method to train model
    Returns:
        AutoModelForCausalLM: The loaded model.
    """
    model, tokenizer = None, None
    
    # 1. Load the tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    # 2. Load the pretrained model from directory in case the method is FMFT
    if model_method == "fmft":
        if model_name in ["codet5", "plbart"]:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
        elif model_name in ["qwen15", "qwen25", "qwen25-0.5b-instruct"]:
            model = AutoModelForCausalLM.from_pretrained(
                model_directory,    
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )

    # 3. Load the model in case the method is PEFT
    else:
        # 3.1. Load adapter config
        peft_cfg = PeftConfig.from_pretrained(model_directory)

        # 3.2. Load the base model
        if model_name in ["codet5", "plbart"]:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_cfg.base_model_name_or_path)
        elif model_name in ["qwen15", "qwen25"]:
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_cfg.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
        
        # 3.3. Resize token embeddings to match the tokenizer
        # This must be done before loading the PEFT adapter
        print(f"Original model vocab size: {base_model.config.vocab_size}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        if len(tokenizer) != base_model.config.vocab_size:
            print(f"Resizing model embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"New model vocab size: {base_model.config.vocab_size}")
            
        # 3.4. Load the adapter (IA3, LoRA, QLoRA, P-Tuning V2)
        model = PeftModel.from_pretrained(base_model, model_directory)

    model.eval()

    return model, tokenizer

def load_dataset_for_eval(dataset_directory, model_name, tokenizer):
    """Load a dataset for evaluation.

    Args:
        dataset_directory (str): The directory containing the dataset files.

    Returns:
        Dataset: The loaded dataset.
    """
    # Load the tokenizer
    global TOKENIZER
    TOKENIZER = tokenizer
    
    if model_name in ["qwen15", "qwen25", "qwen25-0.5b-instruct"]:
        TOKENIZER.padding_side = "left"

    # Load the dataset from jsonl files
    dataset = Dataset.from_list(load_jsonl(dataset_directory))

    # Choose the tokenize function
    tokenize_function = tokenize_for_qwen_eval if model_name in ["qwen15", "qwen25", "qwen25-0.5b-instruct"] else tokenize_for_fedm

    # Map the tokenize function to the dataset
    dataset = dataset.map(
        tokenize_function, 
        batched=False, 
        remove_columns=dataset.column_names, 
        num_proc=NUM_WORKERS
    )

    return dataset
