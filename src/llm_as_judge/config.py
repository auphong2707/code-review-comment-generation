from dataclasses import dataclass
from typing import Optional


DEFAULT_JUDGE_MODEL_PATH = "assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct"
DEFAULT_PROMPT_VERSION = "qwen_judge_v1"


@dataclass(frozen=True)
class JudgeConfig:
    model_path: str = DEFAULT_JUDGE_MODEL_PATH
    k: int = 5
    penalty_lambda: float = 0.25
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 32
    do_sample: bool = True
    prompt_version: str = DEFAULT_PROMPT_VERSION
    cache_path: Optional[str] = None
    local_files_only: bool = False
    device: str = "cuda"
    batch_size: int = 1
    max_code_chars: int = 12000
    code_column: Optional[str] = None
    prediction_column: Optional[str] = None
    reference_column: Optional[str] = None
    mock: bool = False
    # Flag-gated spec completions. Both default OFF: produced Tables 1-5
    # predate them, and neither field enters make_cache_key.
    repair_retry: bool = False
    emit_debug_columns: bool = False

    def __post_init__(self):
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_code_chars < 1:
            raise ValueError("max_code_chars must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.device != "auto" and self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cuda', 'cuda:N', 'cpu', or 'auto'")
        if self.temperature == 0.0 and self.do_sample:
            object.__setattr__(self, "do_sample", False)
