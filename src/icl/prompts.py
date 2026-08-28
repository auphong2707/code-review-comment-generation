"""ICL prompt builders (phase 04): pure functions extracted from
icl_inference.py, with the dataset an explicit argument instead of a module
global.

Prompt versions (D-023, KI-014):
- v1 (legacy, byte-identical): the causal test turn always says "Code:" even
  when the demos say "Code changes:" (the CodeReviewer header bug).
- v2: the test turn uses the dataset's prompt_suffix, matching the demos and
  paper Fig. 4. The seq2seq prompt has no such bug; v1 and v2 render the same.
"""

from src.utils import constants
from src.utils.helper import extract_input_target

SYSTEM_PROMPT = (
    "You are a code reviewer. Output exactly ONE concise, actionable comment "
    "on the code. No restating, no extra context, no pleasantries. "
    "Recommend to use up to 60 words. (longer if needed)"
)


def build_icl_prompt_seq2seq(examples, test_code, dataset_name, prompt_version="v1"):
    """Plain-text prompt for CodeT5/PLBART (identical across versions)."""
    prompt_parts = []
    for example in examples:
        code, review = extract_input_target(example, dataset_name)
        prompt_parts.append(f"Code: {code}\nReview: {review}")
    prompt_parts.append(f"Code: {test_code}\nReview:")
    return "\n\n".join(prompt_parts)


def build_icl_prompt_causal(examples, test_code, dataset_name, prompt_version="v1"):
    """Chat messages for the decoder-only models (zero-shot: system + 1 user turn)."""
    suffix = constants.get_dataset_config(dataset_name)["prompt_suffix"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples:
        code, review = extract_input_target(example, dataset_name)
        messages.extend([
            {"role": "user", "content": f"Code{suffix}:\n```{code}```\nReview comment?"},
            {"role": "assistant", "content": review.strip()},
        ])
    test_suffix = suffix if prompt_version == "v2" else ""
    messages.append({
        "role": "user",
        "content": f"Code{test_suffix}:\n```{test_code}```\nReview comment?",
    })
    return messages
