import json
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a strict evaluator of code review comments.\n"
    "Evaluate whether the candidate review comment is useful for the given code or code change.\n"
    "Judge correctness, grounding, actionability, specificity, and conciseness.\n"
    "The reference comment is guidance only; do not require exact wording.\n"
    "Penalize hallucinated issues, unsupported claims, vague advice, restating the code, and unnecessary verbosity.\n"
    "Output exactly this compact JSON and nothing else:\n"
    '{"c":<1-5>,"g":<1-5>,"a":<1-5>,"s":<1-5>,"q":<1-5>}'
)


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    marker = "\n...[truncated middle]...\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars]

    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def build_user_prompt(code_or_diff, prediction, reference, max_code_chars: int) -> str:
    code_text = truncate_middle(_as_text(code_or_diff), max_code_chars).replace("```", "`\\`\\`")
    prediction_text = json.dumps(_as_text(prediction), ensure_ascii=False)
    reference_text = json.dumps(_as_text(reference), ensure_ascii=False)

    return (
        "Code or code change:\n"
        f"```{code_text}```\n\n"
        "Candidate review comment:\n"
        f"{prediction_text}\n\n"
        "Reference human review comment:\n"
        f"{reference_text}\n\n"
        "Scoring:\n"
        "c = technical correctness\n"
        "g = groundedness in the visible code\n"
        "a = actionability\n"
        "s = specificity\n"
        "q = conciseness\n"
        "1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent"
    )


def build_messages(code_or_diff, prediction, reference, max_code_chars: int) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(code_or_diff, prediction, reference, max_code_chars)},
    ]
