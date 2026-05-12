import json
import re
from typing import Dict


JUDGE_KEYS = ("c", "g", "a", "s", "q")


def _first_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{[^{}]*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("judge output does not contain a JSON object")
    return match.group(0)


def parse_judge_output(text: str) -> Dict[str, int]:
    try:
        parsed = json.loads(_first_json_object("" if text is None else str(text)))
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("judge output must be a JSON object")

    expected = set(JUDGE_KEYS)
    actual = set(parsed)
    if actual != expected:
        raise ValueError(f"judge output keys must be exactly {sorted(expected)}; got {sorted(actual)}")

    scores: Dict[str, int] = {}
    for key in JUDGE_KEYS:
        value = parsed[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"judge field {key} must be an integer")
        if value < 1 or value > 5:
            raise ValueError(f"judge field {key} must be in [1, 5]")
        scores[key] = value
    return scores
