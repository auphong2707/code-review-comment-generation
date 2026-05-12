import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional


def make_cache_key(record: Mapping[str, Any], config, run_index: int) -> str:
    payload = {
        "prompt_version": config.prompt_version,
        "model_path": config.model_path,
        "code_or_diff": record.get("code_or_diff", ""),
        "prediction": record.get("prediction", ""),
        "reference": record.get("reference", ""),
        "run_index": run_index,
        "generation_settings": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        },
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class JsonlCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._entries: Dict[str, Dict[str, Any]] = {}
        if not path or not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = entry.get("key")
                if key:
                    self._entries[key] = entry

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.path:
            return

        entry = {"key": key, **value}
        self._entries[key] = entry
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
