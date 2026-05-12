from typing import Any, Iterable, Mapping

from .cache import JsonlCache, make_cache_key
from .config import JudgeConfig
from .parser import parse_judge_output
from .prompt import build_messages
from .scorer import aggregate_scores, score_components


CODE_COLUMNS = ("code_or_diff", "code", "patch", "diff", "source", "input", "function", "code_tokens")
PREDICTION_COLUMNS = ("prediction", "pred", "generated_review", "generated_comment", "output")
REFERENCE_COLUMNS = ("reference", "label", "target", "ground_truth", "review", "msg", "docstring")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _select_column(record: Mapping[str, Any], explicit: str, candidates, label: str) -> str:
    if explicit:
        if explicit not in record:
            raise ValueError(
                f"missing explicit {label} column '{explicit}'; available columns: {sorted(record.keys())}"
            )
        return explicit

    for column in candidates:
        if column in record:
            return column

    raise ValueError(
        f"could not detect {label} column; expected one of {list(candidates)}; "
        f"available columns: {sorted(record.keys())}"
    )


def normalize_record(record: Mapping[str, Any], config: JudgeConfig) -> dict:
    code_column = _select_column(record, config.code_column, CODE_COLUMNS, "code/diff")
    prediction_column = _select_column(
        record, config.prediction_column, PREDICTION_COLUMNS, "prediction"
    )
    reference_column = _select_column(record, config.reference_column, REFERENCE_COLUMNS, "reference")

    return {
        "code_or_diff": _to_text(record[code_column]),
        "prediction": _to_text(record[prediction_column]),
        "reference": _to_text(record[reference_column]),
    }


class MockJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config

    def score_record(self, record: Mapping[str, Any]) -> dict:
        normalize_record(record, self.config)
        score = score_components({"c": 4, "g": 4, "a": 4, "s": 4, "q": 4})
        return aggregate_scores(
            [score for _ in range(self.config.k)],
            k=self.config.k,
            parse_errors=0,
            model_path=self.config.model_path,
            prompt_version=self.config.prompt_version,
            penalty_lambda=self.config.penalty_lambda,
        )


class QwenJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.cache = JsonlCache(config.cache_path)
        self.model = None
        self.tokenizer = None
        self._torch = None

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        self.model.eval()
        self._torch = torch

    def _device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _render_prompt(self, messages) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    def _generate(self, record: Mapping[str, Any]) -> str:
        self._load()
        messages = build_messages(
            record["code_or_diff"],
            record["prediction"],
            record["reference"],
            self.config.max_code_chars,
        )
        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device())
        input_length = inputs["input_ids"].shape[1]

        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p

        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)

        generated = generated[0][input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_record(self, record: Mapping[str, Any]) -> dict:
        normalized = normalize_record(record, self.config)
        run_scores = []
        parse_errors = 0

        for run_index in range(self.config.k):
            key = make_cache_key(normalized, self.config, run_index)
            cached = self.cache.get(key)
            if cached and cached.get("ok"):
                run_scores.append(cached["run_score"])
                continue

            try:
                raw_output = self._generate(normalized)
                component_scores = parse_judge_output(raw_output)
                run_score = score_components(component_scores)
            except ValueError:
                parse_errors += 1
                continue

            self.cache.set(
                key,
                {
                    "ok": True,
                    "run_score": run_score,
                    "component_scores": component_scores,
                    "raw_output": raw_output,
                },
            )
            run_scores.append(run_score)

        return aggregate_scores(
            run_scores,
            k=self.config.k,
            parse_errors=parse_errors,
            model_path=self.config.model_path,
            prompt_version=self.config.prompt_version,
            penalty_lambda=self.config.penalty_lambda,
        )


def score_record(record: Mapping[str, Any], config: JudgeConfig) -> dict:
    judge = MockJudge(config) if config.mock else QwenJudge(config)
    return judge.score_record(record)


def score_records(records: Iterable[Mapping[str, Any]], config: JudgeConfig) -> list[dict]:
    judge = MockJudge(config) if config.mock else QwenJudge(config)
    return [judge.score_record(record) for record in records]
