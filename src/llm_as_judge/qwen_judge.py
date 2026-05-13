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

    def score_records(self, records: Iterable[Mapping[str, Any]]) -> list[dict]:
        return [self.score_record(record) for record in records]


class QwenJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.cache = JsonlCache(config.cache_path)
        self.model = None
        self.tokenizer = None
        self._torch = None
        self._target_device = None

    def _device_map(self, torch):
        if self.config.device == "auto":
            return "auto"
        if self.config.device == "cpu":
            self._target_device = torch.device("cpu")
            return {"": "cpu"}

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLM-as-Judge is configured to run on CUDA, but CUDA is not available. "
                "Use a GPU runtime or set device='cpu' explicitly for debugging."
            )

        target = "cuda:0" if self.config.device == "cuda" else self.config.device
        if ":" in target:
            device_index = int(target.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LLM-as-Judge requested {target}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible."
                )
        self._target_device = torch.device(target)
        return {"": target}

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
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = self._device_map(torch)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=self.config.local_files_only,
        )
        self.model.eval()
        self._torch = torch

    def _device(self):
        if self._target_device is not None:
            return self._target_device
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

    def _generate_batch(self, records: list[Mapping[str, Any]]) -> list[str]:
        self._load()
        prompts = [
            self._render_prompt(
                build_messages(
                    record["code_or_diff"],
                    record["prediction"],
                    record["reference"],
                    self.config.max_code_chars,
                )
            )
            for record in records
        ]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self._device())
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

        outputs = []
        for row in generated:
            output_ids = row[input_length:]
            outputs.append(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
        return outputs

    def _generate(self, record: Mapping[str, Any]) -> str:
        return self._generate_batch([record])[0]

    def score_record(self, record: Mapping[str, Any]) -> dict:
        return self.score_records([record])[0]

    def score_records(self, records: Iterable[Mapping[str, Any]]) -> list[dict]:
        normalized_records = [normalize_record(record, self.config) for record in records]
        run_scores = [[] for _ in normalized_records]
        parse_errors = [0 for _ in normalized_records]

        if not normalized_records:
            return []

        record_indices = list(range(len(normalized_records)))
        for run_index in range(self.config.k):
            uncached = []
            for record_index in record_indices:
                normalized = normalized_records[record_index]
                key = make_cache_key(normalized, self.config, run_index)
                cached = self.cache.get(key)
                if cached and cached.get("ok"):
                    run_scores[record_index].append(cached["run_score"])
                else:
                    uncached.append((record_index, key, normalized))

            for start in range(0, len(uncached), self.config.batch_size):
                batch = uncached[start:start + self.config.batch_size]
                batch_records = [item[2] for item in batch]
                raw_outputs = self._generate_batch(batch_records)

                for (record_index, key, _), raw_output in zip(batch, raw_outputs):
                    try:
                        component_scores = parse_judge_output(raw_output)
                        run_score = score_components(component_scores)
                    except ValueError:
                        parse_errors[record_index] += 1
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
                    run_scores[record_index].append(run_score)

        return [
            aggregate_scores(
                scores,
                k=self.config.k,
                parse_errors=parse_errors[index],
                model_path=self.config.model_path,
                prompt_version=self.config.prompt_version,
                penalty_lambda=self.config.penalty_lambda,
            )
            for index, scores in enumerate(run_scores)
        ]


def score_record(record: Mapping[str, Any], config: JudgeConfig) -> dict:
    judge = MockJudge(config) if config.mock else QwenJudge(config)
    return judge.score_record(record)


def score_records(records: Iterable[Mapping[str, Any]], config: JudgeConfig) -> list[dict]:
    judge = MockJudge(config) if config.mock else QwenJudge(config)
    return judge.score_records(records)
