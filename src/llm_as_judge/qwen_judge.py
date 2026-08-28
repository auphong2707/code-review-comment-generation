from typing import Any, Iterable, Mapping

from .cache import JsonlCache, make_cache_key
from .config import JudgeConfig
from .parser import parse_judge_output
from .prompt import build_messages
from .scorer import aggregate_scores, score_components


CODE_COLUMNS = ("code_or_diff", "code", "patch", "diff", "source", "input", "function", "code_tokens")
PREDICTION_COLUMNS = ("prediction", "pred", "generated_review", "generated_comment", "output")
REFERENCE_COLUMNS = ("reference", "label", "target", "ground_truth", "review", "msg", "docstring")

MOCK_RAW_OUTPUT = '{"c":4,"g":4,"a":4,"s":4,"q":4}'
REPAIR_INSTRUCTION = (
    "Your previous output was not the required compact JSON. "
    'Respond with exactly {"c":<1-5>,"g":<1-5>,"a":<1-5>,"s":<1-5>,"q":<1-5>} '
    "using integer values from 1 to 5 and no other text."
)


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
        components = {"c": 4, "g": 4, "a": 4, "s": 4, "q": 4}
        score = score_components(components)
        prefix = self.config.column_prefix
        result = aggregate_scores(
            [score for _ in range(self.config.k)],
            k=self.config.k,
            parse_errors=0,
            model_path=self.config.model_path,
            prompt_version=self.config.prompt_version,
            penalty_lambda=self.config.penalty_lambda,
            column_prefix=prefix,
        )
        if self.config.emit_debug_columns:
            result[f"{prefix}raw_outputs"] = [MOCK_RAW_OUTPUT] * self.config.k
            result[f"{prefix}component_scores"] = [dict(components) for _ in range(self.config.k)]
        return result

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

    def _generate_from_prompts(self, prompts: list[str], *, deterministic: bool = False,
                               run_index: int = 0) -> list[str]:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self._device())
        input_length = inputs["input_ids"].shape[1]

        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False if deterministic else self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not deterministic and self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p

        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)

        outputs = []
        for row in generated:
            output_ids = row[input_length:]
            outputs.append(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
        return outputs

    def _generate_batch(self, records: list[Mapping[str, Any]], run_index: int = 0) -> list[str]:
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
        return self._generate_from_prompts(prompts, run_index=run_index)

    def _generate_repair_batch(self, items: list[tuple[Mapping[str, Any], str]]) -> list[str]:
        """One deterministic re-prompt per parse failure (spec: repair retry)."""
        self._load()
        prompts = []
        for record, previous_output in items:
            messages = list(
                build_messages(
                    record["code_or_diff"],
                    record["prediction"],
                    record["reference"],
                    self.config.max_code_chars,
                )
            )
            messages.append({"role": "assistant", "content": previous_output})
            messages.append({"role": "user", "content": REPAIR_INSTRUCTION})
            prompts.append(self._render_prompt(messages))
        return self._generate_from_prompts(prompts, deterministic=True)

    def _generate(self, record: Mapping[str, Any]) -> str:
        return self._generate_batch([record])[0]

    def score_record(self, record: Mapping[str, Any]) -> dict:
        return self.score_records([record])[0]

    def score_records(self, records: Iterable[Mapping[str, Any]]) -> list[dict]:
        normalized_records = [normalize_record(record, self.config) for record in records]
        run_scores = [[] for _ in normalized_records]
        parse_errors = [0 for _ in normalized_records]
        raw_history = [[] for _ in normalized_records]
        component_history = [[] for _ in normalized_records]

        if not normalized_records:
            return []

        def accept(record_index, key, raw_output, component_scores, run_score):
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
            component_history[record_index].append(component_scores)

        record_indices = list(range(len(normalized_records)))
        for run_index in range(self.config.k):
            uncached = []
            for record_index in record_indices:
                normalized = normalized_records[record_index]
                key = make_cache_key(normalized, self.config, run_index)
                cached = self.cache.get(key)
                if cached and cached.get("ok"):
                    run_scores[record_index].append(cached["run_score"])
                    raw_history[record_index].append(cached.get("raw_output"))
                    component_history[record_index].append(cached.get("component_scores"))
                else:
                    uncached.append((record_index, key, normalized))

            for start in range(0, len(uncached), self.config.batch_size):
                batch = uncached[start:start + self.config.batch_size]
                batch_records = [item[2] for item in batch]
                raw_outputs = self._generate_batch(batch_records, run_index=run_index)

                repair_items = []
                for (record_index, key, normalized), raw_output in zip(batch, raw_outputs):
                    raw_history[record_index].append(raw_output)
                    try:
                        component_scores = parse_judge_output(raw_output)
                        run_score = score_components(component_scores)
                    except ValueError:
                        if self.config.repair_retry:
                            repair_items.append((record_index, key, normalized, raw_output))
                        else:
                            parse_errors[record_index] += 1
                        continue

                    accept(record_index, key, raw_output, component_scores, run_score)

                if repair_items:
                    repaired_outputs = self._generate_repair_batch(
                        [(normalized, raw) for _, _, normalized, raw in repair_items]
                    )
                    for (record_index, key, _, _), repaired_raw in zip(repair_items, repaired_outputs):
                        raw_history[record_index].append(repaired_raw)
                        try:
                            component_scores = parse_judge_output(repaired_raw)
                            run_score = score_components(component_scores)
                        except ValueError:
                            parse_errors[record_index] += 1
                            continue

                        accept(record_index, key, repaired_raw, component_scores, run_score)

        results = []
        prefix = self.config.column_prefix
        for index, scores in enumerate(run_scores):
            result = aggregate_scores(
                scores,
                k=self.config.k,
                parse_errors=parse_errors[index],
                model_path=self.config.model_path,
                prompt_version=self.config.prompt_version,
                penalty_lambda=self.config.penalty_lambda,
                column_prefix=prefix,
            )
            if self.config.emit_debug_columns:
                result[f"{prefix}raw_outputs"] = raw_history[index]
                result[f"{prefix}component_scores"] = component_history[index]
            results.append(result)
        return results


def make_judge(config: JudgeConfig):
    """Single backend dispatch (phase 05): mock=True always wins."""
    if config.mock or config.backend == "mock":
        return MockJudge(config)
    if config.backend == "hf-chat":
        from .hf_chat_judge import HFChatJudge

        return HFChatJudge(config)
    return QwenJudge(config)


def score_record(record: Mapping[str, Any], config: JudgeConfig) -> dict:
    return make_judge(config).score_record(record)


def score_records(records: Iterable[Mapping[str, Any]], config: JudgeConfig) -> list[dict]:
    return make_judge(config).score_records(records)
