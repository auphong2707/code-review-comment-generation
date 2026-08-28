"""Generic HF chat-model judge backend (phase 05, D-019).

Same rubric prompt, cache and scoring as QwenJudge; only the generation layer
differs so that non-Qwen instruction models (e.g. Llama-3.1-8B-Instruct)
behave correctly:

- the rendered chat template is tokenised with add_special_tokens=False so the
  template's BOS is not doubled;
- eos_token_id is the model's generation-config list (Llama has several),
  falling back to the tokenizer's;
- top_k / repetition_penalty are forwarded when set (D-019 pins them to the
  Qwen judge's effective settings);
- when config.seed is set, torch.manual_seed(seed + run_index) runs before
  each batch — reproducible for identical batching; the cache is the replay
  mechanism.
"""

from .qwen_judge import QwenJudge


class HFChatJudge(QwenJudge):
    def _render_prompt(self, messages) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _eos_token_ids(self):
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return eos if isinstance(eos, list) else [eos]

    def _generate_from_prompts(self, prompts: list[str], *, deterministic: bool = False,
                               run_index: int = 0) -> list[str]:
        if self.config.seed is not None:
            self._torch.manual_seed(self.config.seed + run_index)

        inputs = self.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(self._device())
        input_length = inputs["input_ids"].shape[1]

        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False if deterministic else self.config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._eos_token_ids(),
        }
        if not deterministic and self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                kwargs["top_k"] = self.config.top_k
        if self.config.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.config.repetition_penalty

        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)

        outputs = []
        for row in generated:
            output_ids = row[input_length:]
            outputs.append(self.tokenizer.decode(output_ids, skip_special_tokens=True).strip())
        return outputs
