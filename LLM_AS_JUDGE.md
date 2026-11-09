# LLM-as-Judge

This extension adds a Qwen-based rubric judge alongside BLEU-4, Exact Match, and BERTScore for review comment generation. It is complementary to string/embedding metrics and should not be treated as a replacement for BLEU or BERTScore.

## Model Path

The default local Hugging Face model path is:

```text
assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
```

Pass another local path with `--judge-model-path` when needed.

## Required Inputs

Each judged sample needs:

- code/diff input: auto-detected from columns such as `code_or_diff`, `patch`, or `code`, or set explicitly with `--code-column` for the standalone CLI and `--judge-code-column` for integrated scripts.
- prediction: auto-detected from `prediction`, or set explicitly with `--prediction-column` / `--judge-prediction-column`.
- reference: auto-detected from `reference` or `msg`, or set explicitly with `--reference-column` / `--judge-reference-column`.

## Standalone CLI

```bash
python -m src.llm_as_judge.cli input.jsonl output.jsonl \
  --judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct \
  --judge-local-files-only
```

Mock smoke test without loading the real model:

```bash
python -m src.llm_as_judge.cli input.jsonl output.jsonl \
  --mock --judge-k 1 --judge-temperature 0.0 --judge-no-sample
```

## Integrated Usage

Re-evaluate a trained model with judge columns appended:

```bash
python src/re_eval_model.py \
  --model_directory ./results-cc/code-t5/codet5_fmft_official_0.0001 \
  --model_name codet5 \
  --model_method fmft \
  --llm-judge \
  --judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
```

Run ICL inference with judge columns appended:

```bash
python src/icl_inference.py \
  --model_name qwen25 \
  --num_examples 3 \
  --selection_method random \
  --llm-judge \
  --judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
```

Use `--judge-output-path` to write judged output to a separate JSON file.

## Score Fields

- `qwen_judge_score`: final score, computed as `qwen_judge_mean - penalty_lambda * qwen_judge_std`.
- `qwen_judge_mean`: average score across valid judge runs.
- `qwen_judge_std`: population standard deviation across valid judge runs.
- `qwen_judge_agreement`: frequency of the modal rounded score divided by valid runs.

Other columns include `qwen_judge_median`, `qwen_judge_norm`, `qwen_judge_k`, `qwen_judge_valid_runs`, `qwen_judge_parse_errors`, `qwen_judge_model`, and `qwen_judge_prompt_version`.
