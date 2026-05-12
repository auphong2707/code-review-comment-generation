# Commands For LLM-as-Judge Tables 1-5

Run from the repository root. These commands score the saved result artifacts
used by proposal Tables 1-5; they do not rerun model inference.

## Full Run

```bash
python tools/update_llm_judge_tables.py \
  --judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct \
  --judge-local-files-only \
  --judge-k 5
```

The local judge model must already exist at:

```text
assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
```

## Smoke Test Without Real Model

```bash
python tools/update_llm_judge_tables.py --mock --limit-jobs 1 --limit-samples 1
```

## Outputs

```text
assets/models/llm-as-judge/table-results/manifest.jsonl
assets/models/llm-as-judge/table-results/scored/
assets/models/llm-as-judge/table-results/summary.csv
```
