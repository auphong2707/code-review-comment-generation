# Commands For LLM-as-Judge Tables 1-5

Run from the repository root. These commands score the saved result artifacts
used by proposal Tables 1-5; they do not rerun model inference. The runner
prints phase logs and progress bars for each model-method-dataset job.

## Split Full Run

Use four smaller runs so logs are easier to inspect and rerun.

```bash
COMMON_ARGS="--judge-model-path assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct --judge-local-files-only --judge-device cuda --judge-batch-size 3 --judge-k 5 --log-level INFO"

python tools/update_llm_judge_tables.py --job-group trans-review-finetuned $COMMON_ARGS
python tools/update_llm_judge_tables.py --job-group code-reviewer-finetuned $COMMON_ARGS
python tools/update_llm_judge_tables.py --job-group trans-review-icl $COMMON_ARGS
python tools/update_llm_judge_tables.py --job-group code-reviewer-icl $COMMON_ARGS
python tools/update_llm_judge_tables.py --job-group all --summary-only --log-level INFO
```

The local judge model must already exist at:

```text
assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
```

## Smoke Test Without Real Model

```bash
python tools/update_llm_judge_tables.py \
  --mock \
  --job-group trans-review-finetuned \
  --limit-jobs 1 \
  --limit-samples 3 \
  --judge-batch-size 3
```

## Outputs

```text
assets/models/llm-as-judge/table-results/manifest_<job-group>.jsonl
assets/models/llm-as-judge/table-results/summary_<job-group>.csv
assets/models/llm-as-judge/table-results/manifest.jsonl
assets/models/llm-as-judge/table-results/summary.csv
assets/models/llm-as-judge/table-results/scored/
```
