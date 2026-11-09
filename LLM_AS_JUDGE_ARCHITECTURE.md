# LLM-as-Judge Architecture Note

## Abstract

This note describes the LLM-as-Judge extension for evaluating generated code review comments in the PEFT, FMFT, and ICL study. The extension adds a rubric-based qualitative evaluation signal beside the existing automatic metrics: BLEU-4, Exact Match, and BERTScore. It is designed as an optional post-hoc evaluator, not as a replacement for existing metrics and not as part of model training or inference.

The judge is pointwise, reference-aware, and grounded in the visible code or code change. For each generated review comment, a local Qwen-compatible instruction model evaluates whether the comment is technically correct, grounded in the provided code, actionable, specific, and concise. The model returns compact machine-parseable JSON, which is transformed into scalar judge scores and uncertainty-aware aggregate columns for later analysis.

This document is intended as a methodology reference for paper writing. It does not report results.

## Motivation

Code review comment generation is difficult to evaluate with lexical overlap alone. A generated comment can be useful even when it uses different wording from the human reference, and a comment can have high overlap while still being vague, unsupported, or unhelpful. This is especially important when comparing PEFT, FMFT, and ICL settings, because different adaptation strategies may produce comments with different styles, verbosity, and levels of practical usefulness.

The existing metrics remain necessary:

- BLEU-4 captures n-gram overlap with the reference.
- Exact Match identifies strict string-level matches.
- BERTScore captures embedding-based semantic similarity.

However, these metrics do not directly evaluate whether a generated review comment is a good code review comment. The LLM-as-Judge layer adds a complementary signal that targets review utility:

- whether the comment is technically correct,
- whether it is grounded in the visible code or diff,
- whether it gives a developer a clear action,
- whether it is specific enough to locate the issue,
- whether it is concise rather than verbose or repetitive.

The judge should therefore be interpreted as an additional evaluation lens. It supports richer discussion of model behavior, but it should not be treated as a sole authority or as a substitute for human evaluation.

## Evaluation Architecture

The judge is implemented as an isolated post-processing module. It does not alter dataset formats, training logic, model checkpoints, inference prompts, or the semantics of existing metrics. The high-level flow is:

1. A generation method produces a candidate review comment.
2. Existing automatic metrics are computed normally.
3. If judge mode is explicitly enabled, the judge module scores each result row.
4. Judge columns are appended to the result artifact.
5. Existing columns remain unchanged.

The evaluated generation methods are the paper's existing families:

- FMFT: full-model fine-tuning.
- PEFT: parameter-efficient fine-tuning variants such as LoRA, QLoRA, IA3, or related adapter methods used in the experiments.
- ICL: in-context learning inference without fine-tuning the evaluated model.

The judge model is separate from the evaluated generator models. Its default local path is:

```text
assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct
```

The implementation is configured to load a local Hugging Face-compatible model only when judge mode is enabled. This keeps normal PEFT, FMFT, and ICL runs unchanged when the judge is disabled. The default prompt version is:

```text
qwen_judge_v1
```

## Judge Prompting Design

The judge uses a pointwise prompt. Each sample is scored independently from other samples, which avoids pairwise ranking dependencies and makes the judge columns easy to join with existing result tables.

The judge input contains three fields:

- code or code change,
- candidate generated review comment,
- reference human review comment.

The reference is used as guidance, not as a required wording target. This matters because review comments often have multiple valid phrasings. A candidate can receive a high score if it identifies a valid issue and gives useful guidance, even when it is not lexically close to the reference.

The system instruction asks the judge to evaluate usefulness for the given code or code change and to penalize unsupported claims, hallucinated issues, vague advice, restating the code, and unnecessary verbosity.

The required model output is exactly compact JSON:

```json
{"c":4,"g":4,"a":3,"s":4,"q":5}
```

The fields are:

- `c`: technical correctness.
- `g`: groundedness in the visible code or diff.
- `a`: actionability.
- `s`: specificity.
- `q`: conciseness.

Each field is an integer from 1 to 5:

- 1 = very poor.
- 2 = poor.
- 3 = acceptable.
- 4 = good.
- 5 = excellent.

Long code inputs are truncated from the middle when needed, preserving the beginning and end of the input. The purpose is to fit the prompt into the judge model context while retaining surrounding context and changed regions as much as possible. Backticks and quotes are escaped so the prompt remains structurally stable.

## Rubric Dimensions

The judge rubric is designed around practical code review quality rather than pure textual similarity.

| Field | Dimension | Interpretation |
| --- | --- | --- |
| `c` | Technical correctness | Whether the comment identifies a real technical issue or improvement and avoids false claims. |
| `g` | Groundedness | Whether the comment is supported by the visible code or diff rather than hallucinated context. |
| `a` | Actionability | Whether the comment gives the developer a clear next step or fix direction. |
| `s` | Specificity | Whether the comment is precise enough to locate the relevant code behavior or issue. |
| `q` | Conciseness | Whether the comment is focused and avoids unnecessary verbosity. |

Correctness and groundedness receive the largest weights because an incorrect or unsupported review comment is actively harmful, even if it is fluent. Actionability receives the next largest weight because useful review comments should help the author improve the code. Specificity and conciseness are weighted lower but still affect the final score because vague or bloated comments reduce review quality.

## Scoring and Aggregation

For one valid judge output, the component scores are converted into a weighted scalar score:

```text
score = 0.35*c + 0.25*g + 0.20*a + 0.10*s + 0.10*q
```

Hard caps are applied after the weighted score:

```text
if c <= 2 or g <= 2: score = min(score, 2.5)
if a <= 2:           score = min(score, 3.5)
if q <= 2:           score = min(score, 4.0)
```

These caps encode the assumption that certain failures should limit the maximum possible quality score:

- Incorrect or ungrounded comments cannot be high quality.
- Non-actionable comments are limited even if they are plausible.
- Overly verbose or unfocused comments are penalized but not as severely as incorrect comments.

Each sample is judged `k` times. The default setting is:

```text
k = 5
penalty_lambda = 0.25
temperature = 0.3
top_p = 0.9
max_new_tokens = 32
do_sample = true
```

For deterministic smoke checks, the judge can use:

```text
k = 1
temperature = 0.0
do_sample = false
```

Across the valid judge runs for a sample, the module computes:

```text
qwen_judge_mean       = mean(run_scores)
qwen_judge_median     = median(run_scores)
qwen_judge_std        = population_std(run_scores)
qwen_judge_agreement  = modal_rounded_score_count / valid_runs
qwen_judge_score      = qwen_judge_mean - penalty_lambda * qwen_judge_std
qwen_judge_norm       = (qwen_judge_score - 1) / 4
```

The final `qwen_judge_score` is uncertainty-aware: unstable judge outputs receive a small penalty through the standard deviation term. The normalized score maps the 1 to 5 scale into an approximate 0 to 1 range, which can be convenient for comparison with other metrics.

The required appended output columns are:

```text
qwen_judge_score
qwen_judge_mean
qwen_judge_median
qwen_judge_std
qwen_judge_agreement
qwen_judge_norm
qwen_judge_k
qwen_judge_valid_runs
qwen_judge_parse_errors
qwen_judge_model
qwen_judge_prompt_version
```

If all judge runs for a sample fail to parse, numeric judge fields are set to missing values, valid runs are set to zero, and the original evaluation row is preserved. This prevents a single judge failure from invalidating a full experiment artifact.

## Integration With PEFT, FMFT, and ICL Experiments

The judge is intended to run after the evaluated model has already generated comments. It should not influence the generated predictions. For FMFT and PEFT experiments, the judge is attached to re-evaluation outputs after the trained checkpoint has been evaluated. For ICL experiments, the judge is attached after ICL inference has produced predictions and existing metric outputs.

This separation supports fair comparison:

- PEFT, FMFT, and ICL methods use their existing inference procedures.
- BLEU-4, Exact Match, and BERTScore keep their original definitions.
- The judge adds a new family of columns rather than replacing existing columns.
- Judge mode is optional and disabled by default.

For paper analysis, the judge columns can support several result views once experiments have been run:

- model-level average judge score by method,
- comparison between overlap metrics and judge score,
- stability analysis using `qwen_judge_std`,
- agreement analysis using `qwen_judge_agreement`,
- normalized score comparison using `qwen_judge_norm`,
- qualitative discussion of examples where overlap metrics and judge scores diverge.

These uses should be framed as analysis opportunities until actual experimental results are available.

## Reliability Considerations

The judge is designed to reduce avoidable noise and parsing failures:

- The output schema is compact JSON with fixed keys.
- The parser accepts valid JSON and rejects missing fields, non-integer values, and out-of-range values.
- Generation is short, with `max_new_tokens = 32`, to discourage long explanations.
- Multiple judge runs estimate output variability.
- The standard deviation penalty reduces final scores for unstable samples.
- The agreement column records how often rounded run scores converge.
- The model is loaded once per process when judge mode is enabled.
- A cache can avoid recomputing previously judged sample and setting combinations.

The cache key is based on stable inputs such as prompt version, judge model path, sample content, run index, and generation settings. This is important because sampled judge runs are part of the measurement process; changing generation settings should produce different cache entries.

## Threats to Validity

LLM-as-Judge evaluation has limitations that should be acknowledged in the paper.

First, the judge may share biases with the generator family if both are instruction-tuned code models. A high judge score is not equivalent to verified human preference. The judge should be treated as a structured proxy for review usefulness.

Second, the reference comment may bias the judge toward the human annotation even though the prompt states that the reference is guidance only. This is intentional but should be described as reference-aware evaluation, not reference-free evaluation.

Third, the visible code or diff may not contain all context needed to determine correctness. The groundedness dimension partially addresses this by penalizing claims unsupported by the provided input, but it cannot recover missing repository context.

Fourth, sampled judge outputs can vary. The k-run design, standard deviation, and agreement fields expose this variability instead of hiding it. Results should report or discuss stability when using judge scores as a comparison signal.

Fifth, rubric weights and hard caps are methodological choices. They emphasize correctness and groundedness over style, which is appropriate for code review utility, but other studies might choose different weights.

## Paper-Writing Guidance

The LLM-as-Judge method can be described in the evaluation methodology section as follows:

> In addition to BLEU-4, Exact Match, and BERTScore, we include an optional LLM-as-Judge evaluation to estimate the practical usefulness of generated review comments. The judge is a local Qwen-compatible instruction model prompted with the code or code change, the generated review comment, and the human reference comment. The reference is used as guidance rather than as an exact wording target. The judge returns compact JSON scores for technical correctness, groundedness, actionability, specificity, and conciseness, each on a 1 to 5 scale.

The scoring paragraph can be adapted as:

> We convert rubric scores into a weighted scalar score using `0.35*c + 0.25*g + 0.20*a + 0.10*s + 0.10*q`, where correctness and groundedness receive the largest weights. We apply hard caps when correctness, groundedness, actionability, or conciseness are poor, preventing fluent but incorrect or unsupported comments from receiving high scores. Each sample is judged multiple times, and the final score subtracts a small standard-deviation penalty from the mean score to account for judge instability.

The limitations paragraph can be adapted as:

> LLM-as-Judge scores are complementary to lexical and embedding metrics, not replacements for them. They provide a structured proxy for review usefulness but remain model-based judgments. We therefore analyze judge scores alongside BLEU-4, Exact Match, and BERTScore, and we expose judge variability through standard deviation and agreement columns.

Until experiments are complete, the paper should avoid claims such as "method X is better according to the judge." The architecture supports those comparisons, but the claims must be based on actual computed result tables.
