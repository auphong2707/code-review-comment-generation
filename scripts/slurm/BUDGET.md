# A5000 planning budget — 2026-09-11

No A5000 run has been measured. These are **assumption-based planning scenarios**,
not benchmark results, confidence intervals, deadlines or guaranteed bounds.
For one serial A5000, provision roughly **1,260–6,460 occupied hours (53–269 days)**
for the default queue, or **1,565–7,980 hours (65–333 days)** with every supported
optional block. Queue waits, transfers and operator resubmission delays are additional.
Actual time can fall outside these ranges; calibrate before reserving the full campaign.
Six sbatch files do not imply six allocations or an 18-day campaign.

| Phase | Default occupied hours | All supported options, hours |
|---|---:|---:|
| 00–07 qualification | 4–24 | 4–24 |
| A: seeds / optional HP grid | 413–2,142 | 454–2,373 |
| B: DoRA + controls | 153–829 | 319–1,604 |
| C: ICL | 191–1,147 | 225–1,355 |
| D: 14B QLoRA, conditional | 142–525 | 142–525 |
| J: both judges | 359–1,793 | 420–2,100 |

All options means A `RUN_P1=1`, B `RUN_P2=1`, C `RUN_P2=1`, **each on its own
submission**. D keeps defaults. No estimate includes unavailable GaLore, 7B FMFT,
ordinary 14B LoRA or 14B ICL; there is no feasible all-original-H100-jobs total for
this profile. If a remaining workload fails admission, mark its evidence unavailable
and recompute the selected scope rather than silently substituting another method.

## Work counted

Read-only corpus counts: TR train/validation/test = 13,756 / 1,719 / 1,719;
CR = 117,739 / 10,319 / 10,169. Scripts, registry and archived protocol determine
model, epochs and subsets. Every epoch includes validation generation, and every
completed run includes test generation. Accumulation preserves effective batch but
still processes microbatch-1 forwards/backwards.

Default A/B/C/D schedule 144 commands, producing **139 unique outputs**: 54 A,
11 additional B (five controls reuse A), 66 C and eight D (six sweeps, two officials).
All supported options schedule 170 commands and produce **163 unique outputs**:
62 A, 15 additional B, 78 C, eight D. This deduplicates A's two baseline HP-grid
commands as well as the five A/B controls. D sweep test rows use the same prefix
fraction as training/validation; they are excluded from judging.

Default revision judging covers 93 TR and 40 CR artifacts = 566,627 rows;
all options cover 107 TR and 50 CR = 692,383 rows. At k=3, cold-cache Qwen budgets
are 1,699,881 / 2,077,149 calls. Llama adds 27 TR + 27 CR legacy artifacts,
uses 2,000 rows per CR artifact, and budgets 1,020,840 / 1,153,038 calls.
Duplicate predictions/cache hits can reduce these call counts; no reduction is
assumed. Repair retries stay off. Legacy inputs were counted from the frozen
54 JSONL files, read-only.

## Explicit assumptions and formula

The following seconds are analyst-selected sensitivity inputs, **not observed A5000
throughput**. They deliberately span different output lengths, method costs and
utilization. One training microbatch is one sample; average optimizer work is
amortized into that rate. Generation means one complete beam-3 response including
prompt processing, with early EOS allowed and maximum 256 output tokens.

| Model family | Train seconds/microbatch | Validation/test seconds/row | ICL seconds/row |
|---|---:|---:|---:|
| CodeT5 / PLBART | 0.03–0.12 | 0.15–1.5 | — |
| Qwen 0.5B | 0.06–0.25 | 0.3–2 | 0.5–3 |
| Qwen 7B | 0.25–1 | 1–6 | 2–12 |
| Qwen 14B QLoRA | 0.6–2 | 2–10 | — |

CPU corpus metrics allowance: 0.02–0.15 seconds per validation/test/ICL row.
Qwen judge: 0.3–1.5 seconds/call; Llama judge: 0.6–3 seconds/call (max 32 tokens,
batch 1). Add 15% to A–J totals for loading, preprocessing/retrieval, validation
forward passes, checkpoint I/O and restarts. This is an allowance, not an I/O bound;
slow shared storage or CPU BERTScore may require substantially more. Qualification
4–24 hours is a separate allowance for 35 lifecycle and five inference pilots,
code checks and failures, assuming model files are already available.

For each training run:

```
seconds = train_rows * epochs * train_seconds_per_microbatch
        + (validation_rows * epochs + test_rows)
          * (generation_seconds_per_row + metric_seconds_per_row)
hours = seconds * 1.15 / 3600
```

For D, floor each split size times its sweep fraction (TR .2, CR .05); only the
CR official validation uses min(rows, CR_EVAL_MAX), default 2,000. For ICL use
test_rows * (ICL_seconds_per_row + metric_seconds_per_row); for J use calls times
seconds/call. Sum distinct outputs and apply the overhead allowance once.

## Replace assumptions with measurements

Run the user-launched server qualification first, then time representative admitted
runs by model/method/dataset. Synthetic two-step memory pilots establish bounded fit
and resume, not steady-state corpus throughput. Measure a full validation pass and
save/reload duration before relying on the ten-minute stop warning.

`complete_results.json.resources.train_step_time_s` is a list of **total training
seconds per epoch**, despite its name; do not multiply those values by step count.
Divide each epoch total by processed microbatches for the formula above, or use total
optimizer time divided by actual optimizer updates with effective batch accounted for.
`eval_time_s` already includes validation computation/metrics: when using measured
validation totals, do not add generation/metrics a second time. Test-generation timing
excludes final corpus metrics. For interrupted runs, resource summaries may cover
only the resumed process; combine persistent logs across allocations.

Measure representative ICL prompt distributions and uncached judge calls separately.
Report queue waits and failed pilots separately. With 71h50m usable per allocation,
sum `ceil(phase_hours / 71.8333)` across phases, rather than ceiling the whole queue:
these scenarios imply about **20–92 allocations default**, **25–115 with options**.
Validation that cannot finish before a hard kill can replay on every resubmission;
increase the warning lead time based on measured validation + checkpoint duration.
No local GPU workload is needed to calculate or revise this budget.
