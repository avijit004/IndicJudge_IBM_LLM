# Evaluation plan

## Questions and scope

1. How do three reference-free judge systems assess the same saved Gemma responses?
2. Where do their 0–5 ratings agree or disagree by task and language?
3. After independent human adjudication, which system has lower score error for each task/language?

These questions are distinct from which Gemma candidate earns higher scores. Judge quality must be measured against external ratings, not a majority vote of the judges being evaluated.

The unit is a candidate response keyed by `(sample_id, source_model)`. Both responses to one sample are related observations. The default frozen target is the 3,998 successful responses; the two missing candidates remain explicit exclusions. Exclusion can bias estimates if failures are systematically difficult, so report the exact exclusions. Original first-five and later responses are not separate independent studies: the expanded dataset reuses the originals once.

## Controlled protocol

- Use one immutable set of candidate responses, including unchanged source text and language directions.
- Remove all dataset reference fields from the judging snapshot, then construct payloads with an explicit allowlist.
- Use the same task rubric for each judge. Preserve prior task score anchors. Record the reference-removal change as a new experiment.
- Use single-response scoring without a competitor on the same page. Hide candidate model metadata and existing judge ratings. Keep text unchanged even if a candidate names itself.
- Preserve the original task evidence policy: internal reasoning is included only for the reasoning task. Evaluate visible output alone for other tasks. Therefore the reasoning score is not strictly a visible-answer-only score.
- Record exact requests, returned text, parsed scores, timestamps, generation lineage and inference settings. Record GGUF hash, quantization, server revision and hardware before running.
- Use the existing model-specific decoding settings. Conclusions concern these configured systems. A separate controlled-decoding experiment would be needed to isolate sampling effects.
- Inspect a technical pilot for output validity and context limits, not to tune prompts until a favored judge wins. If the rubric changes, version it and start a separate run.
- Keep invalid/truncated outputs as failures, never zero ratings. Report coverage and do not silently discard failures from comparisons.

This workflow validates explicit field exclusion; it cannot prove a model has never seen the benchmark during pretraining or that source text itself contains no answer-relevant information. QA context is intentionally answer-bearing evidence.

## Human benchmark

Have two reviewers with the required language competence rate responses independently and blind to LLM scores. Use the same score anchors; adjudicate disagreement while retaining independent ratings. A third reviewer may resolve difficult cases. Workbooks/CSV status alone cannot verify reviewer quality.

Two reference protocols are available: reference-free humans see the same evidence as judges; reference-assisted humans additionally consult dataset answers. Do not combine these protocols in one leaderboard. Reference-assisted ratings are useful for correctness auditing but supply different evidence, which must be disclosed.

References are not human quality scores. Exact match and token-overlap F1 are especially misleading where QA references are English and candidate responses are Indic. For causal reasoning, option correctness is a separate outcome from whether the explanation is justified. Summarization and translation permit multiple valid outputs. This workflow exports reference material for later review; it intentionally does not invent a generic answer-to-score conversion.

The current scripts export independent-review templates but do not calculate inter-reviewer agreement or merge disagreements automatically. Report reviewer agreement and adjudication procedure before treating the adjudicated set as a benchmark. If reviewing a subset, choose it independently of judge scores, with representation across language, task and candidate; disclose selection. Do not select only convenient/high-agreement items. Reserve a held-out evaluation set if developing/tuning the rubric.

## Metrics and leaderboard

Predeclare **MAE** as the primary ranking criterion: mean absolute difference between a judge's score and the adjudicated human score. This assumes adjacent 0–5 steps have equal numerical cost. It is interpretable in score points and complements an ordinal agreement measure.

Report **quadratic weighted Cohen's kappa** as a supporting agreement measure and **within-one-point agreement** as a simple tolerance measure. Kappa uses a fixed six-class scale and is undefined when its expected disagreement is zero; those cells are blank, not assigned zero or one. Its value depends on score prevalence. Include exact agreement and signed score bias. No weighted combination of unrelated metrics is invented.

Before humans exist, calculate the same pairwise descriptive metrics between judges and call MAE a *mean absolute score difference*. High agreement, correlation or high average scores cannot establish judge quality. F1/recall require a separately justified categorical decision task; they are not the primary metrics for these ordinal quality scores.

All judge comparisons use the common intersection of records rated by all three judges; human comparisons additionally require an adjudicated score. Show the denominator and coverage for every group. Candidate mean comparisons further require both Gemmas on the same sample, rated by all judges. Missing data may be nonrandom even after matching; a partially complete leaderboard is provisional.

Export global, per-language, per-task and language×task groups. Aggregation is response-weighted, not an equal-cell macro-average. Per-cell results are primary because task rubrics measure different constructs. First-50 source-order sampling does not justify representative claims about a language population.

Uncertainty uses a reproducible stratified cluster bootstrap: sample IDs are resampled within each language/task cell, keeping the two Gemma responses together. Use the same resampled units for paired differences of judges' absolute errors. Default: 1,000 replicates and 95% percentile intervals. A negative `mae_a_minus_b` favors A on this benchmark. Cross-language aligned questions are not modeled because source IDs do not establish translation alignment; if alignment is later known, use shared underlying question IDs for clustering.

Descriptive ranks require at least 30 sample clusters; this is a conservative display choice, not a power calculation. Confidence intervals are pointwise and not multiple-comparison corrected. Do not claim definitive per-language winners from small gaps or dozens of inspected intervals. Preregister a smaller comparison family or apply a suitable multiplicity correction for formal superiority claims. Distinguish statistical uncertainty from subject-matter significance.

## Limitations to include in the report

- Small, nonrandom source-order selections; only two candidate model configurations.
- Two excluded generations and possible missing judge scores.
- Variable candidate reasoning availability and recorded recovery-generation settings.
- Stochastic judges: one saved rating per candidate does not measure repeat-run stability. A repeated-run study should use separate folders and planned seeds, not overwrite this run.
- Different model sizes, quantizations and decoding policies; no causal attribution to model size alone.
- Human expertise, annotation consistency and possible imperfect dataset references.
- These are quality judgments, not calibrated probabilities or evidence of universal model superiority.

## Sources

The design separates judge/human agreement and model biases following the [MT-Bench/Chatbot Arena judge study](https://arxiv.org/abs/2306.05685). Metric implementations use the documented [quadratic weighted Cohen's kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) and [mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html). The frozen protocol, thresholds and bootstrap design here are project-specific choices, not claims that these sources prescribe them.
