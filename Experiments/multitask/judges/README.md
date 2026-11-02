> For the complete 4,000-response run, use `python3 run.py check`, `python3 run.py judge <model>` and `python3 run.py status` from the project root. Candidate CSVs are in `../candidate/`. The older preparation commands below describe the preserved 3,998-response run.

# Reference-free multitask judging

This is a separate experiment for the existing Gemma-4-26B and Gemma-4-31B responses in `../candidate/multitask_50_samples/`. It does not regenerate responses, reuse earlier judge scores, or change the previous experiment.

Sarvam-30B, Qwen3-14B and Ministral 3 8B Reasoning independently return an integer `overall_score` from 0–5 and a short `justification`.

**No dataset reference answer, correct-option label, reference summary or reference translation is sent to any judge.** QA context, reasoning options, source documents and translation source text remain necessary task inputs. The candidate's own answer is always included because it is what the judge must assess.

## Design

```text
multitask_50_samples: saved selection + validated Gemma responses
                             |
                  prepare.py: freeze and strip references
                             |
                    data/inputs.json (immutable)
                             |
               same task rubric and candidate text
                  /           |           \
             Sarvam          Qwen       Ministral
                  \           |           /
                  separate results in outputs/judges/
                             |
                         analyze.py
                  /                     \
       agreement + candidate scores    judge-quality leaderboard
                                       only with human ratings
```

The expected dataset is 10 languages × 4 tasks × 50 inputs × 2 candidates = 4,000 responses. You chose to leave two failed generations out, so preparation defaults to **3,998 available responses**, or **11,994 judgments** across three judges. It records the exact missing sample/model pairs. It will stop if the input count differs instead of silently using an incomplete local copy. Use `--expected-responses 4000` only if both missing generations are later completed, before freezing a new experiment.

The same frozen candidate records and ordering are used for all judges. Candidate model names, dataset IDs, original generation instructions and other judges' scores are omitted from API messages. Candidate text is preserved verbatim; any self-identification inside that text is not redacted. Each record is judged separately.

## What gets scored

The full, readable `JUDGE_PROMPT` and all four rubrics are in [rubric.py](rubric.py).

| Task | Inputs sent to the judge | What the score measures |
|---|---|---|
| QA | Context, question, visible response | Correctness and sufficiency supported by the context. |
| Reasoning | Premise, cause/effect question, both options, internal reasoning if available, visible response | Appropriate choices and logically justified steps. Arithmetic execution errors alone are ignored, following the existing reasoning rubric. |
| Summarization | Source document and visible summary | Faithfulness, central coverage and conciseness. |
| Translation | Source text, explicit direction and visible translation | Preserved meaning/details/tone, completeness and target-language fluency. |

The reasoning data is COPA-style causal reasoning, not the earlier math dataset. Selected translation pairs are Indic-to-English; a Telugu translation score, for example, describes a Telugu-source task, not Telugu output fluency.

Task score anchors follow the previous multitask experiment. QA and translation guidance is changed to remove reference use. QA/summarization/translation are scored on visible output only; reasoning includes the available internal reasoning as before. These are overall quality scores, not predicted answer strings or probability estimates. A bare correct option cannot automatically receive a high reasoning-quality score.

## Files

| Path | Purpose |
|---|---|
| `prepare.py`, `source_export.py` | Read and validate existing generation files, then freeze reference-free inputs. The original validation code is used only by the offline adapter. |
| `rubric.py` | Shared judge instructions, four task rubrics and explicit input-field allowlists. |
| `config.json` | Model-specific decoding settings and model provenance fields. |
| `judge_sarvam.py`, `judge_qwen.py`, `judge_ministral.py` | Run each judge independently. |
| `run_all.py` | Run all three sequentially if all their servers are already available. |
| `core.py`, `transport.py`, `utils.py` | Validation, API transport, checksums, atomic saves and file locks. |
| `preview.py` | Print one exact judge request without calling a model. |
| `status.py` | Validated completion counts, optionally by language and task. |
| `review.py` | Export blank, blinded human-review files and optional offline reference material. |
| `metrics.py`, `analyze.py` | Matched comparisons, uncertainty estimates and human-grounded judge rankings. |
| `dashboard.py` | Read-only Streamlit view of analysis exports. |
| `tests/` | Offline tests using synthetic responses and mocked inference. |
| `RESEARCH_PLAN.md` | Evaluation protocol, metrics, interpretation and limitations. |

## 1. Prepare on the machine with the latest candidate results

For the completed 4,000-response run, use `python3 run.py check` and `python3 run.py judge MODEL` from the project root. The direct preparation examples below describe the original 3,998-response snapshot.

Run these commands from the project root. They make no model calls.

```bash
python3 -m pip install -r Experiments/multitask/judges/requirements.txt
python3 Experiments/multitask/judges/prepare.py --check-only
python3 Experiments/multitask/judges/prepare.py
```

By default, input files come from `../candidate/multitask_50_samples/`. Use `--source-dir /path/to/multitask_50_samples` if needed. On your server, the project root is `/DATA/swagata/IndicJudge_IBM`.

Preparation refuses to overwrite a different snapshot. No candidate or reference data is bundled with these scripts. Once prepared, judging reads only the new snapshot and no longer needs the original Dataset or original experiment. References do not enter `data/inputs.json`.

## 2. Review settings and load models

Load the correct GGUF models separately in llama-server, with the aliases in `config.json` and at least 32,768 context tokens per slot. Scripts check alias and context, but an alias is not proof of model weights. Fill `model_records` in `config.json` before the first judge run: GGUF filename/hash, quantization, llama.cpp commit and hardware.

Model-specific decoding settings match the existing multitask configuration: Sarvam temperature 0, Qwen 0.6 and Ministral 0.7; seed 42; output cap 8,192 and requested reasoning budget 4,096. The Ministral system message is retained. This compares the **configured judge systems**, not model architecture in isolation. Server support for reasoning-budget options needs a live pilot. Fixed seeds do not guarantee identical output across server builds or hardware.

Endpoint variables are separate from the original experiment. For example, if Qwen is actually running on port 8080:

```bash
export RF_QWEN_API_URL="http://172.30.1.83:8080/v1/chat/completions"
```

Set `RF_SARVAM_API_URL` and `RF_MINISTRAL_API_URL` to their actual running endpoints too. No default ports are assumed. Use `127.0.0.1` when scripts and model servers run on the same machine. Optional authentication uses `RF_API_KEY`.

## 3. Validate, pilot and judge

```bash
# Inspect the exact reference-free payload for one task.
python3 Experiments/multitask/judges/preview.py --judge qwen --language bn --task qa

# All languages and tasks, offline only.
python3 Experiments/multitask/judges/run_all.py --check-data

# Small technical pilot across the available tasks/languages.
python3 -u Experiments/multitask/judges/judge_qwen.py --limit 10

# Full runs, each after its endpoint is ready.
python3 -u Experiments/multitask/judges/judge_qwen.py
python3 -u Experiments/multitask/judges/judge_ministral.py
python3 -u Experiments/multitask/judges/judge_sarvam.py

python3 Experiments/multitask/judges/status.py --detail
```

The pilot uses the first ten pending records in the frozen mixed order; it is not guaranteed to cover every task/language. Inspect real responses before continuing. For an explicit scope, use `--languages bn hi --tasks qa reasoning`. `--source-models gemma26b` or `gemma31b` selects one candidate. The default is both, across all languages/tasks.

Different judges may run concurrently against separate model servers because they write separate CSVs. `run_all.py` runs judges sequentially; it does not boot or swap models. On macOS, `caffeinate -i` may prefix a Python command.

Results go to `outputs/judges/{sarvam,qwen,ministral}_results.csv` inside this folder, on the machine running Python. Each valid judgment is saved immediately. Rerunning skips saved records. A limited run exits nonzero while selected work remains. Invalid JSON, invalid scores, empty or truncated replies and HTTP errors are logged and are not saved as successes. No candidate text is truncated to fit context.

Changes to the snapshot, rubric or model settings invalidate checkpoints instead of silently mixing experiments. Endpoint-only changes are allowed. Do not run two writers for the same judge. Keep this experiment separate from the original reference-assisted results.

## 4. Compare before human review

```bash
python3 Experiments/multitask/judges/analyze.py
```

This produces:

- `analysis/judge_agreement.csv`: quadratic weighted kappa, exact agreement, mean absolute score difference and within-one-point agreement. All pairs use the same records rated by all three judges.
- `analysis/candidate_scores.csv`: Gemma mean scores per judge, language and task, using only samples with both Gemmas rated by all three judges.
- `analysis/coverage.json`: counts, exclusions and analysis settings.
- `analysis/judge_leaderboard.csv` and `paired_mae_differences.csv`: empty until adjudicated human ratings are supplied.

**A judge giving higher scores is not necessarily a better judge. Agreement with other judges is not accuracy.** Missing scores stay missing, not zero. Means are response-weighted within the displayed group. Per-language/task tables are the main results; overall means mix different task criteria.

## 5. Compare with original answers and human ratings later

```bash
# Include reference material only for offline human review.
python3 Experiments/multitask/judges/review.py --include-references
```

This creates a reading pack, two blank reviewer CSVs, an adjudication CSV, the rubric, and a separate `references_for_review.json`. Candidate model identities and LLM scores are omitted. Existing review files are never overwritten. Omit `--include-references` for human review under exactly the same reference-free evidence conditions.

Have two qualified reviewers independently score each candidate using the same task score anchors. Keep their original ratings. Resolve disagreements by discussion or a third reviewer. In `review/adjudicated.csv`, fill:

- `human_score`: integer 0–5.
- `status`: `adjudicated` for completed entries.
- `reviewer_ids`: at least two distinct reviewer IDs separated by `|`.
- `protocol`: `human_reference_assisted` if references informed review, otherwise `human_reference_free`.
- `notes`: evidence and the adjudication decision.

Keep `evaluation_id` and `item_hash` unchanged. Do not mix human protocols in one analysis. The script validates and joins these ratings; it does not automatically merge reviewer opinions or fabricate consensus. Incomplete rows remain outside the benchmark.

Original answers are evidence about the **candidate**, not a ready-made 0–5 judgment. An exact option match cannot establish reasoning quality; a reference summary/translation is not the only valid wording. Do not turn answer matches into 0 or 5 labels. Human review should use references critically and retain the task-specific quality criteria. Reference-assisted humans constitute a stronger-evidence benchmark; label that distinction in your report.

```bash
python3 Experiments/multitask/judges/analyze.py \
  --human-ratings Experiments/multitask/judges/review/adjudicated.csv
```

The leaderboard sorts by **MAE against adjudicated human scores** (lower is better), with quadratic weighted kappa, within-one-point agreement, exact agreement and signed bias alongside it. It shows 95% sample-cluster bootstrap intervals for MAE and paired MAE differences. Default bootstrap count: 1,000. A positive signed bias means a judge scores higher than humans.

Ranks are descriptive and withheld below 30 distinct sample clusters per group; 30 is a reporting guardrail, not a guarantee of statistical power. Intervals are pointwise, not adjusted for the many task/language comparisons. The scripts do not declare a statistically established winner. See the research plan before drawing conclusions.

## Optional dashboard

```bash
python3 -m pip install -r Experiments/multitask/judges/requirements-dashboard.txt
streamlit run Experiments/multitask/judges/dashboard.py
```

The dashboard reads saved analysis exports, supports language/task filters, and separates judge quality, judge agreement and candidate performance. Run `analyze.py` again to refresh the exported results, then refresh the page. It makes no model calls.

## Offline tests

```bash
python3 -B -m unittest discover -s Experiments/multitask/judges/tests -v
```

Tests validate reference exclusion, all task/language payloads, resumption, failure handling, changed checkpoints, score metrics, paired sample matching and human-review safeguards. They do not establish live model compatibility or judge accuracy.
