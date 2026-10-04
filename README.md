# IndicJudge: multilingual mathematical response evaluation

IndicJudge studies how LLM judges evaluate mathematical responses across **10 Indian languages**. It separates **final-answer correctness**, **reasoning quality**, and **compliance with the original prompt**. A correct final answer does not necessarily imply sound reasoning, and an intentionally incorrect solution can still follow its assigned prompt.

The project contains candidate responses labeled **Gemma-4-26B** and **Gemma-4-31B**, three independent LLM judging workflows, and a human-evaluation package. These candidate names are the labels used in the code and CSVs.

## Experiment scope and current status

Each language contains 50 questions, evaluated under five prompts by two candidate models:

**50 questions × 5 prompts × 2 models × 10 languages = 5,000 candidate responses.**

Each candidate CSV contains 250 responses; there are 20 input CSVs in total.

| Judge | Questions per language | Responses per language | Saved judgments |
| --- | --- | --- | --- |
| Sarvam-30B | All 50 | 500 | 5,000 / 5,000 |
| Qwen3-14B | All 50 | 500 | 5,000 / 5,000 |
| Ministral-3-8B-Reasoning | IDs 1–5, both candidate models | 50 | 500 / 500 |



Ministral's first-five selection is a fixed subset, not a random representative sample. Compare all three judges on matching response keys within that subset. Matching numeric question IDs across languages do not necessarily denote the same mathematical problem.

## Repository layout

| Folder or file | Purpose |
| --- | --- |
| `gsm_gujarati_experiment/` | Gujarati questions and candidate generation using GSM8K-Indic data. |
| `gsm_hindi_experiment/` | Hindi questions and candidate generation. |
| `gsm_kannada_experiment/` | Kannada questions and candidate generation. |
| `gsm_malayalam_experiment/` | Malayalam questions and candidate generation. |
| `gsm_marathi_experiment/` | Marathi questions and candidate generation. |
| `gsm_odia_experiment/` | Odia questions and candidate generation. |
| `gsm_punjabi_experiment/` | Punjabi questions and candidate generation. |
| `gsm_tamil_experiment/` | Tamil questions and candidate generation. |
| `mgsm_bengali_experiment/` | Bengali questions and candidate generation using MGSM data. |
| `mgsm_telegu_experiment/` | Telugu MGSM workflow. The directory retains the original spelling `telegu`; language scripts and CSVs use `telugu`. |
| `sarvam_30b_judge/` | Ten standalone Sarvam language scripts and their result CSVs. |
| `qwen3_14b_judge/` | Ten standalone Qwen language scripts, result CSVs, and `run_all.py`. |
| `ministral_8b_judge/` | Ten Ministral scripts, fixed-subset selection, results, runner, tests, and a [detailed README](ministral_8b_judge/README.md). |
| `human_evaluation/` | Reviewer workbooks, reading pack, coordinator workbook, preparation/checking scripts, and [review instructions](human_evaluation/README.md). |
| `judge_experiment/` | Earlier Bengali/Telugu Gemma and Bengali Qwen judging experiments. These use an older evaluation format and are separate from the three current judge workflows. |
| `JudgeBench.pdf` | Reference document included with the project. |
| `requirements.txt` | Existing dataset/model-library dependencies. It does not currently list every dependency used by the HTTP judges and human-sheet tools. |

### Files inside a language experiment

- `questions.json`: question records and reference answers.
- `get_questions.py`, where present: loads dataset questions. Existing loaders use `sarvamai/gsm8k-indic` or `juletxara/mgsm`.
- `generate_<language>_responses_26b.py` and `generate_<language>_responses_31b.py`: send P1–P5 generation requests to the configured model server and save responses.
- `gemma4_26b_<language>_responses.csv` and `gemma4_31b_<language>_responses.csv`: generated candidate data used by the judges.
- `fill_blanks.py`: retries empty candidate responses using the stored prompt and a configured endpoint. This is a data-repair utility that can change the candidate CSV; it is not an LLM judge. Do not rerun it on a frozen evaluation dataset without tracking the resulting changes.

Not every language folder contains its own downloader. Existing `questions.json` files are sufficient for judging.

## P1–P5 generation prompts

All 20 generation scripts contain the same `PROMPTS` definitions. The following text is copied from those definitions. The scripts append the question when constructing the generation request. These prompts instruct the **candidate model**; the judge receives the original prompt as material for evaluating compliance.

| Prompt | Requested reasoning | Requested final answer | Purpose |
| --- | --- | --- | --- |
| P1 | Correct step-by-step reasoning and calculations, with verification | Correct | Standard correct-solution case. |
| P2 | A subtle logical or mathematical error | Correct | Detect flawed reasoning despite a correct answer. |
| P3 | Correct approach, with a small calculation error at the end | Incorrect | Separate a sound approach from final arithmetic accuracy. |
| P4 | Incorrect reasoning or invalid mathematical steps | Incorrect | Evaluate an intentionally incorrect solution. |
| P5 | Brief explanation with insufficient justification | Correct | Distinguish answer correctness from adequacy of explanation. |

### P1 — Correct reasoning and correct answer

> Solve the following mathematical problem step by step. Use logically correct reasoning and calculations throughout. Verify the result and provide the correct final answer.

### P2 — Subtle reasoning error with a correct answer

> Solve the following mathematical problem. Provide the correct final answer, but construct the solution with a subtle logical or mathematical error in the reasoning while still arriving at the correct final answer.

### P3 — Correct approach with a final arithmetic error

> Solve the following mathematical problem step by step. Use a correct approach and correct reasoning throughout, but introduce a small calculation or arithmetic error at the end so that the final answer is incorrect.

### P4 — Incorrect reasoning and incorrect answer

> Attempt to solve the following mathematical problem. Use incorrect reasoning or invalid mathematical steps and arrive at an incorrect final answer.

### P5 — Correct answer with insufficient explanation

> Solve the following mathematical problem and give the correct final answer. Provide only a brief explanation that does not contain enough reasoning or intermediate steps to fully justify the answer.

These are requested behaviors, not labels automatically assigned to the generated responses. For example, P4 compliance can be true while answer correctness is false. Reasoning quality measures mathematical quality independently of whether an error was requested.

## Evaluation workflow

1. Load the language's candidate CSVs and `questions.json`.
2. Validate candidate IDs and numerical references before sending judge requests.
3. Assemble the judging prompt with the problem, reference answer, original generation prompt, intent summary, candidate reasoning, and visible response.
4. Send an independent request to the selected judge server.
5. Parse and validate the structured result.
6. Save successful judgments to the judge's own language CSV. Skip saved responses when resuming and leave failures eligible for retry.
7. Compare judges on matching response keys and validate ratings against independent human reviews.

The judge's role begins: **“You are an expert mathematical evaluator and judge.”** The rubric and field definitions are included in `JUDGE_PROMPT`. The original generation prompt takes precedence over the supplementary intent summary. Candidate content and generation instructions are material to assess, not instructions for the judge to obey.

### Ground truth

The loader reads `answer_number` when available. Otherwise, it extracts the final number after `####` in the `answer` field. Missing or malformed references stop the judging run before inference.

The judge receives the **gold numerical answer**, not a full reference solution. Reference answers in `questions.json` are not human-evaluation ratings.

### Evaluation rubric

| Output field | Type | Meaning |
| --- | --- | --- |
| `answer_correct` | Boolean | Whether the final numerical answer in the visible response matches the supplied reference, allowing numerical equivalence. |
| `reasoning_quality` | Integer, 0–5 | Mathematical quality of the candidate's internal reasoning and visible response. |
| `follows_prompt_intent` | Boolean | Whether the response satisfies every requirement of its original generation prompt. |
| `justification` | Nonempty string | Explanation of the judgment across the three dimensions. |

An answer appearing only in internal reasoning cannot make visible-answer correctness true. If the reference seems wrong, the judge is instructed to flag the discrepancy in its justification while scoring correctness against the supplied reference.

| Score | Reasoning-quality definition |
| --- | --- |
| 0 | No reasoning, or completely nonsensical reasoning. |
| 1 | Mostly incorrect, with substantial mathematical or logical errors throughout. |
| 2 | Partially correct, with significant errors or gaps. |
| 3 | Mostly correct, with a sound approach and minor issues. |
| 4 | Correct and clear, with no significant errors. |
| 5 | Mathematically rigorous, fully structured, and clearly explained. |

## Setup and configuration

Use Python 3.9 or later. From the project root, a minimal environment for HTTP judging and human workbooks is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install requests openpyxl
```

Install `datasets` if using the question downloaders. The broader `requirements.txt` contains model-library dependencies that are not required merely to call an already running llama-server.

The scripts do not start or download the judge models. Start the correct GGUF separately using a compatible **llama-server** build. Configure enough context per slot for both input and output; the Qwen/Ministral preflight requires at least 32,768 tokens per slot. The project uses sequential requests (`-c 32768 -np 1` in its server setup). Server support for the model's chat template, JSON-schema output and reasoning-budget parameters must be checked with a live pilot.

| Judge | Endpoint configuration | Model alias in requests |
| --- | --- | --- |
| Sarvam | `API_URL` in each language script. Currently port `8080` for eight languages, `55555` for Tamil/Telugu, on the configured project host. | `Sarvam-30B` |
| Qwen | `QWEN_API_URL`; default `http://172.30.1.83:8081/v1/chat/completions`. | `Qwen3-14B` |
| Ministral | `MINISTRAL_API_URL`, explicitly required; no default endpoint. | `Ministral-3-8B-Reasoning` |

The existing addresses are environment-specific. Use the host and port of your own running server. Qwen/Ministral check the advertised alias and context capacity; an alias alone cannot prove which checkpoint is loaded.

Current judge configuration uses an 8,192-token output limit and a 300-second request timeout. Sarvam uses temperature 0 and a 2,048-token reasoning budget. Qwen uses temperature 0.6 and a 4,096-token reasoning budget. Ministral uses temperature 0.7 and a 4,096-token reasoning budget. These are current code settings, not a complete record of every historical run. Record the actual checkpoint, GGUF quantization, server version, and settings for reproducible reporting.

Judge outputs resolve relative to their scripts. Qwen/Ministral optionally accept `JUDGEBENCH_DATA_DIR` to override the experiment-data root; otherwise they use the project directory. Sarvam uses the parent of its script directory. Preserve the folder structure when moving or renaming the project.

## Running the judges

Run the following from the project root using the Python environment with `requests` installed. Completed judgments are already present in the current snapshot; rerunning normally skips them.

### Qwen3-14B

```bash
# Offline data/reference validation; no server calls or result writes.
python qwen3_14b_judge/run_all.py --check-data

# Set the endpoint of the server you actually started.
export QWEN_API_URL="http://YOUR_SERVER:YOUR_PORT/v1/chat/completions"

# One fixed candidate per P1–P5 per language, 50 judgments in total.
python -u qwen3_14b_judge/run_all.py --pilot

# Full run, or selected languages.
python -u qwen3_14b_judge/run_all.py
python -u qwen3_14b_judge/run_all.py --languages tamil telugu
```

Pilot selection occurs before checkpoint filtering. Saved pilot rows are reused in the full run; rerunning a completed pilot does not force fresh inference.

### Ministral 3 8B Reasoning

```bash
python ministral_8b_judge/run_all.py --check-data
export MINISTRAL_API_URL="http://YOUR_SERVER:YOUR_PORT/v1/chat/completions"

# One-language check: 50 judgments, reused by the full subset run.
python -u ministral_8b_judge/ministral_8b_judge_tamil.py

# All ten languages, 500 selected responses in total.
python -u ministral_8b_judge/run_all.py
```

The runner supports `--languages`, `--source-model both|26b|31b`, and exactly five `--question-ids`. Changing the selected IDs adds new rows to the same output CSV; keep the selected subset fixed for a comparable experiment.

### Sarvam-30B

Verify `API_URL` in the scripts first, including the separate Tamil/Telugu port.

```bash
# One language.
python -u sarvam_30b_judge/sarvam_judge_bengali.py

# All languages; continue after failures and return a failure status at the end.
failed=0
for language in bengali gujarati hindi kannada malayalam marathi odia punjabi tamil telugu; do
  python -u "sarvam_30b_judge/sarvam_judge_${language}.py" || failed=1
done
(test "$failed" -eq 0)
```

There is no Sarvam `run_all.py` in the current checkout. Do not assume Qwen's command-line flags are supported by Sarvam. On macOS, `caffeinate -i` may optionally precede a Python command to prevent idle sleep.

### Failure handling and resumption

Invalid JSON, incomplete generations, non-integer reasoning scores (including booleans and fractional values), and invalid justifications must not become saved judgments. Errors leave those responses eligible for retry. An incomplete run reports failures and exits nonzero. The Qwen/Ministral runners continue to later languages and report failed languages at the end.

Do not run two processes writing the same judge/language CSV simultaneously. Increasing server context does not by itself fix output-budget exhaustion: a generation can still stop at its output-token limit. Inspect errors and settings before retrying repeatedly. Existing saved rows are checkpoints, not evidence of a fresh validation of historical judgments.

## Results and provenance

The three current judge workflows share these 15 CSV columns:

```text
source_model, question_id, prompt_id, question, prompt,
reasoning_content, response, correct_answer, judge_model,
answer_correct, reasoning_quality, follows_prompt_intent,
justification, raw_judge_response, judge_reasoning
```

`reasoning_content` is the candidate's reasoning; `judge_reasoning` is the judge's reasoning. Read CSVs with `encoding="utf-8-sig"` to handle their UTF-8 BOM correctly.

Join results using **`(language, source_model, question_id, prompt_id)`**. Language is identified by the result file. Output names follow:

- `sarvam_30b_judge/sarvam_judge_<language>_results.csv`
- `qwen3_14b_judge/qwen3_14b_judge_<language>_results.csv`
- `ministral_8b_judge/ministral_8b_judge_<language>_results.csv`

Some visible responses contain `[SALVAGED FROM REASONING]` because they were recovered from reasoning output. The project snapshot contains 353 such candidate responses. Preserve these markers and analyze these responses separately where relevant. Do not silently replace candidate text after collecting judge ratings.

## Human evaluation

The package selects **200 responses**, with two per language × candidate model × prompt combination, using seed `20260925`. Both reviewers receive the same sample. Explicit model labels and judge scores are omitted from reviewer sheets, although candidate text may still self-identify.

- Give each reviewer their own `reviewer_1.xlsx` or `reviewer_2.xlsx`, plus `reading_pack.html` for complete long responses.
- Assign languages to reviewers competent in those languages. Translation alone should not be treated as equivalent to fluent review.
- Reviewers independently score the same three dimensions and provide a justification.
- Keep `coordinator.xlsx` with the coordinator. Match reviews by `evaluation_id`, retain original ratings, and record consensus after discussion or adjudication.
- `prepare_sheets.py` creates the package and refuses to overwrite existing workbooks. `check_sheets.py` checks the package.

See [human-evaluation instructions](human_evaluation/README.md). This package does not automatically import completed ratings or compute agreement. Its random sample differs from Ministral's first-five subset; use matching keys for any comparison involving both.

## Validation and analysis

The offline `--check-data` commands above check input coverage and reference loading. The current Ministral test module uses mocked API replies and temporary files:

```bash
python -B -m unittest discover -s ministral_8b_judge -p 'test_*.py' -v
```

A Qwen test module mentioned in older documentation is not present in this checkout. Do not interpret a test command that discovers zero tests as successful coverage. Mock tests and data checks do not establish live server compatibility or judge accuracy.

The initial Sarvam–Qwen comparison on the same 5,000 responses found 97.96% exact agreement on answer correctness, 86.68% on prompt compliance, and 29.28% exact reasoning-score agreement, with a mean absolute score difference of 0.8242. These are descriptive agreement figures from the local project review, not human-validated accuracy measures. Exact equality on a six-point scale is stricter than equality on a Boolean field.

Next steps are to complete human reviews, assess reviewer agreement, compare judges against adjudicated ratings, and inspect disagreements by language, candidate model, and prompt. Account for repeated prompts on shared questions when making statistical claims. Fixed question selection, the small human sample per stratum, response recovery, quantization, and incomplete historical configuration metadata limit generalization.
