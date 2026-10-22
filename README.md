# IndicJudge

IndicJudge compares responses from two Gemma models across ten Indian languages. Three independent LLM judges evaluate the responses using shared rubrics.

- **Candidate models:** Gemma-4-26B and Gemma-4-31B, as named in the scripts.
- **Judge models:** Sarvam-30B, Qwen3-14B and Ministral 3 8B Reasoning.
- **Languages:** Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu.

The project includes mathematical reasoning experiments, a four-task evaluation, and a separate perturbation experiment.

## Project folders

| Folder | What it does |
|---|---|
| `Dataset/` | Source datasets, task CSVs and dataset coverage information. |
| `gsm_gujarati_experiment/` | Gujarati math questions, generation scripts and Gemma responses. |
| `gsm_hindi_experiment/` | Hindi math questions, generation scripts and Gemma responses. |
| `gsm_kannada_experiment/` | Kannada math questions, generation scripts and Gemma responses. |
| `gsm_malayalam_experiment/` | Malayalam math questions, generation scripts and Gemma responses. |
| `gsm_marathi_experiment/` | Marathi math questions, generation scripts and Gemma responses. |
| `gsm_odia_experiment/` | Odia math questions, generation scripts and Gemma responses. |
| `gsm_punjabi_experiment/` | Punjabi math questions, generation scripts and Gemma responses. |
| `gsm_tamil_experiment/` | Tamil math questions, generation scripts and Gemma responses. |
| `mgsm_bengali_experiment/` | Bengali math questions, generation scripts and Gemma responses. |
| `mgsm_telegu_experiment/` | Telugu math questions, generation scripts and Gemma responses; the folder retains its existing spelling. |
| `sarvam_30b_judge/` | Sarvam judgments using the Initial Rubric. |
| `qwen3_14b_judge/` | Qwen judgments using the Initial Rubric. |
| `ministral_8b_judge/` | Ministral judgments using the Initial Rubric. |
| `sarvam_reasoning_judge/` | Sarvam judgments using Rubric Version 2. |
| `qwen_reasoning_judge/` | Qwen judgments using Rubric Version 2. |
| `ministral_reasoning_judge/` | Ministral judgments using Rubric Version 2. |
| `judge_experiment/` | Earlier Bengali and Telugu judging scripts and results. |
| `human_evaluation/` | Reviewer workbooks, a reading pack and scripts for human evaluation of math responses. |
| `multitask_judge_experiment/` | The first-five-sample experiment across four tasks and ten languages. |
| `multitask_50_samples/` | The expanded 50-sample experiment, including generation, judging, progress checks and analysis. |
| `perturbation_experiment/` | Generates responses with targeted reasoning flaws and evaluates them with the Initial Rubric. |
| `ministral-3-8b-reasoning/` | Local model storage, when present; GGUF weights are not needed in Git. |

## Mathematical reasoning: P1–P5

Each language has 50 questions. Both candidate models answer every question under five prompt conditions:

**50 questions × 5 prompts × 2 models × 10 languages = 5,000 candidate responses.**

| Prompt | Requested behavior |
|---|---|
| P1 | Correct step-by-step reasoning, correct calculations and a verified final answer. |
| P2 | A subtle logical or mathematical error in the reasoning, while still giving the correct final answer. |
| P3 | A correct method with a small arithmetic error at the end, producing an incorrect final answer. |
| P4 | Incorrect reasoning or invalid mathematical steps and an incorrect final answer. |
| P5 | A correct final answer with a brief explanation that does not fully justify it. |

These are generation instructions, not guaranteed properties of the responses. Judges evaluate what each candidate actually produced.

### Initial Rubric

The judge receives the problem, reference numerical answer, candidate solution, original generation prompt and prompt-intent summary. It returns:

- `answer_correct`: whether the visible final answer matches the supplied reference.
- `reasoning_quality`: a score from 0–5 for mathematical reasoning.
- `follows_prompt_intent`: whether the response satisfies the original prompt.
- `justification`: a short explanation.

The supplied reference is authoritative for answer correctness. Reasoning quality considers both internal reasoning and the visible response.

### Rubric Version 2

The judge receives only the problem, candidate solution and prompt number. **No gold answer or original generation prompt is supplied.** The prompt number is an identifier, not a scoring instruction.

The judge evaluates operation choice and logical flow together. Arithmetic execution errors alone do not lower the score. For example, choosing addition when multiplication is required is a structural error; writing `6 × 4 = 22` is an arithmetic error.

| Score | Meaning |
|---|---|
| 0 | No assessable solution method. |
| 1 | A method is attempted, but it makes no valid progress. |
| 2 | Major structural problems, a missing central relationship, or multiple independent repairs are needed. |
| 3 | One localized structural error or missing essential step. |
| 4 | A valid method with a minor clarity omission. |
| 5 | A complete, justified method with appropriate operations and logical dependencies. |

Output includes `operator_choice`, `overall_score` and `justification`. The operator label is `correct`, `partially_correct`, `incorrect` or `not_assessable`; it has no separate numerical weight.

Each rubric targets 5,000 judgments per judge. Results from the two rubrics stay in separate folders.

## Four-task experiment

The multitask experiments use **one generation prompt per task**, not P1–P5. They read CSVs from `Dataset/dataset_csv/` and freeze the selected inputs in `data/selection.json`.

| Task | What the judge evaluates | Reference use |
|---|---|---|
| Question answering | Correctness, relevance, completeness and support from context. | Reference answers are supplied; context takes precedence. |
| Reasoning | Appropriate choices and logical support for COPA-style cause/effect questions. | The dataset's correct-option label is withheld from the judge. |
| Summarization | Faithfulness, coverage of the main information and conciseness. | The judge receives the source document, not the reference summary. |
| Translation | Meaning preservation, completeness and fluency in the target language. | A reference translation is supplementary; source text takes precedence. |

All four tasks have references or labels in the dataset, but **candidate generation receives none of them**. Each judge returns one integer `overall_score` from 0–5 and a justification. Task-specific instructions are in each experiment's `prompts.py` and `RUBRICS.md`.

| Experiment | Samples per language/task | Inputs | Candidate responses | Judgments across three judges |
|---|---:|---:|---:|---:|
| `multitask_judge_experiment/` | 5 | 200 | 400 | 1,200 |
| `multitask_50_samples/` | 50 | 2,000 | 4,000 | 12,000 |

The 50-sample experiment uses the first 50 records in source order and reuses validated first-five results. It is not a random sample. The selected translation pairs run from the Indic language into English.

**Last reported server generation status:** 1,999 responses from each Gemma model, or 3,998 total. One Kannada summarization response from 26B and one Tamil summarization response from 31B were left out after generation failures. This is a reported snapshot, not a live count; local and server copies may differ. Status scripts still show these two as missing against the original 4,000-response target.

### Run the 50-sample workflow

Run from the project root. Load the required model in llama-server first, then set its endpoint. The scripts do not boot or switch models.

```bash
python3 -m pip install -r multitask_50_samples/requirements.txt

# Example: Gemma26B is already running on this endpoint.
export MULTITASK_GEMMA26B_API_URL="http://172.30.1.83:8080/v1/chat/completions"
python3 -u multitask_50_samples/generate_gemma26b.py

# Configure MULTITASK_GEMMA31B_API_URL before running the other candidate.
python3 -u multitask_50_samples/generate_gemma31b.py
```

For judging, configure `MULTITASK_SARVAM_API_URL`, `MULTITASK_QWEN_API_URL` and `MULTITASK_MINISTRAL_API_URL` to their running servers:

```bash
python3 -u multitask_50_samples/judge_sarvam.py
python3 -u multitask_50_samples/judge_qwen.py
python3 -u multitask_50_samples/judge_ministral.py

python3 multitask_50_samples/status.py --detail
python3 multitask_50_samples/analyze.py
```

Results are saved on the machine running the scripts, even when the model server is elsewhere:

- `multitask_50_samples/outputs/candidates/`: candidate response CSVs.
- `multitask_50_samples/outputs/judges/`: separate CSVs for each judge.
- `multitask_50_samples/analysis/`: scores, task means, pairwise agreement and coverage.

Rerunning skips validated saved records and retries missing ones. Available responses can be judged even with the two exclusions; commands may still report incomplete coverage. Missing responses and scores must not be treated as zero. Keep the saved selection, prompts and settings unchanged during a run. Recovery requests record their changed settings separately.

See the [50-sample workflow](multitask_50_samples/README.md) and [first-five workflow](multitask_judge_experiment/README.md) for more options.

## Perturbation experiment

`perturbation_experiment/` uses four additional math prompts: partial reasoning with a calculation error, a correct setup with a major logical gap, a correct answer with weak justification, and a correct method with a minor reasoning flaw. It saves new responses separately and compares Initial Rubric scores with the intended score ranges. Targets are not assumed to be achieved.

See [the perturbation instructions](perturbation_experiment/README.md) for generation, judging and summary commands.

## Comparing results

Match multitask results by `sample_id` and `source_model`, keeping each judge's scores separate. Compare candidate models on the same samples within each language and task.

`analyze.py` computes mean score as **sum of available scores ÷ number of available scores** and reports coverage. It also calculates exact agreement, mean absolute score difference and within-one-point agreement between judge pairs using matched records only.

Agreement between judges does not establish which judge is more accurate. To assess judge quality, compare against adjudicated human ratings using weighted Cohen's kappa, MAE and within-one-point agreement. These human-comparison metrics are not currently calculated by the multitask analysis script. The existing [human evaluation package](human_evaluation/README.md) covers the earlier math experiment.
