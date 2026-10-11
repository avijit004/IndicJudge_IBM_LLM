# IndicJudge

IndicJudge evaluates mathematical responses in ten Indian languages. Two Gemma models generate the responses, and three LLM judges evaluate them using two different rubrics.

- **Candidate models:** Gemma-4-26B and Gemma-4-31B, as named in the project scripts.
- **Judge models:** Sarvam-30B, Qwen3-14B, and Ministral 3 8B Reasoning.
- **Languages:** Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu.
- **Data:** GSM8K-Indic for eight languages and MGSM for Bengali and Telugu.

Each language has 50 questions. Both candidate models answer each question under five prompts:

**50 questions × 5 prompts × 2 models × 10 languages = 5,000 candidate responses.**

Each judge evaluates these responses separately. Each rubric therefore targets 15,000 judgments across the three judges. These are target counts, not a live progress report.

## Project folders

```text
IndicJudge_IBM/
├── gsm_gujarati_experiment/    # Gujarati questions, generation scripts, and Gemma responses.
├── gsm_hindi_experiment/       # Hindi questions, generation scripts, and Gemma responses.
├── gsm_kannada_experiment/     # Kannada questions, generation scripts, and Gemma responses.
├── gsm_malayalam_experiment/   # Malayalam questions, generation scripts, and Gemma responses.
├── gsm_marathi_experiment/     # Marathi questions, generation scripts, and Gemma responses.
├── gsm_odia_experiment/        # Odia questions, generation scripts, and Gemma responses.
├── gsm_punjabi_experiment/     # Punjabi questions, generation scripts, and Gemma responses.
├── gsm_tamil_experiment/       # Tamil questions, generation scripts, and Gemma responses.
├── mgsm_bengali_experiment/    # Bengali MGSM questions and Gemma responses.
├── mgsm_telegu_experiment/     # Telugu MGSM questions and Gemma responses; original folder spelling retained.
├── sarvam_30b_judge/           # Sarvam evaluation using the Initital Rubric.
├── qwen3_14b_judge/            # Qwen evaluation using the Initital Rubric.
├── ministral_8b_judge/         # Ministral evaluation using the Initital Rubric.
├── sarvam_reasoning_judge/     # Sarvam evaluation using Rubric Version 2.
├── qwen_reasoning_judge/       # Qwen evaluation using Rubric Version 2.
├── ministral_reasoning_judge/  # Ministral evaluation using Rubric Version 2.
├── human_evaluation/          # Reviewer sheets and tools for human evaluation using the first rubric.
└── judge_experiment/          # Earlier judging experiments, separate from the current workflows.
```

Each language experiment contains `questions.json`, two generation scripts, and two candidate CSVs. Some folders also include `get_questions.py` for downloading questions. `fill_blanks.py` retries empty candidate responses and updates the candidate CSV.

Each current judge folder has ten standalone language scripts. Qwen and Ministral have `run_all.py` in both rubric folders. Sarvam has an all-language runner only in `sarvam_reasoning_judge/`.

## P1–P5 prompts

All 20 generation scripts use the same five prompt types. They request different kinds of solutions; the generated response may or may not follow the request.

| Prompt | What the candidate is asked to do |
| --- | --- |
| P1 | Use correct step-by-step reasoning and calculations, verify the result, and give the correct answer. |
| P2 | Give the correct answer but include a subtle logical or mathematical error in the reasoning. |
| P3 | Use a correct approach, then introduce a small arithmetic error at the end to produce an incorrect answer. |
| P4 | Use incorrect reasoning or invalid mathematical steps and give an incorrect answer. |
| P5 | Give the correct answer with a brief explanation that does not fully justify it. |

The exact wording is stored in `PROMPTS` inside each generation script.

## Initital Rubric

This rubric checks answer correctness, reasoning quality, and whether the response follows its assigned prompt.

The judge receives the problem, gold numerical answer, original generation prompt, intent summary, candidate reasoning, and visible response. The gold answer comes from `answer_number` in `questions.json`, or from the number after `####` in its `answer` field.

| Field | What it measures |
| --- | --- |
| `answer_correct` | True or false: does the visible final answer match the reference? |
| `reasoning_quality` | Integer from 0 to 5: how sound and complete is the reasoning? |
| `follows_prompt_intent` | True or false: does the response meet the original prompt's requirements? |
| `justification` | A short explanation of the ratings. |

Reasoning scores range from **0** for absent or nonsensical reasoning to **5** for complete, rigorous reasoning. Scores **1–2** indicate substantial errors or gaps, **3** indicates mostly correct reasoning with minor issues, and **4** indicates correct, clear reasoning without significant errors.

A response can follow P4 while still receiving a low reasoning score and an incorrect-answer rating. Compliance and mathematical quality are assessed separately.

## Rubric Version 2

This rubric focuses on **operator choice and logical flow**. It ignores arithmetic execution mistakes and does not score final-answer correctness or prompt compliance.

The judge receives only the problem, prompt ID, and candidate solution. The solution includes both stored internal reasoning and the visible response. No gold answer, generation prompt, or intent summary is supplied. The prompt ID is only an identifier.

| Field | What it measures |
| --- | --- |
| `operator_choice` | `correct`, `partially_correct`, `incorrect`, or `not_assessable`. |
| `overall_score` | Integer from 0 to 5, assessing operator choice and logical flow together. |
| `justification` | A short explanation of the label and score. |

| Score | Meaning |
| --- | --- |
| 0 | No assessable solution method. |
| 1 | A method is identifiable but makes no valid progress. |
| 2 | Some valid steps, but the central approach needs repair, a central relationship is missing, or multiple independent structural errors remain. |
| 3 | An appropriate central approach needs one localized structural repair. |
| 4 | The method is valid and sufficient, with only a minor clarity omission. |
| 5 | The method is complete, justified, and free of unresolved contradictions or ambiguities. |

For example, choosing addition when multiplication is needed is a structural error. Writing `6 × 4 = 22` while correctly choosing multiplication is an arithmetic execution error and does not reduce the score. The full scoring rules are in each script's `JUDGE_PROMPT`.

## Running the judges

Run commands from the project root. The HTTP judge scripts need Python with `requests` installed:

```bash
python -m pip install requests
```

Start the appropriate model separately with llama-server before judging. Set its endpoint in the terminal used to run the scripts:

```bash
export QWEN_API_URL="http://YOUR_SERVER:YOUR_PORT/v1/chat/completions"
export MINISTRAL_API_URL="http://YOUR_SERVER:YOUR_PORT/v1/chat/completions"
```

For Sarvam, edit `API_URL` in the language scripts. The current Tamil and Telugu scripts use a different port from the other eight languages. These endpoint settings apply to both rubric versions.

Choose the commands for the rubric you want:

```bash
# Initital Rubric
python -u sarvam_30b_judge/sarvam_judge_bengali.py
python -u qwen3_14b_judge/run_all.py
python -u ministral_8b_judge/run_all.py

# Rubric Version 2
python -u sarvam_reasoning_judge/run_all.py
python -u qwen_reasoning_judge/run_all.py
python -u ministral_reasoning_judge/run_all.py
```

The first Sarvam command runs Bengali only. Use the corresponding language script for each remaining language. The other commands run all ten languages by default.

- All `run_all.py` scripts accept `--languages tamil telugu` to select languages.
- Qwen supports `--check-data` for offline input checks and `--pilot` for one fixed response per P1–P5 in each language.
- Ministral supports `--check-data`, `--question-ids 1 2 3 4 5`, and `--source-model both`, `26b`, or `31b`. Its default is all 50 questions and both models.
- Sarvam does not support the Qwen pilot or offline-check flags.

Use a small run to check the server response before starting the full evaluation. On macOS, `caffeinate -i` can be placed before a Python command to prevent idle sleep.

## Results and resuming

Each language saves a results CSV beside its judge script. The two rubric versions use separate files. Running the same script again skips saved judgments and retries missing ones. Failed or incomplete judgments are not saved as successful results.

Do not run two processes writing to the same result file. Keep the candidate CSVs unchanged during evaluation so saved judgments continue to refer to the same responses.

Match results using `(language, source_model, question_id, prompt_id)`. In Version 2, the original prompt remains in the output CSV for traceability but is never sent to the judge. Responses marked `[SALVAGED FROM REASONING]` retain that marker and should be considered separately during analysis.

## Human evaluation

The existing package contains 200 sampled responses for two independent reviewers. Each reviewer receives their own workbook and `reading_pack.html`; the coordinator keeps `coordinator.xlsx`.

Reviewers should understand the language they assess and rate independently before discussing disagreements. The current sheets use the Initital Rubric and have not been adapted to Version 2. Reference answers are not human ratings.

See [human evaluation instructions](human_evaluation/README.md). The sheet tools need `openpyxl`. Judge agreement alone does not establish accuracy; completed human reviews are needed to assess the ratings.
