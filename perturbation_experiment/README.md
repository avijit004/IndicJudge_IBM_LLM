# Perturbation experiment (new, additive)

Everything here is new. No file from the original IndicJudge project is changed; the
scripts only *read* the original `questions.json` files and write inside this folder.

## The four perturbation prompts (`perturbation_prompts.py`)

They replace P1–P5 in the answer-generation step and target the judge's `reasoning_quality`
(Initial Rubric). Targets, not guarantees: the judge scores the generated response.

| ID | Perturbation | Target `reasoning_quality` | Expected `answer_correct` | Expected `follows_prompt_intent` |
| --- | --- | --- | --- | --- |
| PT1 | Partial reasoning with a calculation error | 2 | usually false | usually true |
| PT2 | Correct setup with a major logical gap | 2 | either | usually true |
| PT3 | Correct answer with weak justification | 2–3 | true | depends on prompt requirements |
| PT4 | Correct method with a minor reasoning flaw | 3 | true | usually true |

Note: PT3 and PT4 are close in spirit to the original P5 and P2; they differ in the explicit
score targets and wording.

## Pipeline

Run from the project root (the folder that contains `gsm_*_experiment/`; place this
`perturbation_experiment/` folder there).

```bash
# 1. Generate (same payload/temperature/max_tokens as the original scripts; resumable)
export GEMMA_31B_API_URL="http://HOST:PORT/v1/chat/completions"
python perturbation_experiment/generate_perturbation_responses.py --language hindi --model 31b
python perturbation_experiment/generate_perturbation_responses.py --language all --model 26b   # needs GEMMA_26B_API_URL

# 2. Judge with the original Initial Rubric (JUDGE_PROMPT copied verbatim)
export JUDGE_API_URL="http://HOST:PORT/v1/chat/completions"
export JUDGE_MODEL_NAME="Ministral-3-8B-Reasoning"
export JUDGE_TAG="ministral8b"
python perturbation_experiment/judge_perturbations.py --language hindi --source-model both

# 3. Compare actual scores with the targets
python perturbation_experiment/summarize_results.py
```

Languages: bengali, gujarati, hindi, kannada, malayalam, marathi, odia, punjabi, tamil, telugu (or `all`).
Useful flags: `--question-ids 1 2 3`, `--dry-run` (generate), `--check-data` (judge).

Outputs: `responses/gemma4_<26b|31b>_<language>_perturbation_responses.csv` (same columns as the original
response CSVs, with `prompt_id` PT1–PT4) and `results/<tag>_<language>_perturbation_results.csv`
(original judge columns plus `target_rq_min`, `target_rq_max`, `rq_in_target`). `results/summary_by_prompt.csv`
holds the per-prompt comparison.

## Controlled comparison

Every perturbation is applied to the same questions, so per question you can compare PT1–PT4 (and, if you
want, the original P1–P5 results) for the same judge. Record the judge's actual scores; do not assume the
targets were reached. `answer_correct`, `reasoning_quality` and `follows_prompt_intent` are independent:
a reasoning score of 2 does not imply a wrong answer.

## Smoke test without a GPU

```bash
python perturbation_experiment/tests/mock_llm_server.py 8099 &
export GEMMA_31B_API_URL=http://127.0.0.1:8099/v1/chat/completions JUDGE_API_URL=$GEMMA_31B_API_URL JUDGE_TAG=mock
python perturbation_experiment/generate_perturbation_responses.py --language hindi --model 31b --question-ids 1 2
python perturbation_experiment/judge_perturbations.py --language hindi --source-model 31b
python perturbation_experiment/summarize_results.py
```
(The mock returns canned text; delete `responses/` and `results/` files it creates afterwards.)

The Gemma endpoint ports are not hard-coded because the original scripts use two different ones
(8080 and 55555); set them via environment variables or `--api-url`.
