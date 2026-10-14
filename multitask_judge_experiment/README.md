# Four-task judging experiment

This is a separate experiment. It uses five samples for each of ten languages and four tasks, with the same selected inputs for every model.

- **Candidate models:** Gemma-4-26B and Gemma-4-31B (the existing project's model labels).
- **Judges:** Sarvam-30B, Qwen3-14B and Ministral-3-8B-Reasoning.
- **Tasks:** question answering, reasoning, summarization and translation.
- **Languages:** Bengali (`bn`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`), Malayalam (`ml`), Marathi (`mr`), Odia (`or`), Punjabi (`pa`), Tamil (`ta`) and Telugu (`te`).
- **Output:** one integer `overall_score` from 0–5 and a 2–4 sentence justification per judgment.

There are **200 inputs → 400 Gemma responses → 1,200 judgments**. Each judge scores 400 responses. This uses **one generation prompt per task**, not the previous P1–P5 conditions.

No model run was launched when creating this folder. Tests use mock replies and temporary outputs. Earlier experiment results are not imported or changed.

## Files to review

| File/folder | What it does |
|---|---|
| `prompts.py` | All four generation instructions, shared judging rules and task-specific 0–5 rubrics. This is the source used at runtime. |
| `RUBRICS.md` | Readable copy of the current rubrics and reference policy. |
| `config.json` | Server endpoints, model aliases and model-specific sampling settings. |
| `data/selection.json` | Frozen 200-sample selection, source IDs, inputs, references and content hashes. |
| `data/selection.csv` | The same selection in a CSV for inspection; runtime reads the JSON. |
| `review_examples/` | Four example generation prompts and four judge prompts with clearly marked candidate placeholders. |
| `prepare_data.py` / `data.py` | Read and validate the source Dataset, select the first five rows per language/task and preserve provenance. |
| `generate_gemma26b.py`, `generate_gemma31b.py` | Separate generation entrypoints; each defaults to all 200 inputs. |
| `judge_sarvam.py`, `judge_qwen.py`, `judge_ministral.py` | Separate judge entrypoints; each defaults to both Gemmas and all 400 responses. |
| `generate.py`, `judge.py`, `common.py` | Shared execution, strict validation, server checks and checkpoint storage. |
| `run_all.py` | Run selected candidate or judge models sequentially. |
| `status.py` | Read-only progress counts, optionally per language and task. |
| `review_prompts.py` | Print an actual request offline with placeholders for candidate output. |
| `tests/test_workflow.py` | Offline data, prompt, API-response and resume tests. |
| `outputs/` | Created when you run generation/judging. Separate candidate and judge CSVs. |

All scripts use paths relative to this folder. They work when the project is renamed or launched from another directory. The shared modules are all inside this folder; the old judge scripts are not dependencies.

## Data selection and task meaning

Source: `../Dataset/dataset_csv/{qa,reasoning,summarization,translation}/{language_code}.csv`.

The selection uses the **first five data records in each file's existing order**. This is deterministic, not a new random or representative sample. All 100 rows in each of the 40 relevant source CSVs are checked for usable fields and duplicate IDs before selection. Selection IDs include task, language code and source ID. Do not assume similarly numbered rows across languages represent the same underlying question.

`selection.json` stores the selected data and source-file hashes. Existing selections are never silently overwritten. `prepare_data.py --check-only` compares the frozen selection with the current Dataset. After preparation, runtime uses the snapshot so subsequent Dataset edits cannot silently change an in-progress run.

The current ten-language data has these formats:

- **QA:** context and a question. Some contexts and reference answers are English; the candidate is asked to answer in the selected Indic language.
- **Reasoning:** COPA-style premise, cause/effect question and two choices. These are causal commonsense questions, not the earlier GSM mathematics questions. The candidate must select an option and explain why. Source labels are zero-based; displayed choices are numbered 1 and 2. The label is never sent to either the candidate or reasoning judge.
- **Summarization:** source document to a summary in the selected Indic language. Some source documents are English, making these cross-lingual tasks.
- **Translation:** preserve the CSV's `source_lang` and `target_lang`. Currently all selected rows translate **from the Indic language into English**. We do not reverse the pairs or claim that the translation outputs are Indic-language text.

## Rubric and reference policy

The complete instructions are in `prompts.py` and `RUBRICS.md`.

| Task | Candidate receives | Judge receives | What the score measures |
|---|---|---|---|
| QA | Context and question | Context, question, reference answers and visible response | Correctness, relevance and support from context |
| Reasoning | Premise, cause/effect relation, both options | Same problem, candidate internal reasoning if available, and visible response; **no gold label** | Valid method and logical flow; arithmetic execution errors excluded |
| Summarization | Source document | Source document and visible summary; **no reference summary** | Faithfulness, central coverage and conciseness |
| Translation | Source text and explicit language direction | Source text, language direction, reference translation and visible translation | Meaning preservation, completeness and target-language fluency |

No reference answer is sent during candidate generation. References retained in the selection for provenance are not automatically inserted into prompts. QA context and translation source take precedence if a reference conflicts with them; the judge should explain the conflict. Different valid wording is accepted.

Judges receive neither candidate model names nor other judges' ratings. They receive the same rubric and content; model-specific decoding settings and the Ministral reasoning-format system message follow the earlier project's settings. The saved CSV still identifies models so you can join results afterward.

Judgments have exactly `overall_score` and `justification` in their model-produced JSON. The score must be an actual integer, excluding Boolean, fractional and quoted values. Justification must be a nonempty string of at most 1,200 characters. The 2–4 sentence requirement is prompted, not enforced with an unreliable sentence-count heuristic.

Report scores per task and language. These tasks measure different things, so do not describe an average across tasks as answer accuracy. Five examples per cell are suitable for a small experiment, not broad performance claims.

## Offline review (safe to run now)

Run the examples below from the project root, using a Python environment with `requests`:

```bash
python -m pip install -r multitask_judge_experiment/requirements.txt

python -B multitask_judge_experiment/prepare_data.py --check-only
python -B multitask_judge_experiment/run_all.py generate --check-data
python -B multitask_judge_experiment/run_all.py judge --check-data
python -B -m unittest discover -s multitask_judge_experiment/tests -v

python -B multitask_judge_experiment/review_prompts.py --language bn --task reasoning --model sarvam
python -B multitask_judge_experiment/status.py --detail
```

These commands make no model API requests. A successful offline check means the inputs/prompts/checkpoints are valid, not that generation or judging has completed. `status.py` is the coverage report.

For a fresh copy without the saved selection, first run `prepare_data.py` without `--check-only`. An alternative dataset root can be passed through `--dataset-dir`.

## Configure servers after review

Load models separately with llama-server and at least 32,768 context tokens per slot. The alias returned by `/v1/models` must match the corresponding `alias` in `config.json`. Edit aliases before starting if your Gemma servers advertise different names.

Current defaults in `config.json` use Qwen on `172.30.1.83:8080` and Ministral on `172.30.1.83:33937`. These apply to the individual judge scripts and `run_all.py`. Sarvam and Gemma endpoints must still be set explicitly when those models are loaded.

Environment variables override the defaults. To use the current servers:

```bash
export MULTITASK_QWEN_API_URL="http://172.30.1.83:8080/v1/chat/completions"
export MULTITASK_MINISTRAL_API_URL="http://172.30.1.83:33937/v1/chat/completions"
```

Remove an old override with `unset MULTITASK_MINISTRAL_API_URL` to use the configured Ministral default. If the model restarts on a different port, update its endpoint. `MULTITASK_API_KEY` can optionally supply a shared Bearer token. Endpoint changes do not invalidate saved judgments.

The transport checks `/v1/models` and `/props`, then uses `/apply-template` and `/tokenize` before each completion to check the input plus output token budget with a 256-token margin. It does not truncate text. Those endpoints must be enabled on the llama-server build. The judging requests preserve the earlier scripts' reasoning-budget options; support and model behavior need a live pilot with your installed build.

Settings are saved in each row's request JSON. Hashes detect changes to the selection, prompts, model settings or saved candidates and stop instead of silently mixing results. Changing only the server address to serve the same model does not invalidate saved results. Do not change the underlying GGUF or chat template mid-run under the same alias; aliases cannot verify model weights. For a different model build or experiment configuration, use a separate experiment/output folder.

## Run when you are ready

Generate with each Gemma, after configuring its endpoint:

```bash
python -u multitask_judge_experiment/generate_gemma26b.py
python -u multitask_judge_experiment/generate_gemma31b.py
```

Once candidate generation is complete, run the judges independently:

```bash
python -u multitask_judge_experiment/judge_sarvam.py
python -u multitask_judge_experiment/judge_qwen.py
python -u multitask_judge_experiment/judge_ministral.py
```

Or use `run_all.py generate` / `run_all.py judge` to run each stage sequentially. Models must already be served; the runner does not boot or swap models. To run just one through the runner: `run_all.py judge --models sarvam`.

All entrypoints accept `--languages bn hi`, `--tasks qa reasoning`, `--config PATH`, `--manifest PATH`, `--output-dir PATH` and `--check-data`. Judge entrypoints also accept `--source-models gemma26b`, `gemma31b`, or `both` (default).

For a small live check after review, select one language and task (five samples). `--limit 1` attempts only the next pending record; it intentionally returns nonzero while the selected scope remains incomplete. Use `--check-data` for a no-network check.

## Results and resuming

```text
outputs/
  candidates/gemma26b_responses.csv
  candidates/gemma31b_responses.csv
  judges/sarvam_results.csv
  judges/qwen_results.csv
  judges/ministral_results.csv
```

Join judgments using `sample_id` and `source_model`. `sample_id` includes task and language; the judgment rows also name the judge. The frozen selection contains the problem/source/reference text, and candidate CSVs contain the responses and any separately returned reasoning. Judge CSVs include score, justification, request, raw judgment and optional judge reasoning.

Each accepted result is saved atomically. Rerunning skips matching valid records and retries unsaved failures. Refusals returned as visible text can be scored normally; empty API content, malformed JSON and length-limited generations are failures, not successful results. No answer is salvaged from internal reasoning to replace missing visible output.

Errors are logged alongside the result file as `.errors.jsonl`. Errors do not enter the successful-results CSV. Missing candidates count as unresolved judgments; available candidates can still be judged. Every run reports saved, successful, failed and unresolved counts and exits nonzero if the selected scope is incomplete. Corrupt or mismatched checkpoints stop without rewriting them.

Different models may run simultaneously against separate servers and write separate files. A file lock prevents two processes from writing the same model output. Avoid editing candidate files or changing configuration while a judge run is active.

The mocked tests do not establish live server compatibility or judge accuracy. No evaluation scores have been fabricated or prefilled.

## Retry responses that hit the output limit

If a candidate repeatedly ends with `finish_reason='length'`, use:

```bash
python -u multitask_judge_experiment/generate_gemma26b.py --retry-length
```

This option only changes unsaved samples with a prior length failure in that model's
`.errors.jsonl` log. Fresh samples use the baseline configuration, and already saved
responses are skipped. Do not edit `config.json` to retry these failures.

The fixed `length-v1` recovery profile uses seed 43, at least 16,384 output tokens,
and a requested thinking budget of 4,096 tokens with a request to finish the task.
Thinking-budget support depends on the running llama-server build. The input plus
output context check still applies, and length-limited replies are still rejected.
This does not guarantee that a model will finish normally.

A recovered row records the actual settings in `request_json` and a distinct
`run_fingerprint`; the existing CSV schema is unchanged. Both baseline and this
specific recovery profile are validated when resuming and judging. Changed prompts
or arbitrary parameter edits are still rejected. Report the use of recovery settings
when describing the experiment; these rows did not use the baseline decoding settings.
