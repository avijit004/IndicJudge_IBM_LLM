# Ministral 3 8B Reasoning — five-question evaluation

One entrypoint per language evaluates **five question IDs × P1–P5 × both Gemma models**:

- 25 judgments per candidate model per language.
- 50 judgments per language.
- **500 judgments across 10 languages**, not the full 5,000-response dataset.

The default question IDs are **1, 2, 3, 4, 5** in each language. This is a fixed first-five subset, not a random or statistically representative sample. Selection happens before checkpoint filtering, so rerunning never selects new questions to replace saved ones. Each selected question must have every P1–P5 response from every requested source model; missing or duplicate rows stop the run before inference.

The default includes Gemma-4-26B and Gemma-4-31B, as requested. `--source-model 26b` or `--source-model 31b` can restrict a run to 250 judgments across ten languages. `--question-ids` accepts exactly five distinct IDs. The same numeric IDs across datasets do not necessarily identify the same underlying mathematical problem.

## Preserved evaluation

The user-message rubric, prompt intents, ground-truth loading and CSV columns match the Qwen judge scripts. All candidate text and original generation prompts are passed unchanged. The three ratings remain `answer_correct`, `reasoning_quality` (integer 0–5), and `follows_prompt_intent`, with justification. Ground truth comes from `answer_number`, or the final numerical answer after `####` in `answer`. Responses marked as salvaged from reasoning remain unchanged.

Compare Ministral ratings with Sarvam/Qwen on these same selected keys `(language, source_model, question_id, prompt_id)`, rather than comparing subset averages with their full-dataset averages.

## Server configuration

Load **mistralai/Ministral-3-8B-Reasoning-2512** (Reasoning, not Instruct or Base) using a compatible llama-server build and its GGUF chat template. Required alias:

```text
Ministral-3-8B-Reasoning
```

Use `--jinja --reasoning-format deepseek --alias Ministral-3-8B-Reasoning` and at least 32,768 context tokens per slot (`-c 32768 -np 1` for sequential requests). The `deepseek` output format tells llama-server to return recognized thoughts in `reasoning_content`; it does not change the selected model. The scripts also handle native `[THINK]...[/THINK]` or `<think>...</think>` blocks returned inline. A current server must recognize Ministral's template and support JSON schema plus the reasoning-budget parameters.

Set the endpoint explicitly in the terminal where you run the scripts:

```bash
# Replace PORT with the port printed by your Ministral server.
export MINISTRAL_API_URL="http://172.30.1.83:PORT/v1/chat/completions"
```

There is no default port, to avoid accidentally connecting to Qwen or another service. Preflight verifies the alias via `/v1/models` and context via `/props`. An alias cannot independently prove which checkpoint was loaded: load the correct GGUF.

## Run

From the local project root:

```bash
cd /Users/avijit/Desktop/IndicJudge_IBM

# Offline input/reference check: no server connection and no result writes.
/opt/anaconda3/bin/python3 -B ministral_8b_judge/run_all.py --check-data

# After setting MINISTRAL_API_URL: test one language first (50 judgments).
caffeinate -i /opt/anaconda3/bin/python3 -u \
  ministral_8b_judge/ministral_8b_judge_tamil.py

# Run the selected five questions in all ten languages (500 total).
# Completed rows from the single-language run are skipped.
caffeinate -i /opt/anaconda3/bin/python3 -u \
  ministral_8b_judge/run_all.py
```

Other selections, if desired:

```bash
/opt/anaconda3/bin/python3 -u ministral_8b_judge/run_all.py --languages tamil telugu
/opt/anaconda3/bin/python3 -u ministral_8b_judge/run_all.py --source-model 31b
/opt/anaconda3/bin/python3 -u ministral_8b_judge/run_all.py --question-ids 6 7 8 9 10 --check-data
```

Changing IDs and running inference adds those judgments to the same language CSV; it does not remove previously saved rows. Keep one selection for the experiment unless intentionally expanding it.

On Linux use the installed Python interpreter with `requests`, and omit macOS `caffeinate`. Input files resolve relative to the project, so folder renames and launches from another working directory work. `JUDGEBENCH_DATA_DIR` optionally overrides the data root. Outputs always resolve relative to the language script.

## Output and resumption

Each script writes `ministral_8b_judge_<language>_results.csv` in this folder. Existing Qwen/Sarvam results are never opened for writing.

Valid saved rows are skipped on rerun. Invalid JSON, truncated generations (`finish_reason=length`), fractional/boolean scores, invalid justifications and HTTP errors are not saved; the run reports failure and exits nonzero. Failed rows remain retryable. The runner continues to subsequent languages and reports any failed languages at the end. Interrupted checkpoint repair preserves a backup before retaining complete rows. Avoid concurrent processes writing the same language CSV.

## Generation settings and limits

- Temperature **0.7**, following the [model card recommendation](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512#recommended-settings).
- Top-p 1, top-k 0 (disabled), min-p 0 and seed 42 are explicit project settings, not vendor benchmark settings.
- Project-specific output limit 8,192 tokens, including a 4,096-token reasoning budget; 300-second HTTP timeout.
- Final JSON schema limits justification to 1,200 characters and requires all four judgment fields.
- A short task-specific system prompt requests Ministral's native thinking delimiters and final JSON. It adapts the [published format](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512/blob/main/SYSTEM_PROMPT.txt); it does not use the full general-chat system prompt or its Markdown-answer instruction.
- No Qwen-specific `enable_thinking` flag is sent.

These bounded settings need a live check after the model is loaded. Offline mocks establish code correctness, not model accuracy or server compatibility. No live judging was started when creating this folder. Human comparisons are still needed to assess judgments in each language.

## Tests

```bash
/opt/anaconda3/bin/python3 -B -m unittest discover \
  -s ministral_8b_judge -p 'test_*.py' -v
```

Tests use mock server responses and temporary output files: 500 selected judgments across all ten datasets; all-five-prompt coverage; single-model scope; rubric/reference parity; literal candidate preservation; native thinking parsing; strict result validation; missing selected rows; failure/retry counts; checkpoint recovery; server alias/context; and runner options.
