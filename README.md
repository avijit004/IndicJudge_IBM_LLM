# Qwen3-14B response judging

Ten standalone language entrypoints evaluate the same 5,000 Gemma responses as the Sarvam evaluation: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu. Each language has 500 responses from the two candidate models.

The mathematical rubric, prompt intents, original generation prompts, ground truth, and 15 CSV columns are preserved. Answer correctness uses the visible response; reasoning quality assesses the internal reasoning and visible response; compliance uses the original generation prompt. Existing salvaged-response markers are preserved. Candidate text is never truncated.

## Server and environment

Requires Python 3.9+ and `requests`. Use `/opt/anaconda3/bin/python3` on the current Mac; it already has requests. No server is started by these scripts.

Load **Qwen/Qwen3-14B**, not its Base version, using a compatible GGUF and llama-server. Set the server alias to **Qwen3-14B** (`--alias Qwen3-14B`). Use the model's Jinja chat template (`--jinja`), reasoning extraction (`--reasoning-format deepseek`), and at least 32,768 context tokens per slot (for a sequential server: `-c 32768 -np 1`). The server must support `reasoning_budget_tokens` and JSON-schema output, as the Sarvam server does. Confirm actual behavior with the pilot after loading the model.

The default endpoint is **http://172.30.1.83:8081/v1/chat/completions**. Port 8081 deliberately differs from the existing Sarvam endpoint. To use a different endpoint, set this before either a language script or `run_all.py`:

```bash
export QWEN_API_URL="http://172.30.1.83:8081/v1/chat/completions"
```

Preflight checks `/v1/models` for the required alias and `/props` for context capacity. The alias check detects accidental connection to a differently named server; it cannot verify that an arbitrarily aliased GGUF really contains Qwen. Load the correct checkpoint yourself.

Input paths resolve from the parent of this folder, regardless of the launch directory. Set `JUDGEBENCH_DATA_DIR` only if the experiment directories are elsewhere. Results always resolve relative to each script.

## Run commands

From the project root:

```bash
# Offline validation: no inference and no result writes.
/opt/anaconda3/bin/python3 qwen3_14b_judge/run_all.py --check-data

# After loading Qwen: 50 judgments, one fixed response per P1–P5 per language.
caffeinate -i /opt/anaconda3/bin/python3 -u qwen3_14b_judge/run_all.py --pilot

# After inspecting the pilot output: all 5,000 responses, skipping saved judgments.
caffeinate -i /opt/anaconda3/bin/python3 -u qwen3_14b_judge/run_all.py

# Individual language or a selection:
/opt/anaconda3/bin/python3 -u qwen3_14b_judge/qwen3_14b_judge_telugu.py
/opt/anaconda3/bin/python3 -u qwen3_14b_judge/run_all.py --languages tamil telugu
```

`caffeinate` is optional and macOS-specific. Use your Python interpreter without it on Linux.

Pilot selection is fixed before checkpoint filtering: rerunning the pilot does not select additional candidates. With the current CSV order it selects Gemma-4-26B question 1, P1–P5. Valid pilot judgments are saved to the normal Qwen results and reused in the full run. A successful technical pilot verifies response handling, not agreement with human reviewers.

The all-language runner continues to later languages if a language fails, then exits nonzero and lists the failed languages. An individual script also exits nonzero if any selected judgment fails. Rerun the same command to retry unresolved rows. Do not run two processes writing the same language's CSV simultaneously.

## Generation configuration

All ten scripts use the same configuration:

- Thinking enabled; temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42.
- Maximum output 8,192 tokens, including thinking; llama-server thinking budget 4,096 tokens.
- JSON schema with a nonempty justification of at most 1,200 characters; prompt requests 2–4 concise sentences.
- 300-second request timeout, no silent fallback to different generation settings.

The sampling settings follow [Qwen's thinking-mode recommendations](https://huggingface.co/Qwen/Qwen3-14B#best-practices). The output and thinking limits are **project-specific bounds**, not Qwen's recommended unrestricted mathematical benchmarking budget. Validate them in the live pilot; an exhausted output budget is rejected, never converted into a saved judgment. A fixed seed does not guarantee identical output across hardware/server versions. Record the actual GGUF, quantization, server version, and settings when reporting your experiment.

## Output and validation

Each script creates `qwen3_14b_judge_<language>_results.csv` in this folder. The existing Sarvam directory and its results are not accessed during normal judging.

All reference answers and candidate IDs are checked before inference. References come from `answer_number`, or the final number following `####` in `answer`. Invalid judgments—including incomplete generations, fractional/boolean scores, and malformed justifications—are not saved. Thinking is stored separately from final JSON, whether the server returns it as `reasoning_content` or an inline `<think>` block.

Checkpoint recovery preserves valid rows and backs up the original before repairing an interrupted or malformed file. Successful rows are skipped; failed judgments stay eligible for retry.

## Offline tests

```bash
/opt/anaconda3/bin/python3 -m unittest discover -s qwen3_14b_judge -p 'test_*.py'
```

Tests use mocked API replies and temporary outputs. They cover all 10 datasets/5,000 candidates, P1–P5 payloads, reference parity with `sarvam_30b_judge`, validation failures, inline thinking, wrong server aliases/context, resumption, failure exit status, and paths from another working directory. No live Qwen inference was performed when these scripts were created.
