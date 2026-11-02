# Matched replacement batch

Replace the entire input pair, not just one model's failed answer:

| Task | Language | Removed input | Replacement (data row 51) |
|---|---|---|---|
| Summarization | Kannada | crosssum-kn-215 | crosssum-kn-490 |
| Summarization | Tamil | crosssum-ta-174 | crosssum-ta-116 |

Generate each replacement with both Gemmas: four new responses. Keep 1,998 original responses per Gemma and add two new responses per Gemma. This gives 2,000 matched inputs and 4,000 candidate responses. Both valid original counterpart responses remain in the old files for audit but are not included in the replacement evaluation.

The script reads generation prompts/settings and validators from `../multitask_50_samples/`. It checks source CSV hashes, freezes row 51 inputs and saves new results here only. Default decoding remains unchanged. No reference is sent to generation. Do not edit original manifests or CSVs.

From the project root on your Mac:

```bash
python3 Experiments/multitask/candidate/multitask_replacements/workflow.py prepare
python3 Experiments/multitask/candidate/multitask_replacements/workflow.py generate --model gemma26b --check-data
python3 Experiments/multitask/candidate/multitask_replacements/workflow.py generate --model gemma31b --check-data
```

With the relevant Gemma loaded on the server and the correct endpoint configured:

```bash
export MULTITASK_GEMMA26B_API_URL="http://172.30.1.83:8080/v1/chat/completions"
caffeinate -i python3 -u Experiments/multitask/candidate/multitask_replacements/workflow.py generate --model gemma26b

# After loading Gemma31B at its own endpoint (8084 here is an example):
export MULTITASK_GEMMA31B_API_URL="http://172.30.1.83:8084/v1/chat/completions"
caffeinate -i python3 -u Experiments/multitask/candidate/multitask_replacements/workflow.py generate --model gemma31b
```

Results save locally to `multitask_replacements/outputs/candidates/`. Resume with the same commands. Failures are not successful results. Explicit `--retry-length` / `--retry-format` applies only to prior recorded failures, with the same recovery profiles as the existing workflow. If a replacement also fails, investigate; do not silently select another input.

After both models show 2/2:

```bash
python3 Experiments/multitask/candidate/multitask_replacements/workflow.py status
python3 Experiments/multitask/candidate/multitask_replacements/workflow.py merge
python3 Experiments/multitask/judges/run_all.py --check-data \
  --snapshot Experiments/multitask/judges/data/inputs_4000.json \
  --output-dir Experiments/multitask/judges/outputs/judges_4000
```

Run each reference-free judge using BOTH overrides below, after loading it and setting its `RF_*_API_URL`:

```bash
caffeinate -i python3 -u Experiments/multitask/judges/judge_ministral.py \
  --snapshot Experiments/multitask/judges/data/inputs_4000.json \
  --output-dir Experiments/multitask/judges/outputs/judges_4000
```

Use the same overrides for `judge_qwen.py`, `judge_sarvam.py`, and `status.py`. Total target: 12,000 new judgments. The previous 3,998 snapshot/results remain separate; scores are not imported automatically.

Analysis:

```bash
python3 Experiments/multitask/judges/analyze.py \
  --snapshot Experiments/multitask/judges/data/inputs_4000.json \
  --output-dir Experiments/multitask/judges/outputs/judges_4000 \
  --analysis-dir Experiments/multitask/judges/analysis_4000
```

For later human review, export a fresh blank review package, then the matching offline references:

```bash
python3 Experiments/multitask/judges/review.py \
  --snapshot Experiments/multitask/judges/data/inputs_4000.json \
  --review-dir Experiments/multitask/judges/review_4000
python3 Experiments/multitask/candidate/multitask_replacements/workflow.py export-references
```

Do not use `review.py --include-references` for this merged selection: its original-source adapter does not represent the swapped inputs. Use the replacement export above. For human-grounded analysis add `--human-ratings Experiments/multitask/judges/review_4000/adjudicated.csv` to the analysis command.

Report the two swaps in research results. Replacing generation failures can change the difficulty distribution, and 50 inputs per cell still does not make this a random sample. The original `data/selection.json` uses a contiguous source range; the merged selection is explicitly different and must not be fed back into the old generation validator.

## Consolidated candidate CSVs

The `merge` command also writes `../gemma26b_responses.csv` and `../gemma31b_responses.csv`: exactly 2,000 responses each, with 50 per language/task. Each includes the two replacements and excludes both original swapped inputs. Original request fingerprints and response hashes are retained unchanged. These are consolidated exports of two validated batches; use the merged reference-free snapshot for judging, rather than putting these CSVs into the old contiguous-selection generation folder. Repeating `merge` verifies existing exports without overwriting different data.
