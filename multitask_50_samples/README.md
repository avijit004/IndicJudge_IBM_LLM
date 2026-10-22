# Multitask experiment: 50 samples

This self-contained folder uses the first **50 data records** from every language/task CSV in `../Dataset/dataset_csv/`. Header rows do not count. There are ten languages and four tasks: question answering, causal reasoning, summarization and translation.

| Stage | Full target | Reused first-five results | New work |
|---|---:|---:|---:|
| Inputs | 2,000 | 200 | 1,800 |
| Gemma responses, both models | 4,000 | 400 | 3,600 |
| Judgments, all three judges | 12,000 | 1,200 | 10,800 |

Gemma-4-26B and Gemma-4-31B generate responses. Sarvam-30B, Qwen3-14B and Ministral 3 8B Reasoning independently judge them, returning a score from 0–5 and a justification. The actual current progress comes from `status.py`, not this target table.

## Files and folders

| Path | Purpose |
|---|---|
| `prepare_data.py` | Validates source CSVs and freezes rows 1–50. |
| `data/selection.json` | All 2,000 selected inputs, references and source hashes. |
| `data/selection.csv` | Readable version of the frozen selection. |
| `prompts.py`, `RUBRICS.md` | Generation instructions and the four judge rubrics, unchanged from the first-five experiment. |
| `config.json` | The same model aliases, decoding settings and endpoint defaults. Environment variables can override endpoints. |
| `generate_gemma26b.py`, `generate_gemma31b.py` | Generate missing responses for each candidate model. |
| `judge_sarvam.py`, `judge_qwen.py`, `judge_ministral.py` | Judge missing records for each judge. |
| `run_all.py` | Runs a requested generation/judging stage sequentially. |
| `common.py`, `data.py`, `generate.py`, `judge.py` | Local transport, validation, checkpoint and workflow code. |
| `import_completed.py` | Imports validated first-five results once, before new output exists. |
| `data/checkpoint_origins/` | Original selection manifests used to validate imported checkpoint fingerprints. |
| `outputs/candidates/`, `outputs/judges/` | All imported and newly generated results in this folder. |
| `outputs/import_provenance.json` | Records original file hashes and imported counts. |
| `status.py` | Reports validated completed and remaining counts. |
| `analyze.py`, `analysis/` | Exports matched scores, task means, pairwise agreement and coverage. |
| `tests/` | Offline workflow and checkpoint reuse tests. |

## Prepare and reuse

The selection and imported results have already been prepared in this workspace. Preparation is safe to repeat. Do not rerun the import once `outputs/` exists.

For a fresh setup only:

```bash
python3 multitask_50_samples/prepare_data.py
python3 multitask_50_samples/import_completed.py
```

The original `multitask_judge_experiment/` remains unchanged. Imported CSVs retain their exact requests, timestamps, fingerprints and response hashes. Their original selection is an explicitly validated subset of the new selection. A fingerprint from the first-five run cannot validate a row outside that first-five subset.

## Generate the remaining responses

Run from the project root. Load the appropriate model into llama-server first. Keep the same weights, quantization, server build and chat template used in the first batch; an alias alone cannot verify these.

Set the corresponding environment variable to the actual running endpoint:

- `MULTITASK_GEMMA26B_API_URL`
- `MULTITASK_GEMMA31B_API_URL`
- `MULTITASK_SARVAM_API_URL`
- `MULTITASK_QWEN_API_URL`
- `MULTITASK_MINISTRAL_API_URL`

For example, if Gemma-4-26B is running on port 8080:

```bash
export MULTITASK_GEMMA26B_API_URL="http://172.30.1.83:8080/v1/chat/completions"
python3 -u multitask_50_samples/generate_gemma26b.py
```

Then, with the 31B model loaded and its endpoint configured:

```bash
python3 -u multitask_50_samples/generate_gemma31b.py
```

Both candidates can run concurrently on separate servers. Each command skips imported and newly completed records. On macOS, prefix the Python command with `caffeinate -i` to prevent idle sleep.

## Judge and analyze

With each judge's correct server loaded and endpoint configured, run:

```bash
python3 -u multitask_50_samples/judge_sarvam.py
python3 -u multitask_50_samples/judge_qwen.py
python3 -u multitask_50_samples/judge_ministral.py
python3 multitask_50_samples/status.py
python3 multitask_50_samples/analyze.py
```

If all relevant servers are already running, the equivalent stage commands are:

```bash
python3 -u multitask_50_samples/run_all.py generate
python3 -u multitask_50_samples/run_all.py judge
```

These scripts call servers; they do not boot or switch models. Repeating generation/judging commands resumes unfinished work. Avoid concurrent writers for the same model CSV; file locks enforce this.

All generation/judge entry points accept `--check-data` for offline validation, `--languages`, `--tasks` and `--limit`. Status accepts `--detail`.

## Comparability and references

Candidate generation receives no gold references. QA judging uses context and reference answers. Reasoning uses the problem, available candidate reasoning and visible response, without a gold option label. Summarization uses the source and summary, without a reference summary. Translation uses the source, language direction and a supplementary reference translation.

Baseline prompts, rubrics and settings match the original experiment. A few original candidates used the existing explicit length-recovery profile. Their actual request settings remain preserved. For new failures, `--retry-length` uses the same opt-in profile only for records with a recorded length failure; normal new samples use baseline settings.

Analysis joins on `sample_id` and `source_model`, keeping judge columns separate. Pending scores stay blank, never zero. Mean scores use available records and show coverage. Pairwise comparisons use only keys rated by both judges. Pairwise agreement is not judge accuracy. These are the first 50 records in source order, not a random sample.

## Retrying Gemma format errors

For missing records whose `.errors.jsonl` contains the exact HTTP 500 `peg-gemma4` format failure:

```bash
python3 -u multitask_50_samples/generate_gemma26b.py --retry-format --retry-length
```

The format profile uses seed 44, a 16,384-token output limit and the existing requested 4,096-token reasoning budget. It takes precedence over length recovery if both failure types were recorded for a sample. This is a new sampling attempt, not a guaranteed fix for the server parser. Parsing stays enabled; incomplete or invalid replies are never saved as successes. Completed records and ordinary new requests are unchanged. Actual recovery settings are retained in `request_json` and the profile fingerprint.

Keep `common.py` and `generate.py` together when syncing this change to the server. The local dashboard also needs the updated validator to recognize recovered records.

For a persistent recorded `peg-gemma4` format failure, use `--retry-format --format-retry-seed 45` (allowed seeds: 44, 45, 46; default: 44). This makes a new sampling attempt, not a parser fix. Only missing records with a matching recorded format error use this profile. Saved responses are skipped; the request and fingerprint retain the actual recovery settings.
