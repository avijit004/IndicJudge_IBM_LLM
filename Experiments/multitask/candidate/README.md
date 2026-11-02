# Candidate responses

| File | Responses |
|---|---:|
| [gemma26b_responses.csv](gemma26b_responses.csv) | 2,000 |
| [gemma31b_responses.csv](gemma31b_responses.csv) | 2,000 |

Each model has 50 responses for each of the 10 languages and 4 tasks. Both files cover the same input IDs, including the two approved replacement inputs.

Generation history is preserved in these subfolders:

- `multitask_50_samples/`: original generation scripts, settings and saved responses.
- `multitask_replacements/`: replacement generation and the merge workflow that creates the two CSVs above.
- `multitask_judge_experiment/`: original five-sample pilot, including its earlier judging results.

The earlier workflows are retained for reproducibility. Use `../judges/` for current reference-free judging.
