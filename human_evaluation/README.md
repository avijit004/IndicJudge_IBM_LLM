# Human evaluation package

The sample contains 200 Gemma responses: two randomly selected responses for each of the 100 language × model × prompt combinations. Seed: `20260925`. Each language has 20 items. Both reviewers receive the same items and order, with model labels and Sarvam judgments omitted. Candidate text is preserved verbatim, including any self-identification it may contain.

## Distribute

- Give reviewer 1 `reviewer_1.xlsx` and `reading_pack.html`.
- Give reviewer 2 `reviewer_2.xlsx` and `reading_pack.html`.
- Keep `coordinator.xlsx` private until independent reviews are complete.

Each reviewer workbook includes an Instructions tab with the existing rubric and one tab per language. Assign each language to reviewers who can evaluate it confidently. The yellow columns are for human input. The reading pack opens locally in a browser and displays complete long responses; search for the evaluation ID to locate an item. No network connection is required.

## Review

Evaluate both internal reasoning and visible responses. Enter reviewer ID, answer correctness (True/False), reasoning quality (integer 0–5), prompt compliance (True/False), and a short justification. Mark status Complete only after all ratings and justification are filled. Use Needs review or Cannot assess for unresolved issues, and explain them in notes. Ratings are intentionally blank; reference answers are not human ratings.

Use a separate small practice batch to align rubric interpretations before independent evaluation. Do not discuss or view another reviewer's ratings before submitting your own. Sarvam scoring is not required to begin this human evaluation.

## Reconcile

Match returned workbooks by `evaluation_id`, not row number. Copy each reviewer's original ratings into the Consensus tab in `coordinator.xlsx`, retain them, and enter consensus ratings after discussion or third-reviewer adjudication. Sample key maps each evaluation ID to language, model, question ID, prompt ID, source CSV, reference answer, and source-file SHA-256 hash.

The package does not automatically import returned ratings or calculate agreement. Once ratings are available, compare completed items with Sarvam and report reviewer agreement, answer-correctness and compliance agreement, and reasoning-score mean absolute error. Two items per stratum provide an initial check, not precise per-stratum estimates.

## Reproducibility

`prepare_sheets.py` uses Python and `openpyxl` to generate the package and verify sampling balance, complete text, blank rating cells, validation rules, and matching keys. It refuses to overwrite existing workbooks to protect human ratings. No judge scripts or source datasets are modified, and no model API is called.
