# 22-Language Fine-Tuning Dataset

Collects up to 100 samples per language x task from Hugging Face for the Qwen2.5-VL-7B and Gemma2-9B fine-tuning project, across 22 Indic languages and 4 tasks (Question Answering, Reasoning, Summarization, Translation Quality).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a Hugging Face account logged in (`hf auth login`) with access to `openlanguagedata/flores_plus` accepted on its dataset page — used for Translation.

## Run

```bash
source .venv/bin/activate
python scripts/collect.py            # pulls from HF into data/raw/<task_slug>/<iso>.csv + data/*.csv
python scripts/build_flat_csvs.py    # flattens into dataset_csv/<task>/<iso>.csv (lean per-task schema)
python scripts/build_coverage_excel.py  # regenerates dataset_coverage.xlsx
python scripts/verify_languages.py   # script/langid sanity check on dataset_csv/, writes language_verification_report.csv
```

All four are safe to re-run any time; every output file is overwritten from scratch. `scripts/probe_schema.py` is a standalone helper for checking a dataset's config names and field shapes on Hugging Face before wiring a new source into `config/sources.py` or `scripts/collect.py`.

## Output

- **`dataset_csv/<task>/<iso>.csv`** — the final deliverable: one folder per task (`qa`, `reasoning`, `summarization`, `translation`), one CSV per language, lean schema (e.g. `id, language, context, question, answers` for QA)
- **`dataset_coverage.xlsx`** — coverage matrix (language x task, green/red) + a Gaps sheet with reasons
- **`language_verification_report.csv`** — per-file script-match and langdetect-match percentages
- `data/raw/<task_slug>/<iso_code>.csv` — same data with full provenance columns (source dataset, config, resource tier)
- `data/<task_slug>.csv`, `data/all_tasks_combined.csv` — combined versions of the above
- `data/run_report.txt`, `data/collect_stdout.log` — full log of what was attempted per cell

## Coverage: 83 / 88 language x task cells filled (100 rows each, 8,300 rows total)

All 83 filled cells are **real, collected data** — nothing synthetic or generated. Verified twice:
1. `scripts/verify_languages.py` checks every file's actual script (Unicode range) and, for the 11 languages a language-ID model supports (hi, bn, mr, ta, te, gu, ur, kn, ml, pa, ne), cross-checks the specific language — not just "same script family." Current result: **0 flags across all 83 files.**
2. A native-script / language-marker preference filter in `collect.py` catches upstream sources that mix wrong-script or wrong-language rows into a language's data: found and fixed for Odia QA (22% romanized before the fix) and for Dogri QA (a handful of questions that leaked plain Hindi instead of Dogri, filtered out via a Dogri-vs-Hindi function-word heuristic since the two share Devanagari script and aren't distinguishable by Unicode range alone).

Translation Quality is fully covered for all 22 languages (via `openlanguagedata/flores_plus`, once its gated terms are accepted on your HF account).

Three gaps were closed via a deeper, targeted research pass (HF + arXiv/ACL Anthology + GitHub + Kaggle + academic paper search) that went beyond the original broad AI4Bharat/Google-org scan:
- **Nepali QA** — `Yunika/Nepali-QA` (individual HF upload, 266 SQuAD-style samples built from annotated Belebele/Flores-200 passages, CC-BY-SA-4.0) — adapter `qa_nepali_yunika`
- **Sindhi QA** and **Dogri QA** — `L3Cube-IndicQuest` (L3Cube Labs, Pune; Rohera et al., arXiv:2409.08706), ~200 native-script Q/A/Domain triples per language pulled directly from `github.com/l3cube-pune/indic-nlp` — adapters `qa_sindhi_indicquest` / `qa_dogri_indicquest`. **License note**: no LICENSE file found in the repo — treat as research-use only pending clarification from the authors before any commercial use.
- **Bhojpuri Reasoning** — `mrlbenchmarks/global-piqa-nonparallel` (`bho_deva` config; Watson et al., "Global PIQA," arXiv:2510.24081), 100 physical-commonsense MCQ items, CC-BY-SA-4.0 — adapter `reasoning_global_piqa`

### The 5 remaining gaps — confirmed unfillable from real sources after three research passes (HF, GitHub/arXiv/Kaggle, and academic paper search)

| Language | Missing task(s) | Why |
|---|---|---|
| Kashmiri (ks) | QA, Reasoning, Summarization | No dataset found anywhere (HF, arXiv, ACL Anthology, GitHub, Kaggle, university NLP labs) covering any of these three — Kashmiri NLP work that exists is pretraining corpora, OCR, or word-sense-disambiguation only |
| Sindhi (sd) | Summarization | Only classification/sentiment/instruction/lexicon datasets found, not document/summary pairs |
| Dogri (doi) | Summarization | One lead (`github.com/Sonam2/Dogri_Corpus`, a 2025 Springer/IJDL extractive-summarization paper, DOI 10.1007/s00799-025-00412-0) but only ~50 reference summaries confirmed, below the 100-sample bar; paper is paywalled so a larger release couldn't be confirmed |

**Decision**: these 5 gaps are left undata rather than filled synthetically, to keep the dataset 100% real/collected. If synthetic filling is wanted later (LLM-translated or LLM-generated content), it should go in a clearly separate column/file — never merged into these CSVs unlabeled.

### Non-commercial-licensed sources used

`bharatgenai/IndicParam` (Dogri and Bodo reasoning) and `google/IndicGenBench_crosssum_in` (most of the summarization data) are both CC-BY-NC — fine for research/eval, flag before any commercial fine-tune ships on this data.

## Schema

`dataset_csv/<task>/<iso>.csv` columns:
- **qa**: `id, language, context, question, answers` (`answers` is a JSON list)
- **reasoning**: `id, language, premise, question, choice1, choice2, choice3, choice4, label, prompt, answer` (COPA-style sources use premise/choices; IndicParam-sourced rows use prompt/answer instead)
- **summarization**: `id, language, document, summary`
- **translation**: `id, language, source_lang, target_lang, source_text, target_text`

`data/all_tasks_combined.csv` and the per-task `data/*.csv` carry the full superset plus `iso_code, resource_tier, task, source_dataset, source_config` for provenance.
