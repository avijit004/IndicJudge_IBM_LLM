#!/usr/bin/env python
"""
Main collection script.

For each (language, task) cell, tries data sources in priority order (see
config/sources.py) until one yields >=100 usable rows, normalizes into a
common per-task schema, and writes:
  - data/raw/<task_slug>/<iso_code>.csv        (per language, per task)
  - data/<task_slug>.csv                        (all languages, one task)
  - data/all_tasks_combined.csv                 (everything, superset columns)

Re-run any time; every output file is overwritten from scratch.

Usage:
    source .venv/bin/activate && python scripts/collect.py
"""
import io
import json
import os
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import _compat  # noqa: F401  must be imported before any `datasets` use

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.sources import (  # noqa: E402
    LANGUAGES, ISO_TO_NAME, ISO_TO_TIER, TASK_LABELS,
    XORQA_IN_LANGS, XQUAD_IN_LANGS, INDICQA_LANGS, TYDIQA_LANGS, XQUAD_GOOGLE_LANGS,
    INDICCOPA_LANGS, INDICPARAM_SUBJECT_TO_ISO, INDICXNLI_LANGS, GLOBAL_MMLU_LANGS,
    CROSSSUM_IN_LANGS, INDIC_HEADLINE_LANGS, XLSUM_LANG_NAMES, CROSSSUM_HF_LANG_NAMES,
    INDICWIKIBIO_LANGS, FLORES_IN_LANGS, SAMANANTAR_LANGS,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
N_SAMPLES = 100
SEED = 42

QA_COLUMNS = ["id", "language", "iso_code", "resource_tier", "task",
              "source_dataset", "source_config", "question", "context", "answer"]
REASONING_COLUMNS = ["id", "language", "iso_code", "resource_tier", "task",
                      "source_dataset", "source_config", "premise", "question",
                      "choice1", "choice2", "choice3", "choice4", "label",
                      "prompt", "answer"]
SUMM_COLUMNS = ["id", "language", "iso_code", "resource_tier", "task",
                "source_dataset", "source_config", "document", "summary"]
TRANSLATION_COLUMNS = ["id", "language", "iso_code", "resource_tier", "task",
                        "source_dataset", "source_config", "source_text",
                        "target_text", "source_lang", "target_lang"]

TASK_COLUMNS = {
    "qa": QA_COLUMNS,
    "reasoning": REASONING_COLUMNS,
    "summarization": SUMM_COLUMNS,
    "translation": TRANSLATION_COLUMNS,
}
TASK_SLUG = {
    "qa": "question_answering",
    "reasoning": "reasoning",
    "summarization": "summarization",
    "translation": "translation_quality",
}

LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


def sample_rows(rows, n=N_SAMPLES, seed=SEED):
    if len(rows) <= n:
        return rows
    return random.Random(seed).sample(rows, n)


# Some upstream sources (e.g. IndicGenBench's xorqa_in) mix romanized/Latin-script
# rows in with native-script ones for the same language. Prefer native-script rows
# when there are more than enough to fill N_SAMPLES, so a 100-row sample doesn't
# quietly end up 20% romanized just because random.sample() picked those rows.
_SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)], "mr": [(0x0900, 0x097F)], "ne": [(0x0900, 0x097F)],
    "mai": [(0x0900, 0x097F)], "bho": [(0x0900, 0x097F)], "gom": [(0x0900, 0x097F)],
    "doi": [(0x0900, 0x097F)], "sa": [(0x0900, 0x097F)], "brx": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)], "as": [(0x0980, 0x09FF)],
    "te": [(0x0C00, 0x0C7F)], "ta": [(0x0B80, 0x0BFF)], "gu": [(0x0A80, 0x0AFF)],
    "kn": [(0x0C80, 0x0CFF)], "ml": [(0x0D00, 0x0D7F)], "pa": [(0x0A00, 0x0A7F)],
    "or": [(0x0B00, 0x0B7F)], "sat": [(0x1C50, 0x1C7F)],
    "ur": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    "sd": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    "ks": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
}

_TASK_TEXT_FIELD = {"qa": "question", "reasoning": "premise", "summarization": "summary",
                     "translation": "source_text"}


def _is_native_script(text, iso):
    ranges = _SCRIPT_RANGES.get(iso)
    if not ranges or not isinstance(text, str):
        return True  # nothing to check against (e.g. Sanskrit is in ranges though) -- default to keep
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    in_script = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return (in_script / len(letters)) >= 0.6


def prefer_native_script(rows, task, iso, n=N_SAMPLES):
    field = _TASK_TEXT_FIELD.get(task)
    if field is None or iso not in _SCRIPT_RANGES or len(rows) <= n:
        return rows
    native = [r for r in rows if _is_native_script(r.get(field, "") or r.get("prompt", ""), iso)]
    romanized = [r for r in rows if r not in native]
    if len(native) >= n:
        return native
    return native + romanized  # not enough native rows -- keep the rest as fallback


def base_row(iso, task):
    return {
        "language": ISO_TO_NAME[iso],
        "iso_code": iso,
        "resource_tier": ISO_TO_TIER[iso],
        "task": TASK_LABELS[task],
    }


# ---------------------------------------------------------------------------
# IndicGenBench raw-JSON helpers (bypasses broken/deprecated loading scripts;
# these repos are plain per-language JSON files, downloaded directly).
# ---------------------------------------------------------------------------
_repo_file_cache = {}


def _repo_files(repo):
    if repo not in _repo_file_cache:
        _repo_file_cache[repo] = list_repo_files(repo, repo_type="dataset")
    return _repo_file_cache[repo]


def _igb_load_examples(repo, filename, limit=None):
    path = hf_hub_download(repo, filename, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = data.get("examples", [])
    return examples[:limit] if limit else examples


def _igb_collect(repo, candidate_filenames, need=N_SAMPLES):
    """Try filenames in order (e.g. test, dev, train splits), accumulate
    examples across them until `need` reached; skip files that don't exist."""
    files = set(_repo_files(repo))
    collected = []
    for fn in candidate_filenames:
        if fn not in files:
            continue
        try:
            collected.extend(_igb_load_examples(repo, fn))
        except Exception as e:
            log(f"    [warn] failed to read {repo}/{fn}: {e}")
        if len(collected) >= need:
            break
    return collected


# ---------------------- QA adapters ----------------------

def qa_xorqa_in(iso):
    repo = "google/IndicGenBench_xorqa_in"
    if iso not in XORQA_IN_LANGS:
        return None
    exs = _igb_collect(repo, [f"xorqa_{iso}_test.json", f"xorqa_{iso}_dev.json", f"xorqa_{iso}_train.json"])
    rows = []
    for i, ex in enumerate(exs):
        answers = ex.get("answers") or []
        answer = answers[0]["text"] if answers else ""
        rows.append({"id": f"xorqa-{iso}-{i}", "question": ex.get("question", ""),
                     "context": ex.get("context", ""), "answer": answer})
    return (repo, "default", rows)


def qa_xquad_in(iso):
    repo = "google/IndicGenBench_xquad_in"
    if iso not in XQUAD_IN_LANGS:
        return None
    exs = _igb_collect(repo, [f"xquad_{iso}_test.json", f"xquad_{iso}_dev.json", f"xquad_{iso}_train.json"])
    rows = []
    for i, ex in enumerate(exs):
        answers = ex.get("answers") or []
        answer = answers[0]["text"] if answers else ""
        rows.append({"id": ex.get("id", f"xquad-{iso}-{i}"), "question": ex.get("question", ""),
                     "context": ex.get("context", ""), "answer": answer})
    return (repo, "default", rows)


def qa_indicqa(iso):
    if iso not in INDICQA_LANGS:
        return None
    repo = "ai4bharat/IndicQA"
    cfg = f"indicqa.{iso}"
    ds = load_dataset(repo, cfg, split="test")
    rows = []
    for ex in ds:
        ans = ex.get("answers") or {}
        texts = ans.get("text") or []
        answer = texts[0] if texts else ""
        rows.append({"id": str(ex.get("id", "")), "question": ex.get("question", ""),
                     "context": ex.get("context", ""), "answer": answer})
    return (repo, cfg, rows)


def qa_tydiqa(iso):
    if iso not in TYDIQA_LANGS:
        return None
    repo = "google-research-datasets/tydiqa"
    cfg = "secondary_task"
    lang_name = TYDIQA_LANGS[iso]
    ds = load_dataset(repo, cfg, split="validation", streaming=True)
    rows = []
    for ex in ds:
        if ex.get("id", "").split("-")[0].lower() != lang_name and lang_name not in str(ex.get("id", "")).lower():
            # tydiqa secondary_task has no explicit language column; ids are like
            # "bengali--<hash>". Fall back to checking the id prefix.
            if not str(ex.get("id", "")).lower().startswith(lang_name):
                continue
        ans = ex.get("answers") or {}
        texts = ans.get("text") or []
        answer = texts[0] if texts else ""
        rows.append({"id": ex.get("id", ""), "question": ex.get("question", ""),
                     "context": ex.get("context", ""), "answer": answer})
        if len(rows) >= N_SAMPLES:
            break
    return (repo, cfg, rows)


def qa_xquad_google(iso):
    if iso not in XQUAD_GOOGLE_LANGS:
        return None
    repo = "google/xquad"
    cfg = f"xquad.{iso}"
    ds = load_dataset(repo, cfg, split="validation")
    rows = []
    for ex in ds:
        ans = ex.get("answers") or {}
        texts = ans.get("text") or []
        answer = texts[0] if texts else ""
        rows.append({"id": str(ex.get("id", "")), "question": ex.get("question", ""),
                     "context": ex.get("context", ""), "answer": answer})
    return (repo, cfg, rows)


def qa_nepali_yunika(iso):
    if iso != "ne":
        return None
    repo = "Yunika/Nepali-QA"
    ds = load_dataset(repo, split="train")
    rows = []
    for ex in ds:
        d = ex.get("data") or {}
        ans = d.get("answers") or {}
        texts = ans.get("text") or []
        answer = texts[0] if texts else ""
        rows.append({"id": f"npqa-ne-{d.get('id', len(rows))}", "question": d.get("question", ""),
                     "context": d.get("context", ""), "answer": answer})
    return (repo, "default", rows)


def _fetch_l3cube_indicquest_csv(lang_file):
    import csv as _csv
    import urllib.request
    url = f"https://raw.githubusercontent.com/l3cube-pune/indic-nlp/main/L3Cube-IndicQuest/{lang_file}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    return list(_csv.DictReader(io.StringIO(text)))


def qa_sindhi_indicquest(iso):
    if iso != "sd":
        return None
    repo = "l3cube-pune/indic-nlp (L3Cube-IndicQuest)"
    sindhi_unique = "ڄٺٽڦڪڳڃڇڏڍڙڻۿ"
    try:
        raw_rows = _fetch_l3cube_indicquest_csv("Sindhi.csv")
    except Exception:
        return (repo, "Sindhi.csv", [])
    rows = []
    for i, r in enumerate(raw_rows):
        q = (r.get("Question") or "").strip()
        if not any(c in sindhi_unique for c in q):
            continue  # likely not distinctively Sindhi script -- skip, prefer clean rows
        rows.append({"id": f"indicquest-sd-{i}", "question": q, "context": "",
                     "answer": (r.get("Answer") or "").strip()})
    if len(rows) < N_SAMPLES:  # fall back to including the skipped rows if short
        for i, r in enumerate(raw_rows):
            q = (r.get("Question") or "").strip()
            if any(c in sindhi_unique for c in q):
                continue
            rows.append({"id": f"indicquest-sd-{i}", "question": q, "context": "",
                         "answer": (r.get("Answer") or "").strip()})
    return (repo, "Sindhi.csv", rows)


def qa_dogri_indicquest(iso):
    if iso != "doi":
        return None
    repo = "l3cube-pune/indic-nlp (L3Cube-IndicQuest)"
    dogri_markers = ["गी", "कन्नै", "ते ", "दा ", "दे ", "दी ", "ऐ", "हा", "हे",
                      "थमां", "तगर", "इक ", "बी ", "कुन", "आं", "जंदा", "होंदा"]
    hindi_only_markers = [" को ", " से ", " है", " था ", " हूँ", " और ", " का ", " के ", " की ", " में "]

    def looks_hindi_only(text):
        d = sum(1 for m in dogri_markers if m in text)
        h = sum(1 for m in hindi_only_markers if m in text)
        return d == 0 and h >= 1

    try:
        raw_rows = _fetch_l3cube_indicquest_csv("Dogri.csv")
    except Exception:
        return (repo, "Dogri.csv", [])
    clean = [r for r in raw_rows if not looks_hindi_only((r.get("Question") or ""))]
    pool = clean if len(clean) >= N_SAMPLES else raw_rows
    rows = []
    for i, r in enumerate(pool):
        rows.append({"id": f"indicquest-doi-{i}", "question": (r.get("Question") or "").strip(),
                     "context": "", "answer": (r.get("Answer") or "").strip()})
    return (repo, "Dogri.csv", rows)


QA_SOURCES = [qa_xorqa_in, qa_xquad_in, qa_indicqa, qa_tydiqa, qa_xquad_google, qa_nepali_yunika,
              qa_sindhi_indicquest, qa_dogri_indicquest]


# ---------------------- Reasoning adapters ----------------------

def reasoning_copa(iso):
    if iso not in INDICCOPA_LANGS:
        return None
    repo = "ai4bharat/IndicCOPA"
    cfg = f"translation-{iso}"
    ds = load_dataset(repo, cfg, split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"copa-{iso}-{ex.get('idx', i)}", "premise": ex.get("premise", ""),
            "question": ex.get("question", ""), "choice1": ex.get("choice1", ""),
            "choice2": ex.get("choice2", ""), "choice3": "", "choice4": "",
            "label": ex.get("label", ""), "prompt": "", "answer": "",
        })
    return (repo, cfg, rows)


def reasoning_indicparam(iso):
    subject = next((s for s, i in INDICPARAM_SUBJECT_TO_ISO.items() if i == iso), None)
    if subject is None:
        return None
    repo = "bharatgenai/IndicParam"
    cfg = "IndicParam"
    ds = load_dataset(repo, cfg, split="test", streaming=True)
    rows = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if ex.get("subject") != subject:
            if scanned > 20000:
                break
            continue
        q = ex.get("question_text", "")
        opts = [ex.get("option_a", ""), ex.get("option_b", ""),
                ex.get("option_c", ""), ex.get("option_d", "")]
        answer_letter = str(ex.get("correct_answer", "")).strip().lower()
        rows.append({
            "id": ex.get("unique_question_id", f"indicparam-{iso}-{len(rows)}"),
            "premise": "", "question": q,
            "choice1": opts[0], "choice2": opts[1], "choice3": opts[2], "choice4": opts[3],
            "label": answer_letter, "prompt": q, "answer": answer_letter,
        })
        if len(rows) >= N_SAMPLES or scanned > 20000:
            break
    return (repo, cfg, rows)


def reasoning_indicxnli(iso):
    if iso not in INDICXNLI_LANGS:
        return None
    repo = "Divyanshu/indicxnli"
    cfg = iso
    ds = load_dataset(repo, cfg, split="validation")
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    rows = []
    for i, ex in enumerate(ds):
        lbl = ex.get("label", "")
        rows.append({
            "id": f"xnli-{iso}-{i}", "premise": ex.get("premise", ""),
            "question": ex.get("hypothesis", ""), "choice1": "entailment",
            "choice2": "neutral", "choice3": "contradiction", "choice4": "",
            "label": lbl,
            "prompt": f"Premise: {ex.get('premise', '')}\nHypothesis: {ex.get('hypothesis', '')}\nRelation?",
            "answer": label_map.get(lbl, str(lbl)),
        })
    return (repo, cfg, rows)


def reasoning_global_mmlu(iso):
    if iso not in GLOBAL_MMLU_LANGS:
        return None
    repo = "CohereLabs/Global-MMLU"
    cfg = iso
    ds = load_dataset(repo, cfg, split="test")
    rows = []
    for ex in ds:
        q = ex.get("question", "")
        rows.append({
            "id": ex.get("sample_id", ""), "premise": "", "question": q,
            "choice1": ex.get("option_a", ""), "choice2": ex.get("option_b", ""),
            "choice3": ex.get("option_c", ""), "choice4": ex.get("option_d", ""),
            "label": ex.get("answer", ""), "prompt": q, "answer": ex.get("answer", ""),
        })
    return (repo, cfg, rows)


def reasoning_milu(iso):
    repo = "ai4bharat/MILU"
    ds = load_dataset(repo, iso, split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({"id": f"milu-{iso}-{i}", "premise": "", "question": str(ex),
                      "choice1": "", "choice2": "", "choice3": "", "choice4": "",
                      "label": "", "prompt": str(ex), "answer": ""})
    return (repo, iso, rows)


def reasoning_global_piqa(iso):
    if iso != "bho":
        return None
    repo = "mrlbenchmarks/global-piqa-nonparallel"
    cfg = "bho_deva"
    try:
        ds = load_dataset(repo, cfg, split="test")
    except Exception:
        return (repo, cfg, [])
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": ex.get("example_id", f"piqa-bho-{i}"), "premise": ex.get("prompt", ""),
            "question": "physical_commonsense", "choice1": ex.get("solution0", ""),
            "choice2": ex.get("solution1", ""), "choice3": "", "choice4": "",
            "label": str(ex.get("label", "")), "prompt": "", "answer": "",
        })
    return (repo, cfg, rows)


REASONING_SOURCES = [reasoning_copa, reasoning_indicparam, reasoning_indicxnli,
                      reasoning_global_mmlu, reasoning_milu, reasoning_global_piqa]


# ---------------------- Summarization adapters ----------------------

def summ_crosssum_in(iso):
    if iso not in CROSSSUM_IN_LANGS:
        return None
    repo = "google/IndicGenBench_crosssum_in"
    exs = _igb_collect(repo, [f"crosssum_english-{iso}_test.json",
                               f"crosssum_english-{iso}_dev.json",
                               f"crosssum_english-{iso}_train.json"])
    rows = []
    for i, ex in enumerate(exs):
        rows.append({"id": f"crosssum-{iso}-{i}", "document": ex.get("text", ""),
                     "summary": ex.get("summary", "")})
    return (repo, "default", rows)


def summ_indic_headline(iso):
    if iso not in INDIC_HEADLINE_LANGS:
        return None
    repo = "ai4bharat/IndicHeadlineGeneration"
    ds = load_dataset(repo, iso, split="test")
    rows = []
    for ex in ds:
        rows.append({"id": str(ex.get("id", "")), "document": ex.get("input", ""),
                     "summary": ex.get("target", "")})
    return (repo, iso, rows)


def summ_xlsum(iso):
    if iso not in XLSUM_LANG_NAMES:
        return None
    repo = "csebuetnlp/xlsum"
    cfg = XLSUM_LANG_NAMES[iso]
    ds = load_dataset(repo, cfg, split="test")  # tar-based, non-streaming
    rows = []
    for i, ex in enumerate(ds):
        rows.append({"id": ex.get("id", f"xlsum-{iso}-{i}"), "document": ex.get("text", ""),
                     "summary": ex.get("summary", "")})
        if len(rows) >= N_SAMPLES * 5:
            break
    return (repo, cfg, rows)


def summ_crosssum_hf(iso):
    if iso not in CROSSSUM_HF_LANG_NAMES:
        return None
    repo = "csebuetnlp/CrossSum"
    cfg = f"{CROSSSUM_HF_LANG_NAMES[iso]}-english"
    ds = load_dataset(repo, cfg, split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({"id": f"crosssumhf-{iso}-{i}", "document": ex.get("text", ""),
                     "summary": ex.get("summary", "")})
        if len(rows) >= N_SAMPLES * 5:
            break
    return (repo, cfg, rows)


def summ_indicwikibio(iso):
    if iso not in INDICWIKIBIO_LANGS:
        return None
    repo = "ai4bharat/IndicWikiBio"
    ds = load_dataset(repo, iso, split="test")
    rows = []
    for ex in ds:
        rows.append({"id": str(ex.get("id", "")), "document": ex.get("serialized_infobox", ""),
                     "summary": ex.get("summary", "")})
    return (repo, iso, rows)


SUMM_SOURCES = [summ_crosssum_in, summ_indic_headline, summ_xlsum, summ_crosssum_hf, summ_indicwikibio]


# ---------------------- Translation adapters ----------------------

# Our iso codes aren't all ISO-639-3 (a few are ISO-639-1), and flores_plus
# configs are keyed by "<iso_639_3>_<script>". Map explicitly rather than guess.
FLORES_PLUS_CONFIG = {
    "hi": "hin_Deva", "bn": "ben_Beng", "mr": "mar_Deva", "te": "tel_Telu", "ta": "tam_Taml",
    "gu": "guj_Gujr", "ur": "urd_Arab", "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "or": "ory_Orya", "as": "asm_Beng", "ne": "npi_Deva", "sd": "snd_Arab", "ks": "kas_Arab",
    "mai": "mai_Deva", "bho": "bho_Deva", "gom": "gom_Deva", "doi": "dgo_Deva", "sa": "san_Deva",
    "sat": "sat_Olck", "brx": "brx_Deva",
}


def translate_flores_plus(iso):
    repo = "openlanguagedata/flores_plus"
    cfg = FLORES_PLUS_CONFIG.get(iso)
    if cfg is None:
        return None
    for split in ("devtest", "dev"):
        try:
            tgt_ds = load_dataset(repo, cfg, split=split)
        except Exception:
            continue
        try:
            eng_ds = load_dataset(repo, "eng_Latn", split=split)
            eng_by_id = {ex["id"]: ex["text"] for ex in eng_ds}
        except Exception:
            eng_by_id = {}
        rows = []
        for ex in tgt_ds:
            rows.append({"id": f"floresplus-{iso}-{ex['id']}", "source_text": ex.get("text", ""),
                         "target_text": eng_by_id.get(ex["id"], ""), "source_lang": iso, "target_lang": "en"})
            if len(rows) >= N_SAMPLES:
                break
        if rows:
            return (repo, f"{cfg}/{split}", rows)
    return (repo, cfg, [])


def _resolve_flores_in_filenames(repo, iso):
    files = _repo_files(repo)
    prefixes = [f"flores_{iso}_en_", f"flores_en_{iso}_"]
    matches = []
    for p in prefixes:
        matches.extend(sorted(f for f in files if f.startswith(p)))
    return matches


def translate_flores_in(iso):
    if iso not in FLORES_IN_LANGS:
        return None
    repo = "google/IndicGenBench_flores_in"
    candidates = _resolve_flores_in_filenames(repo, iso)
    # prefer test over dev
    candidates = sorted(candidates, key=lambda f: 0 if "_test" in f else 1)
    exs = _igb_collect(repo, candidates)
    rows = []
    for i, ex in enumerate(exs):
        direction = ex.get("translation_direction", "")
        src, tgt = ex.get("source", ""), ex.get("target", "")
        if direction == "xxen":
            source_lang, target_lang = iso, "en"
        else:
            source_lang, target_lang = "en", iso
        rows.append({"id": f"floresin-{iso}-{i}", "source_text": src, "target_text": tgt,
                     "source_lang": source_lang, "target_lang": target_lang})
    return (repo, "default", rows)


def translate_bpcc(iso):
    repo = "ai4bharat/BPCC"
    ds = load_dataset(repo, iso, split="train", streaming=True)
    rows = []
    for ex in ds:
        rows.append({"id": f"bpcc-{iso}-{len(rows)}", "source_text": ex.get("src", ""),
                     "target_text": ex.get("tgt", ""), "source_lang": "en", "target_lang": iso})
        if len(rows) >= N_SAMPLES:
            break
    return (repo, iso, rows)


def translate_samanantar(iso):
    if iso not in SAMANANTAR_LANGS:
        return None
    repo = "ai4bharat/samanantar"
    ds = load_dataset(repo, iso, split="train", streaming=True)
    reservoir = []
    rng = random.Random(SEED)
    scan_cap = 20000
    for i, ex in enumerate(ds):
        item = {"id": f"samanantar-{iso}-{ex.get('idx', i)}", "source_text": ex.get("src", ""),
                "target_text": ex.get("tgt", ""), "source_lang": "en", "target_lang": iso}
        if len(reservoir) < N_SAMPLES:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < N_SAMPLES:
                reservoir[j] = item
        if i >= scan_cap:
            break
    return (repo, iso, reservoir)


def translate_itihasa(iso):
    if iso != "sa":
        return None
    repo = "rahular/itihasa"
    ds = load_dataset(repo, "Itihasa", split="test")
    rows = []
    for i, ex in enumerate(ds):
        tr = ex.get("translation", {})
        other_key = next((k for k in tr if k != "en"), None)
        if not other_key:
            continue
        rows.append({"id": f"itihasa-sa-{i}", "source_text": tr.get(other_key, ""),
                     "target_text": tr.get("en", ""), "source_lang": "sa", "target_lang": "en"})
    return (repo, "Itihasa", rows)


TRANSLATION_SOURCES = [translate_flores_plus, translate_flores_in, translate_bpcc,
                        translate_samanantar, translate_itihasa]


TASK_SOURCES = {
    "qa": QA_SOURCES,
    "reasoning": REASONING_SOURCES,
    "summarization": SUMM_SOURCES,
    "translation": TRANSLATION_SOURCES,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_known_gated = set()  # source_fn names already confirmed gated -- skip re-attempting


def collect_cell(task, iso):
    """Try each source in order; return (source_dataset, source_config, rows) for
    the best result found, or (None, None, []) with a reason if nothing worked."""
    best = (None, None, [])
    attempts_log = []
    for source_fn in TASK_SOURCES[task]:
        name = source_fn.__name__
        if name in _known_gated:
            attempts_log.append(f"{name}: GATED (requires HF auth, not collected; skipped repeat attempt)")
            continue
        try:
            result = source_fn(iso)
        except GatedRepoError:
            _known_gated.add(name)
            attempts_log.append(f"{name}: GATED (requires HF auth, not collected)")
            continue
        except HfHubHTTPError as e:
            attempts_log.append(f"{name}: HTTP error ({e})")
            continue
        except Exception as e:
            if "gated dataset" in str(e).lower() or "is a gated dataset" in str(e).lower():
                _known_gated.add(name)
                attempts_log.append(f"{name}: GATED (requires HF auth, not collected)")
            else:
                attempts_log.append(f"{name}: error {type(e).__name__}: {e}")
            continue
        if result is None:
            attempts_log.append(f"{name}: no config for this language")
            continue
        repo, cfg, rows = result
        attempts_log.append(f"{name} ({repo}/{cfg}): {len(rows)} rows fetched")
        if len(rows) > len(best[2]):
            best = (repo, cfg, rows)
        if len(rows) >= N_SAMPLES:
            break
    return best, attempts_log


def normalize_rows(task, iso, repo, cfg, raw_rows):
    raw_rows = prefer_native_script(raw_rows, task, iso)
    sampled = sample_rows(raw_rows)
    out = []
    for r in sampled:
        row = base_row(iso, task)
        row["source_dataset"] = repo or ""
        row["source_config"] = cfg or ""
        row.update({k: v for k, v in r.items() if k != "id"})
        row["id"] = r.get("id", "")
        out.append(row)
    return out


def write_csv(rows, columns, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False, encoding="utf-8")


def main():
    (RAW_DIR).mkdir(parents=True, exist_ok=True)
    coverage = {}  # (task, iso) -> count
    gap_reasons = {}
    all_task_rows = {t: [] for t in TASK_SOURCES}

    for task in ["qa", "reasoning", "summarization", "translation"]:
        slug = TASK_SLUG[task]
        cols = TASK_COLUMNS[task]
        for name, iso, tier in LANGUAGES:
            log(f"\n=== {TASK_LABELS[task]} / {name} ({iso}) ===")
            (repo, cfg, raw_rows), attempts_log = collect_cell(task, iso)
            for line in attempts_log:
                log(f"  - {line}")
            n = len(raw_rows)
            coverage[(task, iso)] = n
            out_path = RAW_DIR / slug / f"{iso}.csv"
            if n == 0:
                gap_reasons[(task, iso)] = "; ".join(attempts_log) if attempts_log else "no source configured"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_csv([], cols, out_path)
                log(f"  => GAP: 0 rows. ({gap_reasons[(task, iso)]})")
                continue
            norm_rows = normalize_rows(task, iso, repo, cfg, raw_rows)
            write_csv(norm_rows, cols, out_path)
            all_task_rows[task].extend(norm_rows)
            status = "OK" if n >= N_SAMPLES else f"SHORT ({n}/{N_SAMPLES})"
            log(f"  => {status}: wrote {len(norm_rows)} rows to {out_path.relative_to(ROOT)}")

        # per-task combined
        combined_path = DATA_DIR / f"{slug}.csv"
        write_csv(all_task_rows[task], cols, combined_path)
        log(f"\nWrote combined {task} CSV: {combined_path.relative_to(ROOT)} ({len(all_task_rows[task])} rows)")

    # grand combined CSV: superset of all columns
    grand_columns = ["id", "language", "iso_code", "resource_tier", "task",
                      "source_dataset", "source_config",
                      "question", "context", "answer",
                      "premise", "choice1", "choice2", "choice3", "choice4", "label", "prompt",
                      "document", "summary",
                      "source_text", "target_text", "source_lang", "target_lang"]
    grand_rows = []
    for task in all_task_rows:
        grand_rows.extend(all_task_rows[task])
    grand_path = DATA_DIR / "all_tasks_combined.csv"
    write_csv(grand_rows, grand_columns, grand_path)
    log(f"\nWrote grand combined CSV: {grand_path.relative_to(ROOT)} ({len(grand_rows)} rows)")

    # write run report
    report_path = DATA_DIR / "run_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
        f.write("\n\n=== Coverage summary ===\n")
        filled = 0
        for task in ["qa", "reasoning", "summarization", "translation"]:
            for name, iso, tier in LANGUAGES:
                n = coverage.get((task, iso), 0)
                status = "OK" if n >= N_SAMPLES else ("SHORT" if n > 0 else "GAP")
                if n >= N_SAMPLES:
                    filled += 1
                f.write(f"{TASK_LABELS[task]:25s} {name:12s} ({iso:4s}): {n:4d} rows  [{status}]\n")
        f.write(f"\nCells with >=100 rows: {filled}/{len(LANGUAGES) * 4}\n")
    log(f"\nWrote run report: {report_path.relative_to(ROOT)}")
    log(f"\nFINAL: {filled}/{len(LANGUAGES) * 4} language x task cells reached >= {N_SAMPLES} rows")

    return coverage, gap_reasons


if __name__ == "__main__":
    main()
