"""
Sanity-checks every dataset_csv/<task>/<iso>.csv: does the target-language
column actually contain text in the expected script, and (where a language-ID
model supports the language) the expected language specifically -- not just
"some Devanagari text" that could secretly be a different Devanagari language.

Two independent checks per file:
  1. Unicode script range: catches gross errors (wrong script entirely, or
     English left in a column that should be translated).
  2. langdetect (supports hi, bn, mr, ta, te, gu, ur, kn, ml, pa, ne): catches
     cross-contamination between languages that SHARE a script, e.g. Hindi
     rows accidentally ending up in the Marathi file. Not available for the
     11 languages langdetect doesn't model (or, as, sd, ks, mai, bho, gom,
     doi, sa, sat, brx) -- those get script-check only, flagged as such.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 42  # deterministic

ROOT = Path(__file__).resolve().parent.parent
DATASET_CSV = ROOT / "dataset_csv"

# Unicode block per language's expected script.
SCRIPT_RANGES = {
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

# langdetect ISO codes it can actually distinguish among our set.
LANGDETECT_SUPPORTED = {"hi", "bn", "mr", "ta", "te", "gu", "ur", "kn", "ml", "pa", "ne"}

# Column to check per task (the column that should be IN the target language).
TARGET_COL = {
    "qa": "question",
    "reasoning": None,  # handled specially: premise, else prompt
    "summarization": "summary",
    "translation": "source_text",
}

LANG_NAME = {
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "te": "Telugu", "ta": "Tamil",
    "gu": "Gujarati", "ur": "Urdu", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "or": "Odia", "as": "Assamese", "ne": "Nepali", "sd": "Sindhi", "ks": "Kashmiri",
    "mai": "Maithili", "bho": "Bhojpuri", "gom": "Konkani", "doi": "Dogri", "sa": "Sanskrit",
    "sat": "Santali", "brx": "Bodo",
}


def script_ratio(text: str, iso: str) -> float:
    ranges = SCRIPT_RANGES.get(iso)
    if not ranges:
        return float("nan")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return float("nan")
    in_script = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return in_script / len(letters)


def sample_texts(df: pd.DataFrame, task: str) -> list[str]:
    if task == "reasoning":
        col = "premise"
        texts = df[col].fillna("").tolist()
        prompt_texts = df.get("prompt", pd.Series(dtype=str)).fillna("").tolist()
        texts = [t if t.strip() else p for t, p in zip(texts, prompt_texts)]
    else:
        col = TARGET_COL[task]
        texts = df[col].fillna("").tolist()
    return [t for t in texts if isinstance(t, str) and t.strip()]


def main() -> None:
    rows = []
    for task in ["qa", "reasoning", "summarization", "translation"]:
        task_dir = DATASET_CSV / task
        if not task_dir.exists():
            continue
        for csv_path in sorted(task_dir.glob("*.csv")):
            iso = csv_path.stem
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            texts = sample_texts(df, task)
            n = len(texts)
            if n == 0:
                rows.append({"task": task, "iso": iso, "lang": LANG_NAME.get(iso, iso),
                             "n_checked": 0, "script_match_pct": None, "langid_match_pct": None,
                             "langid_supported": iso in LANGDETECT_SUPPORTED, "flag": "NO TEXT FOUND"})
                continue

            # Unicode script check over all rows
            ratios = [script_ratio(t, iso) for t in texts]
            ratios = [r for r in ratios if r == r]  # drop NaN
            script_pct = round(100 * sum(1 for r in ratios if r >= 0.6) / len(ratios), 1) if ratios else None

            # langdetect check over a sample (it's slow-ish; sample up to 40 rows)
            langid_pct = None
            if iso in LANGDETECT_SUPPORTED:
                sample = texts[:40]
                hits = 0
                checked = 0
                for t in sample:
                    if len(t.strip()) < 8:
                        continue
                    checked += 1
                    try:
                        guesses = detect_langs(t)
                        top = guesses[0].lang if guesses else None
                    except Exception:
                        top = None
                    if top == iso:
                        hits += 1
                if checked:
                    langid_pct = round(100 * hits / checked, 1)

            flag = ""
            if script_pct is not None and script_pct < 90:
                flag += f"SCRIPT MISMATCH ({script_pct}% in-script); "
            if langid_pct is not None and langid_pct < 50:
                flag += f"LANGID MISMATCH ({langid_pct}% detected as {iso}); "

            rows.append({"task": task, "iso": iso, "lang": LANG_NAME.get(iso, iso),
                         "n_checked": len(texts), "script_match_pct": script_pct,
                         "langid_match_pct": langid_pct, "langid_supported": iso in LANGDETECT_SUPPORTED,
                         "flag": flag.strip() or "OK"})

    report = pd.DataFrame(rows)
    out_path = ROOT / "language_verification_report.csv"
    report.to_csv(out_path, index=False)

    print(report.to_string(index=False))
    n_flagged = (report["flag"] != "OK").sum() - (report["flag"] == "NO TEXT FOUND").sum()
    print(f"\n{len(report)} files checked, {n_flagged} with a script/langid mismatch flag, "
          f"{(report['flag'] == 'NO TEXT FOUND').sum()} with no text.")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
