"""
Flattens data/raw/<task>/<iso>.csv (which carries full provenance columns) into
dataset_csv/<task>/<iso>.csv with a lean, task-appropriate schema — one folder
per task, one CSV per language, matching the layout the user wants on disk.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "dataset_csv"

# (source folder under data/raw, destination folder under dataset_csv, columns to keep + rename)
TASKS = {
    "question_answering": "qa",
    "reasoning": "reasoning",
    "summarization": "summarization",
    "translation_quality": "translation",
}


def flatten_qa(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id", "language", "context", "question", "answer"]].copy()
    out["answers"] = out["answer"].apply(lambda a: json.dumps([a], ensure_ascii=False))
    return out[["id", "language", "context", "question", "answers"]]


def flatten_reasoning(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["id", "language", "premise", "question", "choice1", "choice2", "choice3", "choice4",
            "label", "prompt", "answer"]
    out = df[cols].copy()
    return out


def flatten_summarization(df: pd.DataFrame) -> pd.DataFrame:
    return df[["id", "language", "document", "summary"]].copy()


def flatten_translation(df: pd.DataFrame) -> pd.DataFrame:
    return df[["id", "language", "source_lang", "target_lang", "source_text", "target_text"]].copy()


FLATTENERS = {
    "question_answering": flatten_qa,
    "reasoning": flatten_reasoning,
    "summarization": flatten_summarization,
    "translation_quality": flatten_translation,
}


def main() -> None:
    total_files = 0
    total_rows = 0
    for src_task, dst_task in TASKS.items():
        src_dir = RAW / src_task
        dst_dir = OUT / dst_task
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src_dir.exists():
            print(f"!! missing source dir: {src_dir}")
            continue
        for csv_path in sorted(src_dir.glob("*.csv")):
            iso = csv_path.stem
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            flat = FLATTENERS[src_task](df)
            dst_path = dst_dir / f"{iso}.csv"
            flat.to_csv(dst_path, index=False, encoding="utf-8-sig")
            total_files += 1
            total_rows += len(flat)
            print(f"{dst_task}/{iso}.csv  ({len(flat)} rows)")

    print(f"\nWrote {total_files} files, {total_rows} rows, under {OUT}")


if __name__ == "__main__":
    main()
