"""
Generate candidate responses under the four perturbation prompts (PT1-PT4).

Mirrors the original generate_<language>_responses_<size>.py scripts (same
request payload, same CSV columns, resume support) but works for every language
and both Gemma sizes from one file. Reads the original questions.json files
read-only and writes only into perturbation_experiment/responses/.

Examples (run from anywhere):
    export GEMMA_31B_API_URL="http://HOST:PORT/v1/chat/completions"
    python perturbation_experiment/generate_perturbation_responses.py --language hindi --model 31b
    python perturbation_experiment/generate_perturbation_responses.py --language all --model 26b --question-ids 1 2 3
    python perturbation_experiment/generate_perturbation_responses.py --language hindi --model 31b --dry-run
"""
import argparse
import csv
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perturbation_prompts import LANGUAGE_DIRS, PERTURBATION_PROMPTS, SOURCE_MODELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.environ.get("JUDGEBENCH_DATA_DIR", os.path.join(HERE, "..")))
RESPONSES_DIR = os.path.join(HERE, "responses")

TEMPERATURE = 0.8      # same as the original generation scripts
MAX_TOKENS = 8192      # same as the original generation scripts
REQUEST_TIMEOUT = 600
MAX_ATTEMPTS = 3

FIELDNAMES = ["question_id", "prompt_id", "question", "prompt", "reasoning_content", "response"]


def output_path(language, model):
    return os.path.join(RESPONSES_DIR, f"gemma4_{model}_{language}_perturbation_responses.csv")


def load_questions(language):
    path = os.path.join(PROJECT_ROOT, LANGUAGE_DIRS[language], "questions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_completed(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("response") or "").strip():
                    done.add((str(row["question_id"]), row["prompt_id"]))
    return done


def request_response(api_url, full_prompt):
    payload = {
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    r = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    message = r.json()["choices"][0]["message"]
    return (message.get("reasoning_content") or ""), (message.get("content") or "")


def run_language(language, model, api_url, question_ids, dry_run):
    questions = load_questions(language)
    if question_ids:
        wanted = {str(q) for q in question_ids}
        questions = [q for q in questions if str(q["question_id"]) in wanted]
    path = output_path(language, model)
    completed = load_completed(path)
    total = len(questions) * len(PERTURBATION_PROMPTS)
    todo = total - len(completed & {(str(q["question_id"]), pid) for q in questions for pid, _ in PERTURBATION_PROMPTS})
    print("=" * 70)
    print(f"{language.title()} | Gemma-4-{model.upper()} | {len(questions)} questions x "
          f"{len(PERTURBATION_PROMPTS)} prompts = {total} responses ({todo} remaining)")
    print(f"Output: {path}")
    if dry_run:
        return 0
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    failed = 0
    count = 0
    with open(path, "a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if needs_header:
            writer.writeheader()
            fh.flush()
        try:
            for q in questions:
                qid = str(q["question_id"])
                for pid, ptext in PERTURBATION_PROMPTS:
                    if (qid, pid) in completed:
                        continue
                    count += 1
                    print(f"[{count}/{todo}] Question {qid} | Prompt {pid}")
                    full_prompt = ptext + "\n\n" + q["question"]
                    for attempt in range(1, MAX_ATTEMPTS + 1):
                        try:
                            reasoning, answer = request_response(api_url, full_prompt)
                            if not answer.strip():
                                raise ValueError("empty response content")
                            break
                        except Exception as exc:  # noqa: BLE001
                            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
                            reasoning = answer = None
                            time.sleep(2 * attempt)
                    if answer is None:
                        failed += 1
                        print("  Skipped; rerun the script to retry.")
                        continue
                    writer.writerow({"question_id": qid, "prompt_id": pid, "question": q["question"],
                                     "prompt": ptext, "reasoning_content": reasoning, "response": answer})
                    fh.flush()
                    completed.add((qid, pid))
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print("Interrupted; completed responses are saved.")
            return 1
    print(f"Done with {language}: {failed} failed.")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Generate PT1-PT4 perturbation responses with Gemma.")
    parser.add_argument("--language", nargs="+", required=True,
                        help=f"One or more of {', '.join(LANGUAGE_DIRS)} or 'all'.")
    parser.add_argument("--model", choices=tuple(SOURCE_MODELS), required=True)
    parser.add_argument("--api-url", default=None,
                        help="Chat completions endpoint (default: $GEMMA_26B_API_URL or $GEMMA_31B_API_URL).")
    parser.add_argument("--question-ids", nargs="+", default=None, metavar="ID")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only; no requests, no files written.")
    args = parser.parse_args()

    languages = list(LANGUAGE_DIRS) if "all" in args.language else args.language
    unknown = [l for l in languages if l not in LANGUAGE_DIRS]
    if unknown:
        parser.error(f"Unknown language(s): {unknown}")
    api_url = args.api_url or os.environ.get(f"GEMMA_{args.model.upper()}_API_URL", "").strip()
    if not api_url and not args.dry_run:
        parser.error(f"Set --api-url or GEMMA_{args.model.upper()}_API_URL.")

    status = 0
    for language in languages:
        status |= run_language(language, args.model, api_url, args.question_ids, args.dry_run)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
