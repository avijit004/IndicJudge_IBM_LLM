import csv
import json
import os
import re
import time
import requests


# ============================================================
# Configuration
# ============================================================

QWEN_URL = "http://172.30.1.83:8080/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILES = [
    os.path.join(
        BASE_DIR,
        "mgsm_bengali_experiment",
        "gemma4_26b_bengali_responses.csv"
    ),
    os.path.join(
        BASE_DIR,
        "mgsm_bengali_experiment",
        "gemma4_31b_bengali_responses.csv"
    ),
]

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_bengali_judge_results.csv")

JUDGE_MODEL = "Qwen3.5-27B"

TEMPERATURE = 0.0

# Includes Qwen reasoning + final JSON answer
MAX_TOKENS = 4096

REQUEST_TIMEOUT = 600
MAX_RETRIES = 3
REQUEST_DELAY = 0.2


# ============================================================
# Judge Prompt
# ============================================================

JUDGE_PROMPT = """
You are an expert evaluator of mathematical solutions.

You will be given:

1. A mathematical question.
2. An AI-generated candidate solution to that question.

Your task is to evaluate whether the candidate solution is mathematically
correct.

IMPORTANT:
- Solve the mathematical problem independently before judging the candidate.
- Check every important calculation.
- Check the logical reasoning and intermediate steps.
- Check whether the final answer follows from the calculations.
- A candidate is correct only if its reasoning and final answer are
  mathematically valid.
- Do not give credit merely because the final numerical answer happens to
  be correct if the reasoning contains a substantive mathematical error.
- Minor wording, formatting, or stylistic issues should NOT make an otherwise
  mathematically correct solution incorrect.
- The candidate may be written in Telugu or another language. Evaluate the
  mathematics, not the language quality.

Return ONLY a valid JSON object with exactly these three fields:

{{
  "score": 1.0,
  "correct": true,
  "reason": "Brief explanation of why the candidate solution is correct or incorrect."
}}

Scoring:
- 1.0 = mathematically correct
- 0.0 = mathematically incorrect

The "correct" field must be true when score is 1.0 and false when score is 0.0.

The "reason" should briefly identify the key mathematical evidence supporting
the judgment.

MATHEMATICAL QUESTION:
<<<
{question}
>>>

CANDIDATE SOLUTION:
<<<
{answer}
>>>
"""


# ============================================================
# Extract JSON
# ============================================================

def extract_json(text):
    if not text:
        return None

    text = text.strip()

    # Markdown JSON block
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Entire response is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object inside surrounding text
    start = text.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):

        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start:i + 1]

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None


# ============================================================
# Validate Qwen judgment
# ============================================================

def validate_judge_result(result):

    if not isinstance(result, dict):
        return None

    required = ["score", "correct", "reason"]

    for field in required:
        if field not in result:
            return None

    # Score
    try:
        score = float(result["score"])
    except (TypeError, ValueError):
        return None

    if score < 0:
        score = 0.0

    if score > 1:
        score = 1.0

    # Correct
    correct_value = result["correct"]

    if isinstance(correct_value, bool):
        correct = correct_value

    elif isinstance(correct_value, str):

        value = correct_value.strip().lower()

        if value == "true":
            correct = True

        elif value == "false":
            correct = False

        else:
            return None

    else:
        return None

    # Reason
    reason = str(result["reason"]).strip()

    return {
        "score": score,
        "correct": correct,
        "reason": reason
    }


# ============================================================
# Call Qwen
# ============================================================

def judge_with_qwen(question, answer):

    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    last_error = None
    last_raw_response = ""
    last_reasoning = ""

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.post(
                QWEN_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            message = data["choices"][0]["message"]

            content = message.get("content", "")

            reasoning_content = message.get(
                "reasoning_content",
                ""
            )

            last_raw_response = content
            last_reasoning = reasoning_content

            result = extract_json(content)

            result = validate_judge_result(result)

            if result is not None:

                return result, content, reasoning_content

            last_error = (
                "Invalid JSON returned by Qwen.\n"
                f"Response:\n{content}"
            )

            print(
                f"  Attempt {attempt}/{MAX_RETRIES}: "
                "invalid JSON"
            )

        except Exception as e:

            last_error = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"  Attempt {attempt}/{MAX_RETRIES}: "
                f"{last_error}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(2)

    return (
        {
            "score": "",
            "correct": "",
            "reason": f"JUDGE_ERROR: {last_error}"
        },
        last_raw_response,
        last_reasoning
    )


# ============================================================
# Load input CSVs
# ============================================================

def load_input_rows():

    rows = []

    for input_file in INPUT_FILES:

        print(f"Loading: {input_file}")

        if not os.path.exists(input_file):

            raise FileNotFoundError(
                f"Input file not found:\n{input_file}"
            )

        # IMPORTANT:
        # utf-8-sig removes the BOM from question_id
        with open(
            input_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            file_rows = list(reader)

        filename = os.path.basename(input_file)

        if "26b" in filename.lower():
            source_model = "Gemma-4-26B"

        elif "31b" in filename.lower():
            source_model = "Gemma-4-31B"

        else:
            source_model = os.path.splitext(filename)[0]

        for row in file_rows:

            row["_source_model"] = source_model

            rows.append(row)

        print(
            f"  Loaded {len(file_rows)} rows "
            f"from {filename}"
        )

    return rows


# ============================================================
# Load completed judgments
# ============================================================

def load_completed_keys():

    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    # IMPORTANT:
    # utf-8-sig also protects against BOM in output CSV
    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            score = row.get("score", "").strip()
            correct = row.get("correct", "").strip()

            if score != "" and correct != "":

                key = (
                    row.get("source_model", ""),
                    row.get("question_id", ""),
                    row.get("prompt_id", ""),
                    row.get("judge_model", "")
                )

                completed.add(key)

    return completed


# ============================================================
# Output columns
# ============================================================

FIELDNAMES = [
    "source_model",
    "question_id",
    "prompt_id",
    "question",
    "prompt",
    "reasoning_content",
    "response",
    "judge_model",
    "score",
    "correct",
    "reason",
    "raw_judge_response",
    "qwen_reasoning"
]


# ============================================================
# Append one result
# ============================================================

def append_result(row):

    file_exists = os.path.exists(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        if (
            not file_exists
            or os.path.getsize(OUTPUT_FILE) == 0
        ):
            writer.writeheader()

        writer.writerow(row)

        f.flush()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Qwen3.5-27B Mathematical Judge")
    print("=" * 70)

    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Qwen URL    : {QWEN_URL}")
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Max tokens  : {MAX_TOKENS}")
    print()

    # --------------------------------------------------------
    # Check server
    # --------------------------------------------------------

    try:

        response = requests.get(
            "http://127.0.0.1:8080/v1/models",
            timeout=10
        )

        response.raise_for_status()

        print("✓ Qwen server is reachable.")

    except Exception as e:

        print("ERROR: Qwen server is not reachable.")
        print(e)
        return

    print()

    # --------------------------------------------------------
    # Load responses
    # --------------------------------------------------------

    rows = load_input_rows()

    print()
    print(
        f"Total candidate responses: {len(rows)}"
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    completed = load_completed_keys()

    print(
        f"Already completed: {len(completed)}"
    )

    remaining = 0

    for row in rows:

        key = (
            row["_source_model"],
            row["question_id"],
            row["prompt_id"],
            JUDGE_MODEL
        )

        if key not in completed:
            remaining += 1

    print(
        f"Remaining judgments: {remaining}"
    )

    print("=" * 70)
    print()

    if remaining == 0:

        print("All judgments are already complete.")
        print(
            f"Results saved to: {OUTPUT_FILE}"
        )

        return

    # --------------------------------------------------------
    # Judge
    # --------------------------------------------------------

    processed = 0

    for row in rows:

        source_model = row["_source_model"]

        question_id = row["question_id"]
        prompt_id = row["prompt_id"]

        key = (
            source_model,
            question_id,
            prompt_id,
            JUDGE_MODEL
        )

        if key in completed:
            continue

        processed += 1

        print("-" * 70)

        print(
            f"[{processed}/{remaining}] "
            f"{source_model} | "
            f"Question {question_id} | "
            f"Prompt {prompt_id}"
        )

        print(
            "Judging with Qwen3.5-27B..."
        )

        result, raw_response, qwen_reasoning = (
            judge_with_qwen(
                row["question"],
                row["response"]
            )
        )

        # ----------------------------------------------------
        # Prepare result
        # ----------------------------------------------------

        output_row = {
            "source_model": source_model,
            "question_id": question_id,
            "prompt_id": prompt_id,
            "question": row["question"],
            "prompt": row["prompt"],
            "reasoning_content": row.get(
                "reasoning_content",
                ""
            ),
            "response": row["response"],
            "judge_model": JUDGE_MODEL,
            "score": result.get("score", ""),
            "correct": result.get("correct", ""),
            "reason": result.get("reason", ""),
            "raw_judge_response": raw_response,
            "qwen_reasoning": qwen_reasoning
        }

        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        append_result(output_row)

        # ----------------------------------------------------
        # Checkpoint successful result
        # ----------------------------------------------------

        if (
            str(result.get("score", "")).strip() != ""
            and str(result.get("correct", "")).strip() != ""
        ):

            completed.add(key)

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Score   : {result.get('score', '')}"
        )

        print(
            f"Correct : {result.get('correct', '')}"
        )

        print(
            f"Reason  : {result.get('reason', '')}"
        )

        print()

        time.sleep(REQUEST_DELAY)

    print("=" * 70)
    print("Qwen judging completed.")
    print()
    print("Results saved to:")
    print(OUTPUT_FILE)
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
