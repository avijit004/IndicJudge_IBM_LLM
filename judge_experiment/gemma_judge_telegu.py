import csv
import json
import os
import re
import time
import requests


# ============================================================
# Configuration
# ============================================================

GEMMA_URL = "http://172.30.1.83:8080/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILES = [
    os.path.join(
        BASE_DIR,
        "mgsm_telegu_experiment",
        "gemma4_26b_telugu_responses.csv"
    ),
    os.path.join(
        BASE_DIR,
        "mgsm_telegu_experiment",
        "gemma4_31b_telugu_responses.csv"
    ),
]

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma_judge_telegu_results.csv")

JUDGE_MODEL = "Gemma-4-31B"

TEMPERATURE = 0.0
MAX_TOKENS = 200

REQUEST_TIMEOUT = 600
MAX_RETRIES = 3


# ============================================================
# Judge Prompt
# ============================================================

JUDGE_PROMPT = """
You are an expert evaluator of mathematical solutions.

You will be given a mathematical question and an AI-generated solution.

Your task is to evaluate the generated solution strictly for mathematical
quality.

IMPORTANT:
Do NOT assume that the generated solution is correct.

First solve the mathematical problem yourself. Carefully determine the
correct result and reasoning. Then compare the generated solution against
your own solution.

Evaluate the generated solution using all of the following criteria:

1. Logical correctness
   Does the reasoning follow logically from the information given in the
   question?

2. Mathematical correctness
   Are the mathematical operations, formulas, concepts, and assumptions
   correct?

3. Calculation accuracy
   Check every important arithmetic operation and intermediate calculation.
   Identify numerical errors or miscalculations.

4. Reasoning validity
   Does the reasoning actually support the conclusion?
   Do not give credit merely because the final answer happens to be correct.

5. Final answer
   Is the final answer correct and consistent with the reasoning?

Ignore differences in wording, style, verbosity, or formatting unless they
affect the mathematical meaning or correctness.

Scoring:

1.0 = completely correct, logically sound, and mathematically accurate

0.8-0.99 = essentially correct with only a very minor issue

0.5-0.79 = partially correct but contains a meaningful error or omission

0.1-0.49 = mostly incorrect

0.0 = completely incorrect

The "correct" field should be true only when the solution is mathematically
correct overall. If there is a meaningful mathematical or logical error,
set it to false.

Return ONLY a JSON object in this format:

{{
  "score": 0.0,
  "correct": true,
  "reason": "brief explanation"
}}

Do not include markdown.
Do not include ```json.
Do not include any additional text.

QUESTION:
{question}

GENERATED SOLUTION:
{answer}
"""


# ============================================================
# JSON extraction
# ============================================================

def extract_json(text):
    """
    Extract the judge's JSON object.

    Handles:
        {"score": 1.0, ...}

    and also:
        ```json
        {"score": 1.0, ...}
        ```
    """

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # First attempt: entire response is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: find JSON object inside response
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# Validate judge result
# ============================================================

def validate_result(result):
    """
    Validate and normalize the JSON returned by Gemma.
    """

    if not isinstance(result, dict):
        return None

    if "score" not in result:
        return None

    if "correct" not in result:
        return None

    if "reason" not in result:
        return None

    # Score
    try:
        score = float(result["score"])
    except (ValueError, TypeError):
        return None

    score = max(0.0, min(1.0, score))

    # Correct
    correct = result["correct"]

    if isinstance(correct, str):
        correct_lower = correct.strip().lower()

        if correct_lower == "true":
            correct = True
        elif correct_lower == "false":
            correct = False
        else:
            return None

    elif isinstance(correct, bool):
        pass

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
# Call Gemma judge
# ============================================================

def judge_with_gemma(question, answer):

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

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.post(
                GEMMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            content = data["choices"][0]["message"]["content"]

            result = extract_json(content)

            result = validate_result(result)

            if result is not None:

                return (
                    result["score"],
                    result["correct"],
                    result["reason"],
                    content
                )

            print()
            print("WARNING: Gemma returned an invalid JSON judgment.")
            print("Raw Gemma response:")
            print(content)
            print()

        except Exception as e:

            print()
            print(
                f"Gemma request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            print(f"Error: {e}")
            print()

        if attempt < MAX_RETRIES:

            print("Retrying in 5 seconds...")
            time.sleep(5)

    return (
        None,
        None,
        "Gemma judge failed after all retry attempts.",
        ""
    )


# ============================================================
# Load completed judgments
# ============================================================

def load_completed():

    completed = set()

    if not os.path.exists(OUTPUT_FILE):
        return completed

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                return completed

            required_columns = {
                "source_model",
                "question_id",
                "prompt_id",
                "judge_model"
            }

            if not required_columns.issubset(
                set(reader.fieldnames)
            ):
                print(
                    "WARNING: Existing output file has an "
                    "unexpected format."
                )
                print(
                    "Starting without using existing checkpoints."
                )
                return completed

            for row in reader:

                source_model = (
                    row.get("source_model") or ""
                ).strip()

                question_id = (
                    row.get("question_id") or ""
                ).strip()

                prompt_id = (
                    row.get("prompt_id") or ""
                ).strip()

                judge_model = (
                    row.get("judge_model") or ""
                ).strip()

                # Only count successfully completed judgments
                score = row.get("score")

                if (
                    source_model
                    and question_id
                    and prompt_id
                    and judge_model
                    and score not in (None, "")
                ):

                    completed.add(
                        (
                            source_model,
                            question_id,
                            prompt_id,
                            judge_model
                        )
                    )

    except Exception as e:

        print(
            f"WARNING: Could not read existing output file: {e}"
        )

        print(
            "Starting without existing checkpoints."
        )

        completed = set()

    return completed


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    completed = load_completed()

    print("=" * 70)
    print("Gemma 4 LLM-as-a-Judge")
    print("=" * 70)

    print()
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Gemma URL   : {GEMMA_URL}")
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Completed   : {len(completed)}")
    print()

    output_fields = [
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
        "raw_judge_response"
    ]

    file_exists = os.path.exists(OUTPUT_FILE)

    # If the existing file exists but is empty/invalid,
    # start a new file.
    if file_exists:

        if os.path.getsize(OUTPUT_FILE) == 0:

            file_exists = False

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=output_fields
        )

        if not file_exists:

            writer.writeheader()
            outfile.flush()

        total_new = 0
        total_skipped = 0

        for input_file in INPUT_FILES:

            # Determine which generator produced the responses
            if "26b" in input_file.lower():

                source_model = "Gemma-4-26B"

            elif "31b" in input_file.lower():

                source_model = "Gemma-4-31B"

            else:

                source_model = "Unknown"

            print()
            print("-" * 70)
            print(f"Processing: {source_model}")
            print(f"Input: {input_file}")
            print("-" * 70)

            # ------------------------------------------------
            # Read input CSV
            # ------------------------------------------------

            with open(
                input_file,
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as infile:

                reader = csv.DictReader(infile)

                if reader.fieldnames is None:

                    print(
                        f"ERROR: No header found in {input_file}"
                    )
                    continue

                # Clean header names
                reader.fieldnames = [
                    str(field).strip().lstrip("\ufeff")
                    for field in reader.fieldnames
                ]

                required_columns = {
                    "question_id",
                    "prompt_id",
                    "question",
                    "prompt",
                    "response"
                }

                missing = (
                    required_columns
                    - set(reader.fieldnames)
                )

                if missing:

                    print(
                        f"ERROR: Missing columns: {missing}"
                    )

                    print(
                        f"Available columns: "
                        f"{reader.fieldnames}"
                    )

                    continue

                rows = list(reader)

            print(f"Responses found: {len(rows)}")

            # ------------------------------------------------
            # Judge each response
            # ------------------------------------------------

            for row_number, row in enumerate(
                rows,
                start=1
            ):

                # Clean accidental whitespace/BOM
                row = {
                    str(k).strip().lstrip("\ufeff"): v
                    for k, v in row.items()
                }

                question_id = (
                    row.get("question_id") or ""
                ).strip()

                prompt_id = (
                    row.get("prompt_id") or ""
                ).strip()

                question = row.get("question") or ""

                prompt = row.get("prompt") or ""

                answer = row.get("response") or ""

                # Validate identifiers
                if not question_id:

                    print(
                        f"WARNING: Row {row_number} has "
                        f"no question_id. Skipping."
                    )

                    continue

                if not prompt_id:

                    print(
                        f"WARNING: Row {row_number} has "
                        f"no prompt_id. Skipping."
                    )

                    continue

                # ------------------------------------------------
                # Resume checkpoint
                # ------------------------------------------------

                key = (
                    source_model,
                    question_id,
                    prompt_id,
                    JUDGE_MODEL
                )

                if key in completed:

                    total_skipped += 1
                    continue

                # ------------------------------------------------
                # Print progress
                # ------------------------------------------------

                total_new += 1

                print()
                print(
                    f"[{total_new}] "
                    f"Judging {source_model} "
                    f"| Question {question_id} "
                    f"| Prompt {prompt_id}"
                )

                # ------------------------------------------------
                # Send to Gemma
                # ------------------------------------------------

                score, correct, reason, raw = (
                    judge_with_gemma(
                        question,
                        answer
                    )
                )

                # ------------------------------------------------
                # Save result immediately
                # ------------------------------------------------

                writer.writerow({

                    "source_model":
                        source_model,

                    "question_id":
                        question_id,

                    "prompt_id":
                        prompt_id,

                    "question":
                        question,

                    "prompt":
                        prompt,

                    "reasoning_content":
                        row.get(
                            "reasoning_content",
                            ""
                        ),

                    "response":
                        answer,

                    "judge_model":
                        JUDGE_MODEL,

                    "score":
                        score,

                    "correct":
                        correct,

                    "reason":
                        reason,

                    "raw_judge_response":
                        raw
                })

                # IMPORTANT:
                # Flush after every judgment so that if the
                # process stops, completed results are preserved.
                outfile.flush()

                # Only mark as completed if judgment succeeded
                if score is not None:

                    completed.add(key)

                print(
                    f"Score   : {score}"
                )

                print(
                    f"Correct : {correct}"
                )

                print(
                    f"Reason  : {reason}"
                )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("Gemma judging finished")
    print("=" * 70)

    print(
        f"New judgments   : {total_new}"
    )

    print(
        f"Already skipped : {total_skipped}"
    )

    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
