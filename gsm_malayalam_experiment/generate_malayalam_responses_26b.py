import json
import csv
import os
import requests
import time


# ============================================================
# Configuration
# ============================================================

API_URL = "http://172.30.1.83:8080/v1/chat/completions"

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_26b_malayalam_responses.csv")

TEMPERATURE = 0.8
MAX_TOKENS = 8192


# ============================================================
# Five prompt variations
# ============================================================

PROMPTS = [
    (
        "P1",
        "Solve the following mathematical problem step by step. "
        "Use logically correct reasoning and calculations throughout. "
        "Verify the result and provide the correct final answer."
    ),

    (
        "P2",
        "Solve the following mathematical problem. "
        "Provide the correct final answer, but construct the solution "
        "with a subtle logical or mathematical error in the reasoning "
        "while still arriving at the correct final answer."
    ),

    (
        "P3",
        "Solve the following mathematical problem step by step. "
        "Use a correct approach and correct reasoning throughout, "
        "but introduce a small calculation or arithmetic error at the end "
        "so that the final answer is incorrect."
    ),

    (
        "P4",
        "Attempt to solve the following mathematical problem. "
        "Use incorrect reasoning or invalid mathematical steps "
        "and arrive at an incorrect final answer."
    ),

    (
        "P5",
        "Solve the following mathematical problem and give the correct final answer. "
        "Provide only a brief explanation that does not contain enough reasoning "
        "or intermediate steps to fully justify the answer."
    ),
]


# ============================================================
# Load questions
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)

print(f"Loaded {len(questions)} Malayalam questions.")
print(
    f"Generating {len(questions) * len(PROMPTS)} responses."
)
print()


# ============================================================
# CSV configuration
# ============================================================

fieldnames = [
    "question_id",
    "prompt_id",
    "question",
    "prompt",
    "reasoning_content",
    "response",
]


# ============================================================
# Resume support
# ============================================================

completed = set()

if os.path.exists(OUTPUT_FILE):

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            completed.add(
                (
                    str(row["question_id"]),
                    row["prompt_id"],
                )
            )

    print(
        f"Found {len(completed)} existing responses."
    )

    print(
        "Already completed responses will be skipped."
    )

    print()


# ============================================================
# Open output CSV
# ============================================================

file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(
    OUTPUT_FILE,
    "a",
    encoding="utf-8-sig",
    newline=""
)

writer = csv.DictWriter(
    csv_file,
    fieldnames=fieldnames
)

if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:

    writer.writeheader()
    csv_file.flush()


# ============================================================
# Generation
# ============================================================

total = len(questions) * len(PROMPTS)
count = len(completed)


try:

    for question in questions:

        question_id = str(
            question["question_id"]
        )

        question_text = question["question"]

        for prompt_id, prompt_text in PROMPTS:

            key = (
                question_id,
                prompt_id
            )

            if key in completed:
                continue

            count += 1

            print("=" * 70)

            print(
                f"[{count}/{total}] "
                f"Question {question_id} | "
                f"Prompt {prompt_id}"
            )

            print()

            print(
                "Generating response..."
            )

            # ------------------------------------------------
            # Construct prompt
            # ------------------------------------------------

            full_prompt = (
                prompt_text
                + "\n\n"
                + question_text
                + "\n\n(IMPORTANT: You MUST write your entire reasoning and final answer in Malayalam.)"
            )

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS
            }

            # ------------------------------------------------
            # Request
            # ------------------------------------------------

            response = requests.post(
                API_URL,
                json=payload,
                timeout=600
            )

            response.raise_for_status()

            data = response.json()

            message = data["choices"][0]["message"]

            reasoning_content = message.get(
                "reasoning_content",
                ""
            )

            answer = message.get(
                "content",
                ""
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            writer.writerow({
                "question_id": question_id,
                "prompt_id": prompt_id,
                "question": question_text,
                "prompt": prompt_text,
                "reasoning_content": reasoning_content,
                "response": answer,
            })

            csv_file.flush()

            completed.add(key)

            print()
            print("Response generated successfully.")
            print()

            print("Final response:")
            print(answer)
            print()

            time.sleep(0.2)


except KeyboardInterrupt:

    print()
    print("Generation interrupted by user.")
    print("Completed responses have already been saved.")


except Exception as e:

    print()
    print("ERROR:")
    print(e)
    print()
    print("Completed responses have already been saved.")


finally:

    csv_file.close()


# ============================================================
# Finished
# ============================================================

print("=" * 70)
print("Generation finished.")
print()
print(f"Output file:")
print(OUTPUT_FILE)
print("=" * 70)
