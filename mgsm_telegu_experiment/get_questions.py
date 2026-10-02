from pathlib import Path
# pyrefly: ignore [missing-import]
from datasets import load_dataset
import json

# Load the Telugu MGSM test set
dataset = load_dataset("juletxara/mgsm", "te", split="test")

# Take the first 50 questions
questions = []

for i in range(min(50, len(dataset))):
    questions.append({
        "question_id": i + 1,
        "question": dataset[i]["question"],
        "answer_number": dataset[i]["answer_number"]
    })

# Save them
with open(Path(__file__).resolve().with_name("questions.json"), "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"Saved {len(questions)} questions.")

for q in questions:
    print(f"\n--- Question {q['question_id']} ---")
    print(q["question"])
