from pathlib import Path
# pyrefly: ignore [missing-import]
from datasets import load_dataset
import json

# Load the Kannada gsm8k-indic test set
dataset = load_dataset("sarvamai/gsm8k-indic", "kn", split="test")

# Take the first 50 questions
questions = []

for i in range(min(50, len(dataset))):
    questions.append({
        "question_id": i + 1,
        "question": dataset[i]["question"],
        "answer": dataset[i]["answer"],
        "original_question": dataset[i]["original_question"],
        "original_answer": dataset[i]["original_answer"]
    })

# Save them
with open(Path(__file__).resolve().with_name("questions.json"), "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"Saved {len(questions)} Kannada questions.")

for q in questions[:5]:
    print(f"\n--- Question {q['question_id']} ---")
    print(q["question"])
    print(f"Answer: {q['answer']}")
