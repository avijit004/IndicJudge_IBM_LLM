import json
import csv
import os
import requests
import time
import argparse

# ============================================================
# Retry Configuration
# ============================================================
# We increase MAX_TOKENS to give the model more room to finish its thinking.
MAX_TOKENS = 8192
# Slightly adjust temperature to try and break it out of endless loops.
TEMPERATURE = 0.85

def fill_blanks(csv_file, api_url):
    print(f"Processing {csv_file}...")
    
    # 1. Read all rows
    rows = []
    blanks_to_fill = []
    
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader):
            rows.append(row)
            if not row["response"] or str(row["response"]).strip() == "" or str(row["response"]).lower() == "nan":
                blanks_to_fill.append(i)
                
    if not blanks_to_fill:
        print("No blank responses found. You are all set!")
        return

    print(f"Found {len(blanks_to_fill)} blank responses. Retrying with MAX_TOKENS={MAX_TOKENS}...")
    
    # 2. Process the blank rows
    for count, row_idx in enumerate(blanks_to_fill, 1):
        row = rows[row_idx]
        question_id = row["question_id"]
        prompt_id = row["prompt_id"]
        question_text = row["question"]
        prompt_text = row["prompt"]
        
        print("-" * 50)
        print(f"[{count}/{len(blanks_to_fill)}] Retrying Question {question_id} | Prompt {prompt_id}")
        
        full_prompt = prompt_text + "\n\n" + question_text + "\n\n(IMPORTANT: You MUST write your entire reasoning and final answer in Kannada.)"
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "frequency_penalty": 1.2,
            "presence_penalty": 1.2
        }
        
        try:
            response = requests.post(api_url, json=payload, timeout=900) # Increased timeout
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            
            reasoning_content = message.get("reasoning_content", "")
            answer = message.get("content", "")
            
            if answer and answer.strip():
                # Update the row in memory
                rows[row_idx]["reasoning_content"] = reasoning_content
                rows[row_idx]["response"] = answer
                print("SUCCESS: Generated a non-empty response!")
            elif reasoning_content and reasoning_content.strip():
                rows[row_idx]["reasoning_content"] = reasoning_content
                rows[row_idx]["response"] = "[SALVAGED FROM REASONING] " + reasoning_content[-2000:]
                print("SUCCESS: Salvaged response from reasoning block!")
            else:
                print("WARNING: Model still generated an empty response after 8192 tokens.")
                
        except Exception as e:
            print(f"ERROR querying the API: {e}")
            
    # 3. Write everything back to the CSV
    print("\nSaving updated results back to the CSV...")
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill blank responses in a CSV")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--url", default="http://172.30.1.83:8080/v1/chat/completions", help="API URL")
    
    args = parser.parse_args()
    fill_blanks(args.csv, args.url)
