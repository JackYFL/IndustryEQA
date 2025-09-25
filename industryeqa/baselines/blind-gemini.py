import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional
import tqdm
from google import genai
import time
from openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="./results/results/small_all_new_new.json", help="path to EQA dataset")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="model")
    parser.add_argument("--output-directory", type=Path, default="./data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 100 questions")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-{}-blind.json".format(args.model))
    return args

def parse_json_from_response(text: str):
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{[^{]*"direct_answer":[^{]*"reasoning_answer":[^}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {"direct_answer": "", "reasoning_answer": "Failed to parse JSON from response"}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"direct_answer": "", "reasoning_answer": f"JSON parse error in: {text}"}

def ask_question_blind(
    question,
    client,
    model: str = "gemini-2.5-pro-preview-03-25",
    max_retries: int = 3,
    retry_delay: int = 20
) -> Optional[dict]:
    import time
    
    retries = 0
    
    while retries <= max_retries:
        try:
            prompt = load_prompt("blind-llm")
            full_prompt = prompt.format(question=question)
            
            response = client.models.generate_content(
                model=model,
                contents=[full_prompt])
            response_text = response.text
            
            try:
                result = parse_json_from_response(response_text)
                return result
            except Exception as e:
                return {
                    "direct_answer": "",
                    "reasoning_answer": f"Error parsing response: {str(e)}. Raw response: {response_text[:100]}..."
                }
                
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                print(f"API call failed: {str(e)}. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries ({max_retries}) reached. Giving up.")
                return {
                    "direct_answer": "",
                    "reasoning_answer": f"API call failed after {max_retries} attempts: {str(e)}"
                }
    
    return {
        "direct_answer": "",
        "reasoning_answer": "Unknown error occurred in retry logic"
    }
    
def main(args: argparse.Namespace):
    assert "GOOGLE_API_KEY" in os.environ
    api_key = [
    ]
    dataset = json.load(args.dataset.open("r", encoding="utf-8"))
    print("found {:,} questions".format(len(dataset)))

    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    client = genai.Client(api_key=api_key[0])
    
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 100:
            break

        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        question = item["question"]
        
        answer_json = ask_question_blind(
            question=question,
            client=client,
            model=args.model,
        )

        results.append({
            "question_id": question_id,
            "question": question,
            "generated_direct_answer": answer_json.get("direct_answer", ""),
            "generated_reasoning_answer": answer_json.get("reasoning_answer", "")
        })

        json.dump(results, args.output_path.open("w"), indent=2)
        
        time.sleep(4)

    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))

if __name__ == "__main__":
    main(parse_args())