import argparse
import json
import os
import re
import time
import cv2
import base64
from pathlib import Path
from typing import List, Optional
import tqdm
from openai import OpenAI
from openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="./annotation/industryeqa.json", help="path to EQA dataset")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--output-directory", type=Path, default="./data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 100 questions")
    parser.add_argument("--frames", type=int, default=30, help="number of frames to extract from video")
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



def ask_question_openai(
    question,
    client,
    model: str = "gpt-4o",
) -> Optional[dict]:
    """Ask a question about a video using OpenAI's API."""
    

    prompt = load_prompt("blind-llm")
    full_prompt = prompt.format(question=question)
    
    messages_content = [{"type": "text", "text": full_prompt}]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": messages_content
                }
            ],
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
        
        result = parse_json_from_response(response_text)
        return result
    except Exception as e:
        return {
            "direct_answer": "",
            "reasoning_answer": f"API Error: {str(e)}"
        }

def main(args: argparse.Namespace):
    dataset = json.load(args.dataset.open("r", encoding="utf-8"))
    print("Found {:,} questions".format(len(dataset)))
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("Found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    client = OpenAI(api_key="")
    
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        question_id = item["question_id"]
        if question_id in completed:
            continue  # Skip existing

        question = item["question"]
        file_path = item["path"]
        
        answer_json = ask_question_openai(
            question=question,
            client=client,
            model=args.model
        )

        results.append({
            "question_id": question_id,
            "question": question,
            "generated_direct_answer": answer_json.get("direct_answer", ""),
            "generated_reasoning_answer": answer_json.get("reasoning_answer", "")
        })

        json.dump(results, args.output_path.open("w"), indent=2)
        
        time.sleep(1)

    json.dump(results, args.output_path.open("w"), indent=2)
    print("Saved {:,} answers".format(len(results)))

if __name__ == "__main__":
    main(parse_args())