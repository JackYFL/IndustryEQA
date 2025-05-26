import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

import tqdm
from google import genai
from openai import OpenAI
from openeqa.utils.prompt_utils import load_prompt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default="./results/results/small_all_new_new.json")
    parser.add_argument("--generated-answers", type=Path, default="./data/results/small_all_new-gemini-2.0-flash-2.json")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--output-directory", type=Path, default="data/scores")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluator", type=str, choices=["gemini", "openai"], default="gemini", 
                        help="Choose evaluation model provider (gemini or openai)")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / f"scores-{args.generated_answers.stem}-{args.model}.json"
    return args

def parse_score_from_response(text: str, score_type: str) -> Dict[str, Any]:
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        score_key = f'"{score_type}_score"'
        json_match = re.search(r'\{[^{]*' + score_key + r':[^}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {f"{score_type}_score": 0, "error": f"Failed to parse JSON from response"}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {f"{score_type}_score": 0, "error": f"JSON parse error in: {text}"}

def gemini_evaluate(
    question: str,
    ground_direct_answer: str,
    generated_direct_answer: str,
    ground_reasoning_answer: str = None,
    generated_reasoning_answer: str = None,
    client = None,
    model: str = "gemini-1.5-pro",
    prompt_type: str = "direct"
) -> Dict[str, Any]:
    """Evaluate answers using Gemini model."""
    prompt = load_prompt(prompt_type)
    
    if prompt_type == "direct":
        full_prompt = prompt.format(
            question=question,
            ground_direct_answer=ground_direct_answer,
            generated_direct_answer=generated_direct_answer,
        )
        score_type = "direct"
    else:
        full_prompt = prompt.format(
            question=question,
            ground_direct_answer=ground_direct_answer,
            generated_direct_answer=generated_direct_answer,
            ground_reasoning_answer=ground_reasoning_answer,
            generated_reasoning_answer=generated_reasoning_answer,
        )
        score_type = "reasoning"
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=full_prompt
        )
        response_text = response.text
        
        result = parse_score_from_response(response_text, score_type)
        return result
    except Exception as e:
        error_msg = str(e)
        print(error_msg)
        return {f"{score_type}_score": 0, "error": error_msg}

def openai_evaluate(
    question: str,
    ground_direct_answer: str,
    generated_direct_answer: str,
    ground_reasoning_answer: str = None,
    generated_reasoning_answer: str = None,
    client = None,
    model: str = "gpt-4o",
    prompt_type: str = "direct"
) -> Dict[str, Any]:
    """Evaluate answers using OpenAI model."""
    prompt = load_prompt(prompt_type)
    
    if prompt_type == "direct":
        full_prompt = prompt.format(
            question=question,
            ground_direct_answer=ground_direct_answer,
            generated_direct_answer=generated_direct_answer,
        )
        score_type = "direct"
    else:
        full_prompt = prompt.format(
            question=question,
            ground_direct_answer=ground_direct_answer,
            generated_direct_answer=generated_direct_answer,
            ground_reasoning_answer=ground_reasoning_answer,
            generated_reasoning_answer=generated_reasoning_answer,
        )
        score_type = "reasoning"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of question answering systems."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
        
        result = parse_score_from_response(response_text, score_type)
        return result
    except Exception as e:
        error_msg = str(e)
        print(error_msg)
        return {f"{score_type}_score": 0, "error": error_msg}

def evaluate_answer(
    question: str,
    ground_direct_answer: str,
    generated_direct_answer: str,
    ground_reasoning_answer: str = None,
    generated_reasoning_answer: str = None,
    client = None,
    model: str = "gemini-1.5-pro",
    prompt_type: str = "direct",
    evaluator: str = "gemini",
) -> Dict[str, Any]:
    """Choose between Gemini or OpenAI for evaluation based on the evaluator parameter."""
    if evaluator == "gemini":
        return gemini_evaluate(
            question=question,
            ground_direct_answer=ground_direct_answer,
            generated_direct_answer=generated_direct_answer,
            ground_reasoning_answer=ground_reasoning_answer,
            generated_reasoning_answer=generated_reasoning_answer,
            client=client,
            model=model,
            prompt_type=prompt_type
        )
    elif evaluator == "openai":
        return openai_evaluate(
            question=question,
            ground_direct_answer=ground_direct_answer,
            generated_direct_answer=generated_direct_answer,
            ground_reasoning_answer=ground_reasoning_answer,
            generated_reasoning_answer=generated_reasoning_answer,
            client=client,
            model=model,
            prompt_type=prompt_type
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")

def main(args: argparse.Namespace):
    # Set up clients based on evaluator choice
    gemini_client = None
    openai_client = None
    
    if args.evaluator == "gemini":
        gemini_api_key = ""
        gemini_client = genai.Client(api_key=gemini_api_key)
    elif args.evaluator == "openai":
        openai_client = OpenAI(
            api_key=""
        )
    
    ground_truth = json.load(args.ground_truth.open("r", encoding="utf-8"))
    generated_answers = json.load(args.generated_answers.open("r", encoding="utf-8"))
    print(f"Found {len(ground_truth)} ground truth items and {len(generated_answers)} generated answers")
    
    ground_truth_dict = {item["question_id"]: item for item in ground_truth}
    
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open("r"))
        print(f"Found {len(results)} existing results")
    
    completed = [item["question_id"] for item in results]
    
    for idx, item in enumerate(tqdm.tqdm(generated_answers)):
        if args.dry_run and idx >= 5:
            break
            
        question_id = item["question_id"]
        if question_id in completed:
            continue
            
        if question_id not in ground_truth_dict:
            print(f"Warning: Question ID {question_id} not found in ground truth")
            continue
            
        gt_item = ground_truth_dict[question_id]
        question = gt_item["question"]
        
        try:
            client = gemini_client if args.evaluator == "gemini" else openai_client
            
            direct_score_result = evaluate_answer(
                question=question,
                ground_direct_answer=gt_item["direct_answer"],
                generated_direct_answer=item.get("generated_direct_answer", ""),
                client=client,
                model=args.model,
                prompt_type="direct",
                evaluator=args.evaluator,
            )
            
            comprehensive_result = evaluate_answer(
                question=question,
                ground_direct_answer=gt_item["direct_answer"],
                generated_direct_answer=item.get("generated_direct_answer", ""),
                ground_reasoning_answer=gt_item.get("reasoning_answer", ""),
                generated_reasoning_answer=item.get("generated_reasoning_answer", ""),
                client=client,
                model=args.model,
                prompt_type="reasoning",
                evaluator=args.evaluator,
            )
                
            result = {
                "question_id": question_id,
                "question": question,
                "ground_direct_answer": gt_item["direct_answer"],
                "generated_direct_answer": item.get("generated_direct_answer", ""),
                "ground_reasoning_answer": gt_item.get("reasoning_answer", ""),
                "generated_reasoning_answer": item.get("generated_reasoning_answer", ""),
                "direct_score": direct_score_result.get("direct_score", 0),
                "reasoning_score": comprehensive_result.get("reasoning_score", 0),
                "model": args.model
            }
            
            if "error" in direct_score_result or "error" in comprehensive_result:
                continue
                
            results.append(result)
            
            json.dump(results, args.output_path.open("w"), indent=2)
            
            # Add a delay to avoid rate limiting (different for each provider)
            if args.evaluator == "gemini":
                time.sleep(8)  # Original delay for Gemini
            
        except Exception as e:
            if not args.force:
                raise e
            print(f"Error processing question {question_id}: {str(e)}")
        
    json.dump(results, args.output_path.open("w"), indent=2)
    print(f"Saved {len(results)} evaluation results to {args.output_path}")
    
if __name__ == "__main__":
    main(parse_args())