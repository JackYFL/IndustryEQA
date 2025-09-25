import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional
import tqdm
from google import genai
import os
from openeqa.utils.prompt_utils import load_prompt
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="./results/results/small_all_new_new.json", help="path to EQA dataset")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help=" model")
    parser.add_argument("--output-directory", type=Path, default="./data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    args = parser.parse_args(); 
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-{}.json".format(args.model))
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


def ask_question_gemini_video(
    question,
    video_file,
    client,
    model: str = "gemini-2.5-pro-preview-03-25",
) -> Optional[str]:
    
    retries = 0
    
    while retries <= 2:
        try:
                    
            prompt = load_prompt("evaluate")
            full_prompt = prompt.format(question=question)
            
            # Pass the video file reference like any other media part.
            response = client.models.generate_content(
                model=model,
                contents=[
                    video_file,
                    full_prompt])
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
            if retries <= 2:
                print(f"API call failed: {str(e)}. Retrying in {10} seconds... (Attempt {retries}/{2})")
                time.sleep(10)
            else:
                exit()
                
                
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

    added_path = []
    video_file_dict = {}
    client = genai.Client(api_key=api_key[0])
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 100:
            break

        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        question = item["question"]
      
        file_path = item["path"]
        
        if file_path not in added_path:
            added_path.append(file_path)
            print(file_path)
            print("Uploading file...")
            video_file = client.files.upload(file=file_path)
            print(f"Completed upload: {video_file.uri}")

            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            print('Done')
            video_file_dict[file_path]=video_file
        else:
            video_file = video_file_dict[file_path]
            
        answer_json = ask_question_gemini_video(
            question=question,
            video_file=video_file,
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
        
        
        time.sleep(10)

    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results))) 

if __name__ == "__main__":
    main(parse_args())
