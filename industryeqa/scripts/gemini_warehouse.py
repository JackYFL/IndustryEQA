import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional
import base64
import numpy as np
import tqdm
from google import genai
# import google.generativeai as genai
import os
from PIL import Image
from openeqa.utils.prompt_utils import load_prompt
import time
import shutil
import re

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="data/open-eqa-v0.json", help="path to EQA dataset (default: data/open-eqa-v0.json)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-exp-03-25", help=" model")
    parser.add_argument("--output-directory", type=Path, default="./results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    args = parser.parse_args(); 
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-{}.json".format(args.model))
    return args

def ask_question_gemini_video(
    api_key,
    model: str = "gemini-2.5-pro-preview-03-25",
    path: str = "",
) -> Optional[str]:
    
    if os.path.exists(os.path.splitext(path)[0] + ".txt"):
        print(f"break for {path}")
        return
        
    client = genai.Client(api_key=api_key)
    
    print("Uploading file...")
    video_file = client.files.upload(file=path)
    print(f"Completed upload: {video_file.uri}")

    # Check whether the file is ready to be used.
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    print('Done')
    
    prompt = load_prompt("small")
    
    # Pass the video file reference like any other media part.
    response = client.models.generate_content(
        model=model,
        contents=[
            video_file,
            prompt])

    response_text = response.text
    
    json_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(json_path):
        print(f"Skipping existing file: {json_path}")
        return
    
    # Try to extract JSON content if wrapped in code blocks
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    json_match = re.search(json_pattern, response_text)
    
    if json_match:
        json_content = json_match.group(1)
    else:
        # If not in code blocks, use the whole text
        json_content = response_text
    
    try:
        # Validate JSON by parsing it
        qa_pairs = json.loads(json_content)
        
        # Save the parsed JSON to file
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(qa_pairs, file, indent=4, ensure_ascii=False)
        
        print(f"Successfully saved JSON output to {json_path}")
        return qa_pairs
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("Raw response text:")
        print(response_text)
        
        # Save the raw text as a fallback
        with open(os.path.splitext(path)[0] + "_raw.txt", "w", encoding="utf-8") as file:
            file.write(response_text)
        
        print(f"Saved raw response to {os.path.splitext(path)[0]}_raw.txt")
        return None
    
    
def main(args: argparse.Namespace):
    assert "GOOGLE_API_KEY" in os.environ
    api_key = ""
    video_dir = Path("./data/small")
    video_exts = {'.mp4'}
    paths = [str(p) for p in video_dir.rglob("*") if p.suffix.lower() in video_exts]
    print(f"Found {len(paths)} video files")
    
    # Process each video
    for path in paths:
        json_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(json_path):
            print(f"skip: {path}")
            continue
        try:
            print(f"Processing: {path}")
            answer = ask_question_gemini_video(
                api_key=api_key,
                model=args.model,
                path=path,
            )
            print(answer)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            if not args.force:
                raise
                
    # Combine all results into one JSON file
    # combine_json_results(paths)
    
if __name__ == "__main__":
    main(parse_args())

def combine_json_results(paths, output_file="data/results/2_combined_qa_pairs.json"):
    """Combine all individual JSON results into a single JSON file."""
    all_qa_pairs = []
    
    for path in paths:
        json_path = os.path.splitext(path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)
                
                # Add path to each QA pair
                for qa in qa_pairs:
                    qa['path'] = path  # Add the source video path
                
                all_qa_pairs.extend(qa_pairs)
            except json.JSONDecodeError:
                print(f"Error reading {json_path}")
    
    # Save combined results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, indent=4, ensure_ascii=False)
    
    print(f"Combined {len(all_qa_pairs)} QA pairs from {len(paths)} videos to {output_file}")
    return all_qa_pairs