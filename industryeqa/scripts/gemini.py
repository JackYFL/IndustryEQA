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
import os
from PIL import Image
from openeqa.utils.prompt_utils import load_prompt
import time
from IPython.display import Markdown

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="data/open-eqa-v0.json", help="path to EQA dataset (default: data/open-eqa-v0.json)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help=" model")
    parser.add_argument("--temperature", type=float, default=0.9, help="gpt temperature (default: 0.2)")
    parser.add_argument("--frames-directory", type=Path,default="data/frames/",help="path image frames (default: data/frames/)",)
    parser.add_argument("--type",type=str, default="image",help="video or images",)
    parser.add_argument("--num-frames",type=int,default=20,help="num frames in gpt4v (default: 50)",)
    parser.add_argument("--max-tokens", type=int, default=128, help="gpt maximum tokens (default: 128)")
    parser.add_argument("--output-directory", type=Path, default="data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    parser.add_argument("--num-generations", type=int, default=10, help="response size (default: 5)",)
    args = parser.parse_args(); 
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-{}.json".format(args.model))
    return args


def ask_question_gemini_image(
    question: str,
    image_paths: List[str],
    model: str = "gemini-2.5-pro-preview-03-25",
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> Optional[str]:

    prompt = load_prompt("gpt4o")
    if "User Query:" in prompt:
        prefix, suffix_template = prompt.split("User Query:", 1)
        suffix = "User Query:" + suffix_template.format(question=question)
        full_prompt = prefix + suffix
    else:
        full_prompt = prompt.format(question=question)
        
    content = []
    for path in image_paths:
        content.append(Image.open(path))
        
    content.append(full_prompt)
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=content)
    print(response.text)
    
    return(response.text)
    
def ask_question_gemini_video(
    question: str,
    image_paths: List[str],
    model: str = "gemini-2.5-pro-preview-03-25",
    temperature: float = 0.8,
    api_key: Optional[str] = None,
) -> Optional[str]:

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    
    print("Uploading file...")
    video_file = client.files.upload(file="./data/warehouse/small_warehouse1.mp4")
    print(f"Completed upload: {video_file.uri}")

    # Check whether the file is ready to be used.
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    print('Done')
    
    prompt = load_prompt("gemini_warehouse")
    if "User Query:" in prompt:
        prefix, suffix_template = prompt.split("User Query:", 1)
        suffix = "User Query:" + suffix_template.format(question=question)
        full_prompt = prefix + suffix
    else:
        full_prompt = prompt.format(question=question)
    
    # Pass the video file reference like any other media part.
    response = client.models.generate_content(
        model=model,
        contents=[
            video_file,
            prompt])

    # Print the response, rendering any Markdown
    Markdown(response.text)
    
    
def main(args: argparse.Namespace):
    assert "GOOGLE_API_KEY" in os.environ

    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # extract scene paths
        folder = args.frames_directory / item["episode_history"]
        frames = sorted(folder.glob("*-rgb.png"))

        question = item["question"]
        answers = []
        for attempt in range(args.num_generations):
            indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
            shifted_indices = [
                np.clip(index + np.random.randint(-3, 4), 0, len(frames) - 1)
                for index in indices
            ]
            paths = [str(frames[i]) for i in shifted_indices]

            # generate answer 10 times
            if args.type == "image":
                answer = ask_question_gemini_image(
                    question=question,
                    image_paths=paths,
                    model=args.model,
                    temperature=args.temperature,
                )
            elif args.type == "video":
                answer = ask_question_gemini_video(
                    question=question,
                    image_paths=paths,
                    model=args.model,
                    temperature=args.temperature,
                )
                
            answers.append({
                "attempt": attempt + 1,
                "answer": answer
            })

        results.append({
            "question_id": question_id,
            "answers": answers
        })

        json.dump(results, args.output_path.open("w"), indent=2)

    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results))) 

if __name__ == "__main__":
    main(parse_args())
