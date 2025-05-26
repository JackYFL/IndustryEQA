import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional
import base64
import numpy as np
import tqdm
import openai
from openai import OpenAI
from openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="data/open-eqa-v0.json", help="path to EQA dataset (default: data/open-eqa-v0.json)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="GPT model")
    parser.add_argument("--temperature", type=float, default=0.9, help="gpt temperature (default: 0.2)")
    parser.add_argument("--frames-directory", type=Path,default="data/frames/",help="path image frames (default: data/frames/)",)
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


def ask_question(
    question: str,
    image_paths: List,
    openai_model: str = "gpt-4o",
    openai_temperature: float = 0.2,
) -> Optional[str]:
    client = OpenAI()
    prompt = load_prompt("gpt4o")
    if "User Query:" in prompt:
        prefix, suffix_template = prompt.split("User Query:", 1)
        suffix = "User Query:" + suffix_template.format(question=question)
        full_prompt = prefix + suffix
    else:
        full_prompt = prompt.format(question=question)

    content = [{"type": "input_text", "text": full_prompt}]
    for path in image_paths:
        with open(path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        content.append({
            "type": "input_image",
            "image_url": f"data:image/png;base64,{encoded}",
            "detail": "low",
        })
        
    response = client.responses.create(
        model=openai_model,
        input=[{"role": "user", "content": content}],
        # max_output_tokens=openai_max_tokens,
        temperature=openai_temperature,
    )
    return response.output_text



def main(args: argparse.Namespace):
    assert "OPENAI_API_KEY" in os.environ

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
            answer = ask_question(
                question=question,
                image_paths=paths,
                openai_model=args.model,
                openai_temperature=args.temperature,
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
