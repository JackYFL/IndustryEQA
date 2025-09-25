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
    args.output_path = args.output_directory / (args.dataset.stem + f"-{args.frames}-{args.model}.json")
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


def extract_and_encode_frames(video_path, num_frames_to_extract=5):
    base64_frames = []
    video_path_str = str(video_path)  # Convert Path to string if needed
    
    video = cv2.VideoCapture(video_path_str)
    if not video.isOpened():
        print(f"Could not open video: {video_path}")
        return base64_frames
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    if total_frames == 0:
        video.release()
        print(f"Video has 0 frames: {video_path}")
        return base64_frames
    
    num_frames_to_extract = min(num_frames_to_extract, total_frames)
    
    if num_frames_to_extract == 1:
        frame_indices = [total_frames // 2]
    elif num_frames_to_extract > 1:
        frame_indices = [int(i * (total_frames - 1) / (num_frames_to_extract - 1)) 
                         for i in range(num_frames_to_extract)]
    else:
        video.release()
        return base64_frames
    
    successful_frames = 0
    for frame_idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if not success:
            continue
        
        if frame is None or frame.size == 0:
            continue
            
        h, w = frame.shape[:2]
        max_dim = 768
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode(".jpg", frame, encode_param)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        successful_frames += 1
    
    video.release()
    
    return base64_frames


def get_video_frames(
    video_path, 
    num_frames: int = 5, 
    use_cache: bool = True, 
    cache_dir: Optional[Path] = None
) -> List[str]:
    """Get frames from video, using cache if available."""
    if not use_cache or cache_dir is None:
        return extract_and_encode_frames(video_path, num_frames)
    

    temp_path_segment = video_path.replace("/", "_")
    cache_key = f"{temp_path_segment}_{num_frames}frames.json"
    cache_file = cache_dir / cache_key
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            print(f"Using cached frames for {video_path}")
            cached_frames = json.load(f)
            if cached_frames and len(cached_frames) > 0:
                return cached_frames
    
    base64_frames = extract_and_encode_frames(video_path, num_frames)
    
    if base64_frames and len(base64_frames) > 0 and cache_dir is not None:
        try:
            with open(cache_file, 'w') as f:
                json.dump(base64_frames, f)
            print(f"Saved {len(base64_frames)} frames to cache for {video_path.name}")
        except Exception:
            pass
    
    return base64_frames


def ask_question_openai_video(
    question,
    video_path,
    client,
    model: str = "gpt-4o",
    num_frames: int = 50,
    use_cache: bool = True, 
    cache_dir: Optional[Path] = None
) -> Optional[dict]:
    """Ask a question about a video using OpenAI's API."""
    
    # Get frames (from cache if available)
    base64_frames = get_video_frames(video_path, num_frames, use_cache, cache_dir)
    
    if not base64_frames or len(base64_frames) == 0:
        print(f"Warning: Could not extract frames from {video_path}")
        
    if not base64_frames:
        return {
            "direct_answer": "",
            "reasoning_answer": f"Error: Could not extract frames from {video_path}"
        }
    
    # Prepare prompt
    prompt = load_prompt("evaluate")
    full_prompt = prompt.format(question=question)
    
    # Prepare message content
    messages_content = [{"type": "text", "text": full_prompt}]
    for base64_frame in base64_frames:
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}",
                "detail": "low"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": messages_content
                }
            ]
        )
        response_text = response.choices[0].message.content
        
        # Parse the JSON response
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
        
        answer_json = ask_question_openai_video(
            question=question,
            video_path=file_path,
            client=client,
            model=args.model,
            num_frames=args.frames
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