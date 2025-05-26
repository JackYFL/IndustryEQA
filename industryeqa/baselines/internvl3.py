import argparse
import json
import os
import re
import time
import cv2
import base64
from pathlib import Path
from typing import List, Optional, Dict
import tqdm
from openai import OpenAI
from openeqa.utils.prompt_utils import load_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="./results/results/small_all_new_new.json", help="path to EQA dataset")
    parser.add_argument("--model", type=str, default="internvl2.5-78b", help="OpenRouter model to use")
    parser.add_argument("--output-directory", type=Path, default="./data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    parser.add_argument("--frames", type=int, default=10, help="number of frames to extract from video")
    parser.add_argument("--app-name", type=str, default="VideoEvaluation", help="app name for OpenRouter rankings")
    parser.add_argument("--cache-frames", action="store_true", default=True, help="cache extracted frames (default: true)")
    parser.add_argument("--cache-dir", type=Path, default="./data/frames_cache", help="directory for caching frames")
    parser.add_argument("--debug", action="store_true", help="enable debug mode with additional logging")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.cache_frames:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + f"-{args.model}.json")
    return args


def parse_json_from_response(text: str, debug=False):
    """Extract JSON from the model response."""
    if debug:
        print(f"Raw response: {text[:200]}...")  # Print first 200 chars for debug
    
    # Look for JSON content between triple backticks and json
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If not found with ```json format, try just finding JSON object
        json_match = re.search(r'\{[^{]*"direct_answer":[^{]*"reasoning_answer":[^}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {"direct_answer": "", "reasoning_answer": "Failed to parse JSON from response"}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON Decode Error: {e}")
            print(f"Problematic string: {json_str[:100]}...")
        return {"direct_answer": "", "reasoning_answer": f"JSON parse error in: {text[:100]}..."}


def extract_and_encode_frames(video_path, num_frames_to_extract=5, debug=False):
    """Extract frames from video and encode them to base64."""
    base64_frames = []
    video_path_str = str(video_path)  # Convert Path to string if needed
    
    try:
        if debug:
            print(f"Opening video: {video_path_str}")
        
        video = cv2.VideoCapture(video_path_str)
        if not video.isOpened():
            print(f"Could not open video: {video_path}")
            return base64_frames
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        if debug:
            print(f"Video stats: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        if total_frames == 0:
            video.release()
            print(f"Video has 0 frames: {video_path}")
            return base64_frames
        
        num_frames_to_extract = min(num_frames_to_extract, total_frames)
        
        if num_frames_to_extract == 1:
            frame_indices = [total_frames // 2]
        elif num_frames_to_extract > 1:
            step = max(1, total_frames // num_frames_to_extract)
            frame_indices = [min(i * step, total_frames - 1) for i in range(num_frames_to_extract)]
            frame_indices = sorted(list(set(frame_indices)))
        else:
            video.release()
            return base64_frames
        
        if debug:
            print(f"Extracting {len(frame_indices)} frames at indices: {frame_indices[:5]}...")
        
        successful_frames = 0
        for frame_idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                if debug:
                    print(f"Failed to read frame at index {frame_idx}")
                continue
            
            # Check if frame is valid
            if frame is None or frame.size == 0:
                if debug:
                    print(f"Empty frame at index {frame_idx}")
                continue
                
            # Resize very large frames to reduce size
            h, w = frame.shape[:2]
            max_dim = 768
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                if debug:
                    print(f"Resized frame from {w}x{h} to {new_w}x{new_h}")
            
            # Convert to JPEG with quality setting
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            successful_frames += 1
        
        video.release()
        
        if debug:
            print(f"Successfully extracted {successful_frames} frames")
            
        return base64_frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return base64_frames


def get_video_frames(
    video_path, 
    num_frames: int = 5, 
    use_cache: bool = True, 
    cache_dir: Optional[Path] = None,
    debug: bool = False
) -> List[str]:
    """Get frames from video, using cache if available."""
    video_path_x = Path(video_path)
    
    if not video_path_x.exists():
        print(f"Error: Video file does not exist: {video_path_x}")
        return []
    
    if not use_cache or cache_dir is None:
        if debug:
            print(f"Cache disabled, extracting frames directly")
        return extract_and_encode_frames(video_path_x, num_frames, debug)
    
    # Create a unique filename for the cached frames based on video path and frame count
    temp_path_segment = video_path.replace("/", "_")
    cache_key = f"{temp_path_segment}_{num_frames}frames.json"
    cache_file = cache_dir / cache_key
    
    # If cache exists, load frames from cache
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                print(f"Using cached frames for {video_path_x.name}")
                cached_frames = json.load(f)
                # Validate cache
                if cached_frames and len(cached_frames) > 0:
                    if debug:
                        print(f"Cache valid: {len(cached_frames)} frames loaded")
                    # Sample check of base64 data
                    if isinstance(cached_frames[0], str) and cached_frames[0].startswith('/9j/'):
                        return cached_frames
                    else:
                        if debug:
                            print(f"Cache invalid: Frames appear to be corrupted")
                else:
                    print(f"Empty cache for {video_path_x.name}, re-extracting frames...")
        except Exception as e:
            print(f"Error loading cache: {e}, extracting frames...")
    
    # Extract frames
    if debug:
        print(f"Extracting new frames from {video_path_x.name}")
    base64_frames = extract_and_encode_frames(video_path_x, num_frames, debug)
    
    # Save to cache if frames were successfully extracted
    if base64_frames and len(base64_frames) > 0 and cache_dir is not None:
        try:
            with open(cache_file, 'w') as f:
                json.dump(base64_frames, f)
            print(f"Saved {len(base64_frames)} frames to cache for {video_path_x.name}")
        except Exception as e:
            print(f"Failed to save frames to cache: {e}")
    elif debug:
        print(f"No frames extracted or cache disabled, not saving to cache")
    
    return base64_frames


def ask_question_openrouter_video(
    question,
    video_path,
    client,
    model: str = "opengvlab/internvl3-14b:free",
    num_frames: int = 5,
    app_name: str = "VideoEvaluation",
    app_url: str = "https://yourdomain.com",
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    debug: bool = False
) -> dict:
    """Ask a question about a video using OpenRouter's API."""
    # Get frames (from cache if available)
    base64_frames = get_video_frames(video_path, num_frames, use_cache, cache_dir, debug)
    
    if not base64_frames or len(base64_frames) == 0:
        return {
            "direct_answer": "",
            "reasoning_answer": f"Error: Could not extract frames from {video_path}"
        }
    
    if debug:
        print(f"Got {len(base64_frames)} frames, preparing prompt...")
    
    # Prepare prompt
    try:
        prompt = load_prompt("evaluate")
        full_prompt = prompt.format(question=question)
    except Exception as e:
        return {
            "direct_answer": "",
            "reasoning_answer": f"Prompt Error: {str(e)}"
        }
    
    # Prepare message content
    messages_content = [{"type": "text", "text": full_prompt}]
    
    frames_to_send = base64_frames
    
    if debug and len(frames_to_send) < len(base64_frames):
        print(f"Limiting frames sent to API from {len(base64_frames)} to {len(frames_to_send)}")
    
    # Append frames to message content
    for base64_frame in frames_to_send:
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_frame}",
                "detail": "low"
            }
        })
    
    # Calculate approximate token count for base64 images
    approx_token_count = sum(len(frame) // 4 for frame in frames_to_send)
    if debug:
        print(f"Sending request with ~{approx_token_count} tokens of image data")
    
    try:
        # Make the API call with retry mechanism
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                if debug:
                    print(f"API call attempt {attempt+1}/{max_retries}")
                
                # Make the API call
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
                print(response)
                # Debug response
                if debug:
                    print(f"API response status: {getattr(response, 'status', 'unknown')}")
                
                # Basic validation of response
                if response is None:
                    if debug:
                        print("Empty response received")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return {
                        "direct_answer": "",
                        "reasoning_answer": "API Error: Empty response"
                    }
                
                # Check for choices in response
                if not hasattr(response, 'choices') or not response.choices:
                    if debug:
                        print("No choices in response")
                        print(f"Response keys: {dir(response)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return {
                        "direct_answer": "",
                        "reasoning_answer": "API Error: No choices in response"
                    }
                
                # Process the response
                message = response.choices[0].message
                if not message or not hasattr(message, 'content') or not message.content:
                    if debug:
                        print("No message content in response")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return {
                        "direct_answer": "",
                        "reasoning_answer": "API Error: No content in message"
                    }
                
                response_text = message.content
                
                # Parse the JSON response
                result = parse_json_from_response(response_text, debug)
                return result
                
            except Exception as e:
                if debug:
                    print(f"API call attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return {
                        "direct_answer": "",
                        "reasoning_answer": f"API Error after {max_retries} attempts: {str(e)}"
                    }
        
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return {
            "direct_answer": "",
            "reasoning_answer": f"API Error: {str(e)}"
        }


def main(args: argparse.Namespace):
    # Load dataset
    try:
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Found {len(dataset):,} questions in dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load existing results
    results = []
    if args.output_path.exists():
        try:
            with open(args.output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Found {len(results):,} existing results")
        except Exception as e:
            print(f"Error loading existing results: {e}, starting with empty results")
    
    # Track completed questions
    completed = {item["question_id"] for item in results}


    client = OpenAI(
        api_key="",
        base_url="https://chat.intern-ai.org.cn/api/v1/",
    )
    
    # Process data
    try:
        for idx, item in enumerate(tqdm.tqdm(dataset)):
            if args.dry_run and idx >= 1:
                print("Dry run completed")
                break

            # Skip completed questions
            question_id = item["question_id"]
            if question_id in completed:
                continue  # Skip existing
            
            question = item.get("question", "")
            file_path = item.get("path", "")
            
            if not question or not file_path:
                print(f"Skipping item {question_id}: Missing question or path")
                continue
                
            if not os.path.exists(file_path):
                print(f"Skipping item {question_id}: File not found: {file_path}")
                continue
            
            print(f"Processing question {question_id}: {question[:50]}...")
            
            answer_json = ask_question_openrouter_video(
                question=question,
                video_path=file_path,
                client=client,
                model=args.model,
                num_frames=args.frames,
                app_name=args.app_name,
                app_url=args.app_url,
                use_cache=args.cache_frames,
                cache_dir=args.cache_dir,
                debug=args.debug
            )

            results.append({
                "question_id": question_id,
                "question": question,
                "generated_direct_answer": answer_json.get("direct_answer", ""),
                "generated_reasoning_answer": answer_json.get("reasoning_answer", "")
            })

            # Save results after each question
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        # Final save
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results):,} answers")

if __name__ == "__main__":
    main(parse_args())