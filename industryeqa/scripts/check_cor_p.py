import os
import json
import argparse
import time
import traceback
from pathlib import Path
from openeqa.utils.prompt_utils import load_prompt
from google import genai

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-preview-04-17", help="model to use")
    parser.add_argument("--output-directory", type=Path, default="data/validated_results", help="output directory")
    parser.add_argument("--input-json", type=Path, default="data/results/large_combined_qa_pairs.json", help="input JSON file with QA pairs")
    parser.add_argument("--start-index", type=int, default=0, help="start index for processing")
    parser.add_argument("--end-index", type=int, default=None, help="end index for processing (exclusive)")
    parser.add_argument("--batch-size", type=int, default=150, help="number of items to process in this batch")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    return args

def validate_qa_with_gemini(
    client,
    video_file,
    question: str,
    answer: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    prompt = load_prompt("check")
    prompt = prompt.format(question=question, answer=answer)    
    
    response = client.models.generate_content(
        model=model,
        contents=[video_file, prompt]
    )
    print("Gemini response received")
    
    response_text = response.text
    try:
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from response: {response_text}")
        return {
            "remain": 1,
            "correct": 1,
            "new_answer": "Failed to evaluate"
        }

def process_qa_data(args):
    
    api_keys = [
    ]
    
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading input JSON file: {e}")
        return
    
    # Calculate end_index if not provided
    if args.end_index is None:
        args.end_index = min(args.start_index + args.batch_size, len(qa_data))
    else:
        args.end_index = min(args.end_index, len(qa_data))
    
    print(f"Processing items from index {args.start_index} to {args.end_index-1} (Total: {args.end_index-args.start_index})")
    
    # Create a unique output file for this batch
    batch_identifier = f"batch_{args.start_index}_{args.end_index}"
    output_path = args.output_directory / f"validated_results_{batch_identifier}.json"
    
    validated_qa_pairs = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                validated_qa_pairs = json.load(f)
            print(f"Loaded {len(validated_qa_pairs)} existing validated pairs for this batch")
        except (json.JSONDecodeError, FileNotFoundError):
            validated_qa_pairs = []
    
    processed_ids = {qa["id"] for qa in validated_qa_pairs if "id" in qa}
    # Initialize client with the provided API key
    client = genai.Client(api_key=api_keys[0])
    
    uploaded_videos = {}
    
    # Only process items in the specified range
    batch_qa_data = qa_data[args.start_index:args.end_index]
    
    for index, qa_pair in enumerate(batch_qa_data):
        global_index = index + args.start_index
        qa_id = qa_pair.get("id")
        question = qa_pair.get("question", "")
        answer = qa_pair.get("direct_answer", "")
        video_path = qa_pair.get("path", "")
        print(question)
        print(answer)
        print(video_path)
        if not question or not answer or not video_path:
            print("skip")
            continue
            
        if qa_id in processed_ids:
            print(f"Skipping already processed question ID {qa_id}: {question[:30]}...")
            continue
            
        print(f"Processing question {index + 1}/{len(batch_qa_data)} (Global ID: {global_index}, ID: {qa_id}): {question[:30]}...")
        
        try:
            if video_path in uploaded_videos:
                video_file = uploaded_videos[video_path]
                print(f"Using existing uploaded video: {video_path}")
            else:
                print(f"Uploading video: {video_path}")
                
                try:
                    video_file = client.files.upload(file=video_path)
                    print(f"Completed upload: {video_file.uri}")
                    
                    while video_file.state.name == "PROCESSING":
                        print('.', end='')
                        time.sleep(1)
                        video_file = client.files.get(name=video_file.name)

                    if video_file.state.name == "FAILED":
                        raise ValueError(f"Video processing failed: {video_file.state.name}")
                    print('Video processing done')
                    
                    uploaded_videos[video_path] = video_file
                except Exception as e:
                    print(f"Error uploading video: {e}")
                    print(traceback.format_exc())
                    continue
            
            try:
                validation_result = validate_qa_with_gemini(
                    client,
                    video_file,
                    question=question,
                    answer=answer,
                    model=args.model,
                )
                
                validated_qa = {**qa_pair, **validation_result}
                validated_qa_pairs.append(validated_qa)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(validated_qa_pairs, f, indent=2)
                print(f"Saved progress to: {output_path}")
                
                processed_ids.add(qa_id)
                time.sleep(6)  # Rate limiting
                
            except Exception as e:
                print(f"Error validating QA: {e}")
                print(traceback.format_exc())
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(validated_qa_pairs, f, indent=2)
                print(f"Saved progress after error to: {output_path}")
                time.sleep(30)  # Longer pause after error
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(traceback.format_exc())
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_qa_pairs, f, indent=2)
            print(f"Saved progress after unexpected error to: {output_path}")

def main():
    args = parse_args()
    process_qa_data(args)

if __name__ == "__main__":
    main()