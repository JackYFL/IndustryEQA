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
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for processing")
    parser.add_argument("--batch-size", type=int, default=150, help="Number of items to process")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    return args

def validate_qa_with_gemini(
    client,
    video_file,
    question: str,
    direct_answer: str,
    reasoning_answer: str,
    qa_type: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    prompt = load_prompt("check")
    # Format the prompt with all required information
    prompt = prompt.format(
        question=question, 
        direct_answer=direct_answer,
        reasoning_answer=reasoning_answer,
        type=qa_type
    )    
    
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
            "remain": 0,
            "direct_answer_correct": 0, 
            "reasoning_answer_correct": 0,
            "suggested_direct_answer": "Failed to evaluate",
            "suggested_reasoning_answer": "Failed to evaluate"
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
    
    # Calculate the end index for this batch
    start_index = args.start_index
    end_index = min(start_index + args.batch_size, len(qa_data))
    batch_data = qa_data[start_index:end_index]
    
    print(f"Processing batch from index {start_index} to {end_index-1} (total: {len(batch_data)} items)")
    
    # Create a unique output file name based on the start index
    output_filename = f"large_validated_results_new_{start_index}_{end_index-1}.json"
    output_path = args.output_directory / output_filename
    
    validated_qa_pairs = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                validated_qa_pairs = json.load(f)
            print(f"Loaded {len(validated_qa_pairs)} existing validated pairs from {output_filename}")
        except (json.JSONDecodeError, FileNotFoundError):
            validated_qa_pairs = []
    
    processed_ids = {qa["id"] for qa in validated_qa_pairs if "id" in qa}
    
    client = genai.Client(api_key=api_keys[0])
    
    uploaded_videos = {}
    
    for index, qa_pair in enumerate(batch_data):
        qa_id = qa_pair.get("id")
        question = qa_pair.get("question", "")
        direct_answer = qa_pair.get("direct_answer", "") 
        reasoning_answer = qa_pair.get("reasoning_answer", "")
        qa_type = qa_pair.get("type", "")
        video_path = qa_pair.get("path", "")
        
        absolute_index = start_index + index
        
        # Skip if missing essential fields
        if not question or not direct_answer or not reasoning_answer or not video_path or not qa_type:
            print(f"Skipping incomplete entry (ID: {qa_id})")
            continue
            
        if qa_id in processed_ids:
            print(f"Skipping already processed question ID {qa_id}: {question[:30]}...")
            continue
            
        print(f"Processing question {index + 1}/{len(batch_data)} " 
              f"(absolute: {absolute_index + 1}/{len(qa_data)}) "
              f"(ID: {qa_id}): {question[:30]}...")
        
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
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(validated_qa_pairs, f, indent=2)
                    print(f"Saved progress after error to: {output_path}")
                    time.sleep(10)
                    continue
            
            try:
                validation_result = validate_qa_with_gemini(
                    client,
                    video_file,
                    question=question,
                    direct_answer=direct_answer,
                    reasoning_answer=reasoning_answer,
                    qa_type=qa_type,
                    model=args.model,
                )
                
                validated_qa = {**qa_pair, **validation_result}
                validated_qa_pairs.append(validated_qa)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(validated_qa_pairs, f, indent=2)
                print(f"Saved progress to: {output_path}")
                
                processed_ids.add(qa_id)
                time.sleep(10)
                
            except Exception as e:
                print(f"Error validating question: {e}")
                print(traceback.format_exc())
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(validated_qa_pairs, f, indent=2)
                print(f"Saved progress after error to: {output_path}")
                time.sleep(10)
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(traceback.format_exc())
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_qa_pairs, f, indent=2)
            print(f"Saved progress after error to: {output_path}")
            time.sleep(10)

def main():
    args = parse_args()
    process_qa_data(args)

if __name__ == "__main__":
    main()