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
    parser.add_argument("--input-json", type=Path, default="data/results/combined_qa_pairs.json", help="input JSON file with QA pairs")
    parser.add_argument("--start-index", type=int, default=0, help="input JSON file with QA pairs")
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
    # print(response.text)
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

def process_qa_data(args, api_keys):
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading input JSON file: {e}")
        return
    
    output_path = args.output_directory / "validated_results.json"
    
    validated_qa_pairs = []
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                validated_qa_pairs = json.load(f)
            print(f"Loaded {len(validated_qa_pairs)} existing validated pairs")
        except (json.JSONDecodeError, FileNotFoundError):
            validated_qa_pairs = []
    
    processed_ids = {qa["id"] for qa in validated_qa_pairs if "id" in qa}
    
    current_api_index = 0
    client = genai.Client(api_key=api_keys[current_api_index])
    
    uploaded_videos = {}
    
    for index, qa_pair in enumerate(qa_data):
        qa_id = qa_pair.get("id")
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        video_path = qa_pair.get("path", "")
        
        if not question or not answer or not video_path:
            continue
            
        if qa_id in processed_ids:
            print(f"Skipping already processed question ID {qa_id}: {question[:30]}...")
            continue
            
        print(f"Processing question {index + 1}/{len(qa_data)} (ID: {qa_id}): {question[:30]}...")
        
        try:
            if video_path in uploaded_videos:
                video_file = uploaded_videos[video_path]
                print(f"Using existing uploaded video: {video_path}")
            else:
                print(f"Uploading video: {video_path}")
                
                # Try with current API key
                try:
                    video_file = client.files.upload(file=video_path)
                except Exception as e:
                    if current_api_index < len(api_keys) - 1:
                        current_api_index += 1
                        print(f"Switching to next API key: {current_api_index}")
                        client = genai.Client(api_key=api_keys[current_api_index])
                        video_file = client.files.upload(file=video_path)
                    else:
                        print("All API keys exhausted")
                        raise e
                
                print(f"Completed upload: {video_file.uri}")
                
                while video_file.state.name == "PROCESSING":
                    print('.', end='')
                    time.sleep(1)
                    video_file = client.files.get(name=video_file.name)

                if video_file.state.name == "FAILED":
                    raise ValueError(f"Video processing failed: {video_file.state.name}")
                print('Video processing done')
                
                uploaded_videos[video_path] = video_file
            
            # Try validation with current API key
            try:
                validation_result = validate_qa_with_gemini(
                    client,
                    video_file,
                    question=question,
                    answer=answer,
                    model=args.model,
                )
            except Exception as e:
                if current_api_index < len(api_keys) - 1:
                    current_api_index += 1
                    print(f"Switching to next API key: {current_api_index}")
                    client = genai.Client(api_key=api_keys[current_api_index])
                    validation_result = validate_qa_with_gemini(
                        client,
                        video_file,
                        question=question,
                        answer=answer,
                        model=args.model,
                    )
                else:
                    print("All API keys exhausted")
                    raise e
            
            validated_qa = {**qa_pair, **validation_result}
            validated_qa_pairs.append(validated_qa)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_qa_pairs, f, indent=2)
            print(f"Saved progress to: {output_path}")
            
            processed_ids.add(qa_id)
            time.sleep(10)
            
        except Exception as e:
            print(f"Error processing question: {e}")
            print(traceback.format_exc())
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validated_qa_pairs, f, indent=2)
            print(f"Saved progress after error to: {output_path}")
            
            # Try switching to next API key if available
            if current_api_index < len(api_keys) - 1:
                current_api_index += 1
                print(f"Switching to next API key: {current_api_index}")
                client = genai.Client(api_key=api_keys[current_api_index])

def main():
    args = parse_args()
    
    # List of API keys - add your keys here
    api_keys = [
    ]
    
    assert len(api_keys) > 0 and api_keys[0] != "YOUR_API_KEY_1", "Please add real API keys to the list"
    
    process_qa_data(args, api_keys)

if __name__ == "__main__":
    main()