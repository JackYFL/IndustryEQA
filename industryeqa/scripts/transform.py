import argparse
import json
import time
import re
from pathlib import Path
from typing import List, Optional
import tqdm
from google import genai
from openeqa.utils.prompt_utils import load_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="./data/validated_results/large_results_new_new.json")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-preview-04-17")
    parser.add_argument("--output-directory", type=Path, default="data/results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (args.dataset.stem + "-transformed-2.json")
    return args


class APIKeyRotator:
    def __init__(self, api_keys: List[str], max_retries: int = 3, retry_delay: int = 60):
        """
        Initialize the API key rotator.
        
        Args:
            api_keys: List of API keys to rotate through
            max_retries: Maximum number of retries per API key before switching
            retry_delay: Delay in seconds before retrying after an error
        """
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        
        self.api_keys = api_keys
        self.current_idx = 0
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.current_retries = 0
        
    def get_current_api_key(self) -> str:
        """Get the current API key."""
        return self.api_keys[self.current_idx]
    
    def next_api_key(self) -> str:
        """Rotate to the next API key and reset retry count."""
        self.current_idx = (self.current_idx + 1) % len(self.api_keys)
        self.current_retries = 0
        return self.get_current_api_key()
    
    def handle_error(self) -> str:
        """
        Handle API error by waiting and potentially rotating API keys.
        
        Returns:
            The API key to use for the next attempt
        """
        self.current_retries += 1
        
        if self.current_retries > self.max_retries:
            print(f"API key exhausted after {self.max_retries} retries. Switching to next API key.")
            return self.next_api_key()
        
        print(f"API error occurred. Waiting {self.retry_delay} seconds before retry {self.current_retries}/{self.max_retries}...")
        time.sleep(self.retry_delay)
        return self.get_current_api_key()
    
    def get_client(self) -> genai.Client:
        """Get a client using the current API key."""
        return genai.Client(api_key=self.get_current_api_key())


def transform_qa(api_rotator, video_file, item, model="gemini-2.0-flash"):
    prompt = load_prompt("transform")
    full_prompt = prompt.format(
        question=item["question"],
        direct_answer=item["direct_answer"],
        reasoning_answer=item["reasoning_answer"]
    )
    print("transforming")
    while True:
        try:
            client = api_rotator.get_client()
            print(f"Using API key: {api_rotator.get_current_api_key()[:20]}...")
            
            response = client.models.generate_content(
                model=model,
                contents=[video_file, full_prompt]
            )
            
            response_text = response.text
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group(1))
                else:
                    parsed_data = json.loads(response_text)
                
                # Ensure we're returning a dictionary, not a list
                if isinstance(parsed_data, list):
                    # Convert list to dictionary if needed
                    if parsed_data and all(isinstance(item, dict) for item in parsed_data):
                        # Merge all dictionaries in the list
                        result_dict = {}
                        for d in parsed_data:
                            result_dict.update(d)
                        return result_dict
                    else:
                        print(f"Unexpected format in response: {parsed_data}")
                        return {}
                elif isinstance(parsed_data, dict):
                    return parsed_data
                else:
                    print(f"Unexpected data type in response: {type(parsed_data)}")
                    return {}
            except Exception as e:
                print(f"Failed to parse response: {response_text[:100]}... Error: {str(e)}")
                return {}
                
        except Exception as e:
            print(f"Error in transform_qa: {str(e)}")
            api_rotator.handle_error()
def main(args):
    api_keys = [

    ]
    
    print(f"Operating with {len(api_keys)} API keys")
    api_rotator = APIKeyRotator(
        api_keys=api_keys, 
        max_retries=3,
        retry_delay=60
    )
    
    dataset = json.load(args.dataset.open())
    print(f"Found {len(dataset)} questions")

    results = []
    processed_ids = set()
    uploaded_videos = {}
    
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        processed_ids = {item.get("id") for item in results}
        print(f"Found {len(results)} existing results")

    batch_data = dataset[0:150]
    
    for idx, item in enumerate(batch_data):
        if args.dry_run and idx >= 8:
            break

        q_id = item.get("id")
        video_path = item.get("path")
        # print(video_path)
        # print(q_id)
        # print(processed_ids)
        if not video_path or q_id in processed_ids:
            # print("skip")
            continue
            
        print(f"Processing {idx+1}/{len(batch_data)}: {item['question'][:30]}...")
        
        try:
            if video_path in uploaded_videos:
                video_file = uploaded_videos[video_path]
            else:
                uploaded = False
                while not uploaded:
                    try:
                        client = api_rotator.get_client()
                        video_file = client.files.upload(file=video_path)
                        
                        while video_file.state.name == "PROCESSING":
                            print('.', end='')
                            time.sleep(1)
                            video_file = client.files.get(name=video_file.name)
                        
                        if video_file.state.name == "FAILED":
                            raise ValueError(f"Video processing failed")
                        
                        uploaded_videos[video_path] = video_file
                        uploaded = True
                        
                    except Exception as e:
                        print(f"Error uploading video: {str(e)}")
                        api_rotator.handle_error()
            
            transformed_data = transform_qa(
                api_rotator=api_rotator,
                video_file=video_file,
                item=item,
                model=args.model
            )
            
            result = {**item}
            if transformed_data:
                if isinstance(transformed_data, dict):
                    result.update(transformed_data)
                else:
                    print(f"Unexpected transformed_data format: {type(transformed_data)}")
            
            results.append(result)
            processed_ids.add(q_id)
            
            json.dump(results, args.output_path.open("w"), indent=2)
            
        except Exception as e:
            print(f"Error processing item {q_id}: {str(e)}")
            api_rotator.handle_error()
            
            if len(results) > 0:
                json.dump(results, args.output_path.open("w"), indent=2)
    
    print(f"Saved {len(results)} transformed QA pairs")


if __name__ == "__main__":
    main(parse_args())