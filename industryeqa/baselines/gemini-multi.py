import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import tqdm
from google import genai
import os
from openeqa.utils.prompt_utils import load_prompt
import time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="./results/results/small_all_new_new.json", help="path to EQA dataset")
    parser.add_argument("--models", type=str, nargs='+', default=["gemini-2.0-flash", "gemini-2.5-pro-preview-03-25"], 
                        help="models to test (provide two model names)")
    parser.add_argument("--output-directory", type=Path, default="./data/results", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    args = parser.parse_args()
    
    # 验证是否提供了两个模型
    if len(args.models) != 2:
        raise ValueError("Please provide exactly two model names using --models")
        
    args.output_directory.mkdir(parents=True, exist_ok=True)
    
    # 为每个模型创建输出路径
    args.output_paths = []
    for model in args.models:
        output_path = args.output_directory / (args.dataset.stem + "-{}.json".format(model))
        args.output_paths.append(output_path)
        
    return args

def parse_json_from_response(text: str):
    """Extract JSON from the model response."""
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
    except json.JSONDecodeError:
        return {"direct_answer": "", "reasoning_answer": f"JSON parse error in: {text}"}


def ask_question_gemini_video(
    question,
    video_file,
    client,
    model: str,
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
                print(f"API call failed with model {model}: {str(e)}. Retrying in {10} seconds... (Attempt {retries}/{2})")
                time.sleep(10)
            else:
                print(f"Failed after multiple retries with model {model}. Error: {str(e)}")
                return {
                    "direct_answer": "",
                    "reasoning_answer": f"API call failed after multiple retries: {str(e)}"
                }
                
                
def main(args: argparse.Namespace):
    assert "GOOGLE_API_KEY" in os.environ
    api_key = [
    ]
    dataset = json.load(args.dataset.open("r", encoding="utf-8"))
    print("found {:,} questions".format(len(dataset)))

    # 为每个模型加载现有结果
    all_results = []
    completed_ids = []
    for i, output_path in enumerate(args.output_paths):
        results = []
        if output_path.exists():
            results = json.load(output_path.open())
            # print(f"found {:,} existing results for model {args.models[i]}".format(len(results)))
            completed_ids.append({item["question_id"] for item in results})
        else:
            completed_ids.append(set())
        all_results.append(results)

    added_path = []
    video_file_dict = {}
    client = genai.Client(api_key=api_key[0])
    
    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 100:
            break

        question_id = item["question_id"]
        question = item["question"]
        file_path = item["path"]
        
        if question_id in completed_ids[0]:
            continue  # skip existing
        
        # 上传视频文件（如果尚未上传）
        if file_path not in added_path:
            added_path.append(file_path)
            print(file_path)
            print("Uploading file...")
            video_file = client.files.upload(file=file_path)
            print(f"Completed upload: {video_file.uri}")

            # 检查文件是否准备好使用
            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(1)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            print('Done')
            video_file_dict[file_path] = video_file
        else:
            video_file = video_file_dict[file_path]
        
        # 对每个模型分别处理问题
        for i, model in enumerate(args.models):
            # 如果该问题已经处理过了，跳过
            if question_id in completed_ids[i]:
                continue
                
            print(f"Processing question {question_id} with model {model}")
            
            answer_json = ask_question_gemini_video(
                question=question,
                video_file=video_file,
                client=client,
                model=model,
            )

            all_results[i].append({
                "question_id": question_id,
                "question": question,
                "generated_direct_answer": answer_json.get("direct_answer", ""),
                "generated_reasoning_answer": answer_json.get("reasoning_answer", "")
            })

            json.dump(all_results[i], args.output_paths[i].open("w"), indent=2)
            
            time.sleep(1)
        
        # 每个问题处理完两个模型后的额外休息时间
        # time.sleep(5)

    # 最后保存所有结果
    for i, output_path in enumerate(args.output_paths):
        json.dump(all_results[i], output_path.open("w"), indent=2)
        # print(f"saving {:,} answers for model {args.models[i]}".format(len(all_results[i])))

if __name__ == "__main__":
    main(parse_args())