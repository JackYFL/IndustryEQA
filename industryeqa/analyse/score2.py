import json
import os
import sys
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

# Define ground truth file paths
small_gt_path = "./results/results/small_all_new_newr.json"
large_gt_path = "./results/results/large_all_new_newr.json"

# Parse command line arguments for dataset selection
dataset_filter = None
if len(sys.argv) > 1:
    if sys.argv[1].lower() == "small":
        dataset_filter = "small"
        print("Filtering to show only small dataset results")
    elif sys.argv[1].lower() == "large":
        dataset_filter = "large"
        print("Filtering to show only large dataset results")

# Load ground truth files
def load_ground_truth(file_path):
    try:
        with open(file_path, 'r') as f:
            return {item["question_id"]: item for item in json.load(f)}
    except FileNotFoundError:
        print(f"Warning: Ground truth file {file_path} not found. Using empty dictionary.")
        return {}

# 将1-5分数转换为百分比
def convert_to_percentage(score):
    # 先减1，再除以4，乘以100%
    return ((score - 1) / 4) * 100

# 从文件名中提取模型名称
def extract_model_name(file_name):
    # 假设文件名格式是: scores-{dataset}_{stuff}-{model}-{evaluator}.json
    base_name = os.path.basename(file_name)
    parts = base_name.split('-')
    
    # 忽略第一部分 'scores'
    parts = parts[1:]
    
    # 最后一部分是评估器 (如 gpt-4o-mini 或 gemini-2.0-flash)
    evaluator = parts[-1].replace('.json', '')
    
    # 第一部分是数据集标识 (small/large 等)
    dataset_part = parts[0]
    
    # 中间部分是模型名称
    model_parts = parts[1:-1]
    model_name = '-'.join(model_parts)
    
    return model_name, evaluator, dataset_part

# Dictionary to store all our results
results = {}

# Process each scores file
def process_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    
    # Extract the base filename without extension
    file_name = os.path.basename(file_path)
    
    # 从文件名中提取模型名称和评估器
    model_name, evaluator, dataset_part = extract_model_name(file_path)
    
    # Determine if this is small or large dataset
    is_small = "small" in dataset_part.lower()
    
    # Skip if we're filtering by dataset type and this doesn't match
    if dataset_filter:
        if (dataset_filter == "small" and not is_small) or (dataset_filter == "large" and is_small):
            return
    
    gt_data = small_gt if is_small else large_gt
    
    # 首先过滤出reasoning_status为1的数据项
    reasoning_status_1_items = []
    for item in data:
        q_id = item["question_id"]
        if q_id in gt_data:
            # For large dataset, treat all as reasoning_status=1 if not specified
            reasoning_status = gt_data[q_id].get("reasoning_status", "1")
            if reasoning_status == "1":
                reasoning_status_1_items.append(item)
    
    # Calculate statistics
    direct_scores = [item["direct_score"] for item in data]
    
    # 只使用reasoning_status=1的项目计算reasoning分数
    reasoning_scores = [item["reasoning_score"] for item in reasoning_status_1_items]
    
    # 转换为百分比
    direct_scores_pct = [convert_to_percentage(score) for score in direct_scores]
    reasoning_scores_pct = [convert_to_percentage(score) for score in reasoning_scores]
    
    # 1. Average direct score (as percentage)
    avg_direct = sum(direct_scores_pct) / len(direct_scores_pct) if direct_scores_pct else 0
    
    # 2. Average reasoning score for reasoning_status=1 (as percentage)
    avg_reasoning_status_1 = sum(reasoning_scores_pct) / len(reasoning_scores_pct) if reasoning_scores_pct else 0
    reasoning_status_1_count = len(reasoning_scores)
    
    # 3. Distribution of direct scores (1-5 range)
    direct_score_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for score in direct_scores:
        direct_score_dist[score] = direct_score_dist.get(score, 0) + 1
    
    # 4. Scores by question type
    type_scores = defaultdict(lambda: {"direct": [], "reasoning": []})
    
    # 计算每种类型的直接分数
    for item in data:
        q_id = item["question_id"]
        if q_id in gt_data:
            q_type = gt_data[q_id].get("type", "Unknown")
            type_scores[q_type]["direct"].append(convert_to_percentage(item["direct_score"]))
    
    # 只为reasoning_status=1的项目计算reasoning分数
    for item in reasoning_status_1_items:
        q_id = item["question_id"]
        if q_id in gt_data:
            q_type = gt_data[q_id].get("type", "Unknown")
            type_scores[q_type]["reasoning"].append(convert_to_percentage(item["reasoning_score"]))
    
    type_avg_scores = {}
    for q_type, scores in type_scores.items():
        type_avg_scores[q_type] = {
            "avg_direct": sum(scores["direct"]) / len(scores["direct"]) if scores["direct"] else 0,
            "avg_reasoning": sum(scores["reasoning"]) / len(scores["reasoning"]) if scores["reasoning"] else 0,
            "count": len(scores["direct"])
        }
    
    # 5. Scores by human/no_human
    human_direct_scores = []
    no_human_direct_scores = []
    human_reasoning_scores = []
    no_human_reasoning_scores = []
    
    # 为所有项目计算human/no_human的直接分数
    for item in data:
        q_id = item["question_id"]
        if q_id in gt_data:
            path = gt_data[q_id].get("path", "")
            if "no_human" in path:
                no_human_direct_scores.append(convert_to_percentage(item["direct_score"]))
            elif "human" in path and "no_human" not in path:
                human_direct_scores.append(convert_to_percentage(item["direct_score"]))
    
    # 只为reasoning_status=1的项目计算human/no_human的reasoning分数
    for item in reasoning_status_1_items:
        q_id = item["question_id"]
        if q_id in gt_data:
            path = gt_data[q_id].get("path", "")
            if "no_human" in path:
                no_human_reasoning_scores.append(convert_to_percentage(item["reasoning_score"]))
            elif "human" in path and "no_human" not in path:
                human_reasoning_scores.append(convert_to_percentage(item["reasoning_score"]))
    
    human_avg = {
        "avg_direct": sum(human_direct_scores) / len(human_direct_scores) if human_direct_scores else 0,
        "avg_reasoning": sum(human_reasoning_scores) / len(human_reasoning_scores) if human_reasoning_scores else 0,
        "count": len(human_direct_scores)
    }
    
    no_human_avg = {
        "avg_direct": sum(no_human_direct_scores) / len(no_human_direct_scores) if no_human_direct_scores else 0,
        "avg_reasoning": sum(no_human_reasoning_scores) / len(no_human_reasoning_scores) if no_human_reasoning_scores else 0,
        "count": len(no_human_direct_scores)
    }
    
    # Store all results
    results[file_name] = {
        "avg_direct": avg_direct,
        "avg_reasoning_status_1": avg_reasoning_status_1,
        "reasoning_status_1_count": reasoning_status_1_count,
        "direct_score_dist": direct_score_dist,
        "type_avg_scores": type_avg_scores,
        "human_avg": human_avg,
        "no_human_avg": no_human_avg,
        "total_samples": len(data),
        "model": model_name,  # 使用提取出的模型名称
        "evaluator": evaluator,  # 添加评估器信息
        "is_small": is_small
    }

# Print detailed analysis for a single file in tabular format
def print_file_analysis(file_name, data):
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR: {file_name}")
    print(f"{'='*80}")
    
    # Basic info table
    basic_info = pd.DataFrame([{
        "Model": data['model'],
        "Evaluator": data['evaluator'],
        "Dataset": 'Small' if data['is_small'] else 'Large',
        "Total Samples": data['total_samples'],
        "Avg Direct Score": f"{data['avg_direct']:.2f}%",
        "Avg Reasoning Score": f"{data['avg_reasoning_status_1']:.2f}%",
        "Reasoning Status=1 Count": data['reasoning_status_1_count']
    }])
    
    print("\nBASIC INFORMATION:")
    print(tabulate(basic_info, headers='keys', tablefmt='grid', showindex=False))
    
    # Direct score distribution table
    dist_data = []
    for score, count in sorted(data['direct_score_dist'].items()):
        percentage = (count / data['total_samples']) * 100 if data['total_samples'] > 0 else 0
        score_pct = convert_to_percentage(score)
        dist_data.append({
            "Score": score,
            "Percentage Value": f"{score_pct:.2f}%",
            "Count": count,
            "Distribution": f"{percentage:.1f}%"
        })
    
    dist_df = pd.DataFrame(dist_data)
    print("\nDIRECT SCORE DISTRIBUTION:")
    print(tabulate(dist_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Question type scores table
    type_data = []
    for q_type, scores in sorted(data['type_avg_scores'].items()):
        type_data.append({
            "Question Type": q_type,
            "Sample Count": scores['count'],
            "Avg Direct Score": f"{scores['avg_direct']:.2f}%",
            "Avg Reasoning Score": f"{scores['avg_reasoning']:.2f}%"
        })
    
    type_df = pd.DataFrame(type_data)
    print("\nSCORES BY QUESTION TYPE:")
    print(tabulate(type_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Human/No-human paths table
    human_data = [
        {
            "Path Type": "Human",
            "Sample Count": data['human_avg']['count'],
            "Avg Direct Score": f"{data['human_avg']['avg_direct']:.2f}%",
            "Avg Reasoning Score": f"{data['human_avg']['avg_reasoning']:.2f}%"
        },
        {
            "Path Type": "No Human",
            "Sample Count": data['no_human_avg']['count'],
            "Avg Direct Score": f"{data['no_human_avg']['avg_direct']:.2f}%",
            "Avg Reasoning Score": f"{data['no_human_avg']['avg_reasoning']:.2f}%"
        }
    ]
    
    human_df = pd.DataFrame(human_data)
    print("\nSCORES BY HUMAN/NO_HUMAN PATH:")
    print(tabulate(human_df, headers='keys', tablefmt='grid', showindex=False))

# Assuming the code is running in the directory where the score files are located
try:
    # Load ground truth files
    print(f"Loading ground truth files from: {small_gt_path} and {large_gt_path}")
    small_gt = load_ground_truth(small_gt_path)
    large_gt = load_ground_truth(large_gt_path)
    
    print(f"Loaded {len(small_gt)} small samples and {len(large_gt)} large samples ground truth data")
    
    folder_path = "./data/scores/"
    score_files = [f for f in os.listdir(folder_path) if f.startswith("scores-") and f.endswith(".json")]
    
    score_files = [folder_path + f for f in score_files]
    print(f"Found {len(score_files)} score files")
    
    if not score_files:
        print("No score files found. Please make sure file names start with 'scores-' and end with '.json'")
    else:
        # Process each file
        processed_count = 0
        for file in score_files:
            process_file(file)
            if file in results:
                processed_count += 1
        
        if not results:
            print("No results were processed. Check if files match the dataset filter.")
            sys.exit(1)
        
        # Print summary results
        print(f"\nProcessed {processed_count} files")
        
        # Create a summary dataframe
        summary_data = []
        for file_name, data in results.items():
            dataset_type = "Small" if data["is_small"] else "Large"
            summary_data.append({
                "File": file_name,
                "Dataset": dataset_type,
                "Model": data["model"],
                "Evaluator": data["evaluator"],
                "Avg Direct Score": f"{data['avg_direct']:.2f}%",
                "Avg Reasoning Score": f"{data['avg_reasoning_status_1']:.2f}%",
                "Total Samples": data["total_samples"],
                "Reasoning Status=1 Count": data["reasoning_status_1_count"]  # 添加这个字段
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nSUMMARY OF AVERAGE SCORES:")
        print(tabulate(summary_df.sort_values(["Dataset", "Model"]), headers='keys', tablefmt='grid', showindex=False))
        
        # Print detailed analysis for each file
        for file_name, data in sorted(results.items()):
            print_file_analysis(file_name, data)
        
        # Compare models if multiple models are present
        models = {}
        for file_name, data in results.items():
            model_name = data["model"]
            dataset_type = "small" if data["is_small"] else "large"
            
            if model_name not in models:
                models[model_name] = {"small": [], "large": []}
            
            models[model_name][dataset_type].append({
                "file": file_name,
                "avg_direct": data["avg_direct"],
                "avg_reasoning": data["avg_reasoning_status_1"],
                "evaluator": data["evaluator"],
                "reasoning_status_1_count": data["reasoning_status_1_count"]  # 添加这个字段
            })
        
        if len(models) > 1:
            print(f"\n{'='*80}")
            print("MODEL COMPARISON")
            print(f"{'='*80}")
            
            model_comparison = []
            for model_name, datasets in sorted(models.items()):
                for dataset_type in ["small", "large"]:
                    if datasets[dataset_type]:
                        avg_direct = sum(item["avg_direct"] for item in datasets[dataset_type]) / len(datasets[dataset_type])
                        avg_reasoning = sum(item["avg_reasoning"] for item in datasets[dataset_type]) / len(datasets[dataset_type])
                        total_reasoning_count = sum(item["reasoning_status_1_count"] for item in datasets[dataset_type])
                        
                        model_comparison.append({
                            "Model": model_name,
                            "Dataset": dataset_type.capitalize(),
                            "Avg Direct Score": f"{avg_direct:.2f}%",
                            "Avg Reasoning Score": f"{avg_reasoning:.2f}%",
                            "File Count": len(datasets[dataset_type]),
                            "Total Reasoning Status=1": total_reasoning_count  # 添加这个字段
                        })
            
            if model_comparison:
                comparison_df = pd.DataFrame(model_comparison)
                print("\nMODEL PERFORMANCE COMPARISON:")
                print(tabulate(comparison_df.sort_values(["Dataset", "Avg Direct Score"], ascending=[True, False]), 
                               headers='keys', tablefmt='grid', showindex=False))
                               
        # 按评估器比较模型
        print(f"\n{'='*80}")
        print("MODEL COMPARISON BY EVALUATOR")
        print(f"{'='*80}")
        
        evaluator_comparison = []
        
        for file_name, data in results.items():
            model_name = data["model"]
            evaluator = data["evaluator"]
            dataset_type = "Small" if data["is_small"] else "Large"
            
            evaluator_comparison.append({
                "Model": model_name,
                "Evaluator": evaluator,
                "Dataset": dataset_type,
                "Avg Direct Score": f"{data['avg_direct']:.2f}%",
                "Avg Reasoning Score": f"{data['avg_reasoning_status_1']:.2f}%",
                "Reasoning Status=1 Count": data["reasoning_status_1_count"]  # 添加这个字段
            })
        
        if evaluator_comparison:
            comparison_df = pd.DataFrame(evaluator_comparison)
            print("\nMODEL PERFORMANCE BY EVALUATOR:")
            print(tabulate(comparison_df.sort_values(["Dataset", "Evaluator", "Avg Direct Score"], 
                           ascending=[True, True, False]), headers='keys', tablefmt='grid', showindex=False))
        
except Exception as e:
    print(f"Error in processing: {e}")
    import traceback
    traceback.print_exc()