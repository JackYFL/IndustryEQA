import json
import os
import glob

def process_json_files(input_pattern, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_processed_data = []
    total_items = 0
    processed_files = 0
    
    # Find all files matching the pattern
    input_files = glob.glob(input_pattern)
    
    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_processed_data = []
            file_items = 0
            
            for item in data:
                if item.get('remain', 0) == 0:
                    continue
                
                if item.get('remain', 0) == 1:
                    # Create a new item with only the required fields
                    new_item = {
                        'id': item.get('id'),
                        'type': item.get('type'),
                        'question': item.get('question'),
                        'direct_answer': item.get('direct_answer'),
                        'reasoning_answer': item.get('reasoning_answer'),
                        'path': item.get('path')
                    }
                    
                    if item.get('direct_answer_correct', 1) == 0:
                        suggested = item.get('suggested_direct_answer', '')
                        if suggested and suggested != "Same as original":
                            new_item['direct_answer'] = suggested
                    
                    if item.get('reasoning_answer_correct', 1) == 0:
                        suggested = item.get('suggested_reasoning_answer', '')
                        if suggested and suggested != "Same as original":
                            new_item['reasoning_answer'] = suggested
                    
                    file_processed_data.append(new_item)
                    file_items += 1
            
            all_processed_data.extend(file_processed_data)
            total_items += file_items
            processed_files += 1
            
            print(f"Processed {file_items} items from {os.path.basename(input_file)}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
    
    # Save all processed data to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Total: Processed {total_items} items from {processed_files} files and saved to {output_path}")
    return total_items

if __name__ == "__main__":
    # Directory containing the JSON files
    base_dir = "./data/validated_results"
    
    # Process validated_results_*.json files
    # validated_pattern = os.path.join(base_dir, "validated_results_*.json")
    # validated_output = os.path.join(base_dir, "small_results.json")
    # validated_count = process_json_files(validated_pattern, validated_output)
    # print(f"Successfully processed {validated_count} items from validated results")
    
    # Process large_validated_results_*.json files
    large_pattern = os.path.join(base_dir, "large_validated_results_new_*.json")
    large_output = os.path.join(base_dir, "large_results.json")
    large_count = process_json_files(large_pattern, large_output)
    print(f"Successfully processed {large_count} items from large validated results")