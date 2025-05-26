import json

def reindex_json_file_ids(input_filepath, output_filepath):
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            json_data = json.load(f_in)
        
        if not isinstance(json_data, list):
            print(f"Error: JSON content in {input_filepath} is not a list.")
            return

        for new_id, item in enumerate(json_data, 1):
            if isinstance(item, dict):
                item['question_id'] = new_id
            
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(json_data, f_out, ensure_ascii=False, indent=4)
        print(f"Successfully re-indexed IDs from {input_filepath} and saved to {output_filepath}")
    except FileNotFoundError:
        print(f"Error: Input file {input_filepath} not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_filepath}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    input_file_path = "./results/results/industryeqa.json"
    output_file_path = "./results/results/industryeqa.json"
    reindex_json_file_ids(input_file_path, output_file_path)