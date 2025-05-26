import json
from collections import Counter

def count_json_types(file_path):
    types = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if 'type' in item:
                    types.append(item['type'])
        elif isinstance(data, dict) and 'type' in data:
            types.append(data['type'])
      
    
    type_counter = Counter(types)
    
    print(f"Fouond {len(type_counter)} types:")
    for type_name, count in type_counter.items():
        print(f"- {type_name}: {count}")
    
    return dict(type_counter)

if __name__ == "__main__":
    # count_json_types("./results/results/small_all_new_reasoning_status.json")
    count_json_types("./results/results/large_all_new_new.json")