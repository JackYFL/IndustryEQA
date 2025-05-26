import json

def update_reasoning_status(dataset_path, score_files, output_path, condition_func):
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Load all score files and create dictionaries for easier lookup
    score_dicts = []
    for path in score_files:
        with open(path, 'r') as f:
            scores = json.load(f)
        score_dicts.append({item['question_id']: item for item in scores})
    
    # Find common question IDs across all score files
    common_qids = set.intersection(*[set(sd.keys()) for sd in score_dicts])
    
    # Apply custom condition to determine perfect_ids
    perfect_ids = set()
    for qid in common_qids:
        items = [sd[qid] for sd in score_dicts]
        if condition_func(items):
            perfect_ids.add(qid)
    
    print(f"Found {len(perfect_ids)} questions meeting criteria")
    
    # Update dataset
    for item in dataset:
        if item.get('question_id') in perfect_ids:
            item['reasoning_status'] = "0"
        else:
            item['reasoning_status'] = "1"
    
    # Save to new file
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

# Example usage:
score_files = [
    './data/scores/scores-large_all_new_new-o4-mini-gpt-4o-mini.json',
    './data/scores/scores-large_all_new_new-gemini-2.0-flash-gpt-4o-mini.json',
    './data/scores/scores-large_all_new_new-gemini-2.5-flash-preview-04-17-gpt-4o-mini.json',
    './data/scores/scores-large_all_new_new-gpt-4o-gpt-4o-mini.json',
]

# Custom condition function that takes a list of score items for the same question_id
def my_condition(items):
    # return (items[0]['direct_score'] >= 4 and items[0]['reasoning_score'] >= 4 and
    #         items[1]['direct_score'] >= 4 and items[1]['reasoning_score'] >= 4 and
    #         items[2]['direct_score'] >= 5 and items[2]['reasoning_score'] >= 4)
    return (items[0]['reasoning_score'] >= 5 and
            items[1]['reasoning_score'] >= 5 and
            items[2]['reasoning_score'] >= 5 and
            items[3]['reasoning_score'] >= 5)
    
update_reasoning_status(
    './results/results/large_all_new_new.json',
    score_files,
    './results/results/large_all_new_newr.json',
    my_condition
)