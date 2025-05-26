import json

# Load score files
# with open('./data/scores/scores-small_all-gemini-2.0-flash-gemini-2.0-flash.json', 'r') as f1:
#     scores1 = json.load(f1)
# with open('./data/scores/scores-small_all-gemini-2.0-flash-2-gemini-2.0-flash.json', 'r') as f2:
#     scores2 = json.load(f2)

with open('./data/scores/scores-large_all-gemini-2.0-flash-gemini-2.0-flash.json', 'r') as f1:
    scores1 = json.load(f1)
with open('./data/scores/scores-large_all-gemini-2.0-flash-2-gemini-2.0-flash.json', 'r') as f2:
    scores2 = json.load(f2)
with open('./data/scores/scores-large_all_new-gemini-2.0-flash-blind-gpt-4o-mini.json', 'r') as f2:
    scores3 = json.load(f2) 
# Create dictionaries for easier lookup
scores_dict1 = {item['question_id']: item for item in scores1}
scores_dict2 = {item['question_id']: item for item in scores2}
scores_dict3 = {item['question_id']: item for item in scores3}

# Find question IDs with all scores of 5
perfect_ids = set()
for qid in set(scores_dict1.keys()) & set(scores_dict2.keys()) & set(scores_dict3.keys()):
    item1 = scores_dict1[qid]
    item2 = scores_dict2[qid]
    item3 = scores_dict3[qid]
    if (item1['direct_score'] >= 4 and item1['reasoning_score'] >= 4 and
            item2['direct_score'] >= 4 and item2['reasoning_score'] >= 4 and
            item3['direct_score'] >= 5 and item3['reasoning_score'] >= 4):
        perfect_ids.add(qid)

print(len(perfect_ids))
print(sorted(list(perfect_ids)))
# Load original data
# with open('./results/results/large_all.json', 'r', encoding='utf-8') as f3:
#     original_data = json.load(f3)

# # Separate data into matching and non-matching
# matching = []
# non_matching = []

# for item in original_data:
#     if item['question_id'] in perfect_ids:
#         matching.append(item)
#     else:
#         non_matching.append(item)

# # Write output files
# with open('./data/easy/large_easy_data.json', 'w') as out1:
#     json.dump(matching, out1, indent=4)
# with open('./data/easy/large_difficult_data.json', 'w') as out2:
#     json.dump(non_matching, out2, indent=4)

# print(f"Found {len(matching)} perfect score records and {len(non_matching)} other records")