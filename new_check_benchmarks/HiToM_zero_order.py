import json
import random
from collections import defaultdict

def filter_hitom_order_zero(input_file, output_file, sample_limit=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Identify Inconsistencies (Contradictions)
    answer_registry = {}
    blacklisted_pairs = set()
    
    for entry in data:
        # Key on Story + Question
        pair_key = (tuple(entry['story']), entry['question'])
        current_ans = entry['answer']

        if pair_key in answer_registry:
            if answer_registry[pair_key] != current_ans:
                blacklisted_pairs.add(pair_key)
        else:
            answer_registry[pair_key] = current_ans

    # 2. Filter for Clean, Unique, and Order == 0
    seen_pairs = set()
    order_zero_data = []
    
    for entry in data:
        pair_key = (tuple(entry['story']), entry['question'])
        # Change 'question_order' to 'order_type' if your JSON uses that key instead
        current_order = entry.get('question_order') 

        # Filter logic:
        # - Must be order 0
        # - Must not be in the contradiction blacklist
        # - Must not be a duplicate we've already seen
        if (current_order == 0 and 
            pair_key not in blacklisted_pairs and 
            pair_key not in seen_pairs):
            
            order_zero_data.append(entry)
            seen_pairs.add(pair_key)

    # 3. Optional Sampling
    # If you want exactly 60 (or any specific number), we shuffle and slice
    if sample_limit and len(order_zero_data) > sample_limit:
        random.shuffle(order_zero_data)
        order_zero_data = order_zero_data[:sample_limit]
        print(f"Sampled {sample_limit} entries from the available pool.")
    else:
        print(f"Retaining all {len(order_zero_data)} valid order 0 entries.")

    # 4. Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(order_zero_data, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered data saved to {output_file}.")

if __name__ == "__main__":
    filter_hitom_order_zero("./../data/eval/HiToM/hitom.json", "hitom_order_0.json", sample_limit=None)