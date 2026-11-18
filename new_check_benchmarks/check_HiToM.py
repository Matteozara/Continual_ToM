import json
import random
from collections import defaultdict

def clean_sample_unique_hitom(input_file, output_file, samples_per_type=60):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Identify Inconsistencies
    answer_registry = {}
    blacklisted_pairs = set()
    
    for entry in data:
        # Create a unique key for the story + question
        pair_key = (tuple(entry['story']), entry['question'])
        current_ans = entry['answer']

        if pair_key in answer_registry:
            if answer_registry[pair_key] != current_ans:
                blacklisted_pairs.add(pair_key)
        else:
            answer_registry[pair_key] = current_ans

    # 2. Filter for Clean AND Unique pairs
    seen_pairs = set()
    clean_data = []
    
    for entry in data:
        pair_key = (tuple(entry['story']), entry['question'])
        
        # Only add if:
        # 1. It's not blacklisted (no contradiction)
        # 2. We haven't added this specific story/question pair yet (no duplicates)
        if pair_key not in blacklisted_pairs and pair_key not in seen_pairs:
            clean_data.append(entry)
            seen_pairs.add(pair_key)

    # 3. Group and Sample
    grouped_by_type = defaultdict(list)
    for entry in clean_data:
        o_type = entry.get('question_order') 
        grouped_by_type[o_type].append(entry)

    final_sampled_list = []
    for o_type, entries in grouped_by_type.items():
        random.shuffle(entries)
        selection = entries[:samples_per_type]
        final_sampled_list.extend(selection)
        print(f"Type {o_type}: {len(entries)} unique clean samples available, selected {len(selection)}")

    random.shuffle(final_sampled_list)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_sampled_list, f, indent=4, ensure_ascii=False)
    
    print(f"\nSuccess! Saved {len(final_sampled_list)} unique samples to {output_file}")

if __name__ == "__main__":
    clean_sample_unique_hitom("./../data/eval/HiToM/hitom.json", 'hitom_cleaned.json')