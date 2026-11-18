import json
from collections import defaultdict

def process_and_verify_category_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Identify Contradictions
    # Key: (story_tuple, question_text) -> answer
    answer_registry = {}
    blacklisted_pairs = set()
    
    for entry in data:
        # Convert list to tuple to make it hashable for the dictionary key
        story_val = tuple(entry['story']) if isinstance(entry['story'], list) else entry['story']
        q_text = entry['question']
        pair_key = (story_val, q_text)
        current_ans = entry['answer']

        if pair_key in answer_registry:
            # If the same story and question have a different answer, it's a contradiction
            if answer_registry[pair_key] != current_ans:
                blacklisted_pairs.add(pair_key)
        else:
            answer_registry[pair_key] = current_ans

    # 2. Filter for Clean and Unique entries
    seen_pairs = set()
    final_output = []
    category_counts = defaultdict(int)

    for entry in data:
        story_val = tuple(entry['story']) if isinstance(entry['story'], list) else entry['story']
        q_text = entry['question']
        pair_key = (story_val, q_text)

        # Keep if not blacklisted and not a duplicate
        if pair_key not in blacklisted_pairs and pair_key not in seen_pairs:
            final_output.append(entry)
            seen_pairs.add(pair_key)
            
            # Tracking category stats
            cat = entry.get('category', 'unknown')
            category_counts[cat] += 1

    # 3. Save the clean dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    # 4. Final Report
    print("--- Process Report ---")
    print(f"Total entries processed: {len(data)}")
    print(f"Unique clean entries saved: {len(final_output)}")
    print(f"Contradictory pairs removed: {len(blacklisted_pairs)}")
    print("\n--- Samples per category (Cleaned) ---")
    
    for cat in sorted(category_counts.keys()):
        print(f"{cat}: {category_counts[cat]}")

if __name__ == "__main__":
    process_and_verify_category_data("./../data/eval/ToMi/train.json", "train.json")




'''
#to create also the small dataset to check manually the memory:
import json
import random
from collections import defaultdict

def process_and_verify_category_data(input_file, output_file, sample_output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Identify Contradictions
    answer_registry = {}
    blacklisted_pairs = set()
    
    for entry in data:
        story_val = tuple(entry['story']) if isinstance(entry['story'], list) else entry['story']
        q_text = entry['question']
        pair_key = (story_val, q_text)
        current_ans = entry['answer']

        if pair_key in answer_registry:
            if answer_registry[pair_key] != current_ans:
                blacklisted_pairs.add(pair_key)
        else:
            answer_registry[pair_key] = current_ans

    # 2. Filter for Clean and Unique entries
    seen_pairs = set()
    final_output = []
    # Grouping entries by category for sampling
    categories_map = defaultdict(list)

    for entry in data:
        story_val = tuple(entry['story']) if isinstance(entry['story'], list) else entry['story']
        q_text = entry['question']
        pair_key = (story_val, q_text)

        if pair_key not in blacklisted_pairs and pair_key not in seen_pairs:
            final_output.append(entry)
            seen_pairs.add(pair_key)
            
            cat = entry.get('category', 'unknown')
            categories_map[cat].append(entry)

    # 3. Select 5 Random Samples per Category
    random_samples = []
    for cat, entries in categories_map.items():
        # If there are fewer than 5, take them all; otherwise, pick 5
        num_to_sample = min(len(entries), 5)
        sampled_subset = random.sample(entries, num_to_sample)
        random_samples.extend(sampled_subset)

    # 4. Save the clean dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    # 5. Save the random samples
    with open(sample_output_file, 'w', encoding='utf-8') as f:
        json.dump(random_samples, f, indent=4, ensure_ascii=False)

    # 6. Final Report
    print("--- Process Report ---")
    print(f"Total entries processed: {len(data)}")
    print(f"Unique clean entries saved: {len(final_output)}")
    print(f"Contradictory pairs removed: {len(blacklisted_pairs)}")
    print(f"Random samples saved to: {sample_output_file}")
    print("\n--- Samples per category (Cleaned) ---")
    
    for cat in sorted(categories_map.keys()):
        print(f"{cat}: {len(categories_map[cat])}")

if __name__ == "__main__":
    # Added a third argument for the samples file
    process_and_verify_category_data(
        "./../data/eval/ToMi/train.json", 
        "train_cleaned.json", 
        "category_samples.json"
    )
'''