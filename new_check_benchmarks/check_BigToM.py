import json
from collections import defaultdict

def process_verify_and_count(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Detect Contradictions
    registry = {}
    blacklisted_pairs = set()
    
    for entry in data:
        narr_tuple = tuple(entry['narrative'])
        q_text = entry['question']
        pair_key = (narr_tuple, q_text)
        
        current_true = entry['true_answer']
        current_wrong = entry['wrong_answer']

        if pair_key in registry:
            existing = registry[pair_key]
            if existing['true'] != current_true or existing['wrong'] != current_wrong:
                blacklisted_pairs.add(pair_key)
        else:
            registry[pair_key] = {'true': current_true, 'wrong': current_wrong}

    # 2. Filter for Clean and Unique entries + Swap logic
    seen_pairs = set()
    final_output = []
    type_counts = defaultdict(int)

    for entry in data:
        narr_tuple = tuple(entry['narrative'])
        q_text = entry['question']
        pair_key = (narr_tuple, q_text)

        if pair_key not in blacklisted_pairs and pair_key not in seen_pairs:
            # --- START SWAP LOGIC ---
            if entry.get('q_type') == "1_percept_to_belief_true_belief":
                # Pythonic way to swap variables
                entry['true_answer'], entry['wrong_answer'] = entry['wrong_answer'], entry['true_answer']
            # --- END SWAP LOGIC ---

            final_output.append(entry)
            seen_pairs.add(pair_key)
            
            q_type = entry.get('q_type', 'unknown')
            type_counts[q_type] += 1

    # 3. Save the clean dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    # 4. Final Report
    print("--- Process Report ---")
    print(f"Total entries processed: {len(data)}")
    print(f"Unique clean entries saved: {len(final_output)}")
    print(f"Note: Swapped answers for '1_percept_to_belief_true_belief'")
    print("\n--- Samples per q_type (Cleaned) ---")
    
    for q_type in sorted(type_counts.keys()):
        print(f"{q_type}: {type_counts[q_type]}")

if __name__ == "__main__":
    process_verify_and_count("./../data/eval/BigToM/bigtom.json", "narrative_cleaned.json")