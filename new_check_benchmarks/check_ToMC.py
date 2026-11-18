import json
from collections import defaultdict

def process_and_verify_tomc(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Identify Contradictions (Same Story + Question = Different Answers)
    answer_registry = {}
    blacklisted_pairs = set()
    
    for entry in data:
        # CONVERSION: Turn list into tuple to make it "hashable" (usable as a key)
        story_val = tuple(entry['story']) if isinstance(entry['story'], list) else entry['story']
        q_val = tuple(entry['question']) if isinstance(entry['question'], list) else entry['question']
        
        pair_key = (story_val, q_val)
        current_ans = entry['answer']

        if pair_key in answer_registry:
            # If the same story/question has a different answer elsewhere, it's a contradiction
            if answer_registry[pair_key] != current_ans:
                blacklisted_pairs.add(pair_key)
        else:
            answer_registry[pair_key] = current_ans

    # 2. Filter for Clean and Unique entries
    seen_pairs = set()
    final_output = []
    type_counts = defaultdict(int)

    for entry in data:
        story_val = tuple(entry['story']) if isinstance(entry['story'], list) else entry['story']
        q_val = tuple(entry['question']) if isinstance(entry['question'], list) else entry['question']
        
        pair_key = (story_val, q_val)

        # Logic: 
        # - Keep if it's NOT in the blacklist (contradiction free)
        # - Keep if we haven't already added this exact pair (duplicate free)
        if pair_key not in blacklisted_pairs and pair_key not in seen_pairs:
            final_output.append(entry)
            seen_pairs.add(pair_key)
            
            # Tracking distribution
            q_type = entry.get('question_type', 'unknown')
            type_counts[q_type] += 1

    # 3. Save the clean dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    # 4. Final Summary
    print("--- Process Report ---")
    print(f"Total entries processed: {len(data)}")
    print(f"Contradictory pairs excluded: {len(blacklisted_pairs)}")
    print(f"Unique clean entries saved: {len(final_output)}")
    print("\n--- Samples per question_type (Cleaned) ---")
    
    for q_type in sorted(type_counts.keys()):
        print(f"{q_type}: {type_counts[q_type]}")

if __name__ == "__main__":
    process_and_verify_tomc("./../data/eval/ToMChallenges/Smarties.json", "Smarties.json")