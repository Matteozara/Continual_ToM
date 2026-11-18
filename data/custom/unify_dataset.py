import json

def standardize_and_unify(file1_path, file2_path, output_path):
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    unified_list = []
    current_id = 1

    # Process Dataset 1
    for entry in data1:
        standardized = {
            "id": current_id,
            "category": entry.get("category"),  # category -> category
            "classification": entry.get("type"), # type -> classification
            "story": entry.get("story"),
            "memory_tracking": entry.get("memory_tracking"),
            "question": entry.get("question"),
            "answer": entry.get("answer")
        }
        unified_list.append(standardized)
        current_id += 1

    # Process Dataset 2
    for entry in data2:
        standardized = {
            "id": current_id,
            "category": entry.get("question_type"), # question_type -> category
            "classification": entry.get("category"), # category -> classification
            "story": entry.get("story"),
            "memory_tracking": entry.get("memory_tracking"),
            "question": entry.get("question"),
            "answer": entry.get("answer")
        }
        unified_list.append(standardized)
        current_id += 1

    # Save to file
    with open(output_path, 'w') as out_file:
        json.dump(unified_list, out_file, indent=4)

    print(f"Standardized {len(unified_list)} samples into {output_path}")


# Usage
standardize_and_unify('dataset1.json', 'dataset2.json', 'memory_dataset_OQ.json')