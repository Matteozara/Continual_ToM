import json

# Define your paths here
file_paths = [
    #'./data/eval/ToMi/train.json',
    './data/eval/ToMi/val.json',
    './data/eval/ToMi/test.json'
]

output_filename = 'ToMi_reality.json'
unified_data = []

for path in file_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Filter for the "reality" type
            # Using .get() ensures it won't crash if 'type' is missing in an object
            reality_entries = [item for item in data if item.get("type") == "reality"]
            
            unified_data.extend(reality_entries)
            print(f"Successfully pulled {len(reality_entries)} entries from: {path}")
            
    except FileNotFoundError:
        print(f"Skipping: File not found at {path}")
    except json.JSONDecodeError:
        print(f"Skipping: {path} is not a valid JSON file.")

# Save the combined list
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(unified_data, f, indent=4)

print(f"\nDone! Created {output_filename} with {len(unified_data)} total entries.")