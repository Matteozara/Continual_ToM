import json
import csv

# Paths to your files
csv_file = "data/expert_data/ullman.csv"

output_file = "data/expert_data/ullman.json"


stories_json = []

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        if len(row) < 4:  # skip incomplete rows
            continue
        
        story = row[0].strip()
        question = row[1].strip()
        # All possible answers
        answers = [row[2].strip(), row[3].strip()]
        # If there is a third column (uncertain guess), include it
        if len(row) > 4:
            answers.append(row[4].strip())
        
        stories_json.append({
            "story": story,
            "question": question,
            "options": answers,
            "answer": "",
        })

# Save to JSON
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(stories_json, f, indent=4, ensure_ascii=False)

print(f"Saved {len(stories_json)} stories to {output_file}")