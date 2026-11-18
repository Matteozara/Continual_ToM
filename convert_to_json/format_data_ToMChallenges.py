import json
import csv

# Paths to your files
csv_file = "data/ToMChallenges_data/Smarties_new_all.csv"

output_file = "data/ToMChallenges_data/Smarties_new_al.json"

stories_json = []

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:

        # skip rows without story or question
        if not row["prompt"] or not row["question"]:
            continue

        qtype = row["question_type"].strip()

        # remove question_type if it is "memory"
        if qtype.lower() == "memory":
            qtype = None

        entry = {
            "story": row["prompt"].strip(),
            "question": row["question"].strip(),
            "answer": row["short_answer"].strip(),
            "question_type": qtype
        }

        stories_json.append(entry)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(stories_json, f, indent=4, ensure_ascii=False)

print(f"Saved {len(stories_json)} entries to {output_file}")