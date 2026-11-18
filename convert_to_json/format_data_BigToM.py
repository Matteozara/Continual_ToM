import json
import csv

# Paths to your files
csv_file = "data/bigtom/bigtom.csv"
#output_file = "data/bigtom/test.json"

output_file = "data/bigtom/bigtom.json"

stories_json = []

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    for row in reader:
        if len(row) < 10:  # skip incomplete rows
            continue

        # The story sentences are all columns before the first question
        # Assuming questions start after the 5th column (adjust if necessary)
        # From your example, it seems:
        # Columns 0-4 → story events
        # Columns 5-7 → questions
        # Columns 8-10 → answers
        # Last column → category/type

        story_sentences = row[0:5]  # first 5 columns are story
        questions = row[5:8]        # next 3 columns are questions
        answers = row[8:11]         # next 3 columns are answers
        category = row[-2]           # second to last column as category
        q_type = row[-3]             # third to last column as type (auto/user)

        # Combine story sentences into one text
        story_text = " ".join(sentence.strip() for sentence in story_sentences)

        # Create a structured JSON entry
        stories_json.append({
            "concept": q_type,
            "story": story_text,
            "questions": questions,
            "answers": answers
        })

# Save to JSON
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(stories_json, f, indent=4, ensure_ascii=False)

print(f"Saved {len(stories_json)} stories to {output_file}")