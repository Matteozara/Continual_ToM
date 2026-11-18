import json

txt_file = "../data/eval/ToMi/train.txt"
trace_file = "../data/eval/ToMi/train.trace"
output_file = "../data/eval/ToMi/train.json"

with open(trace_file, "r") as f:
    trace_lines = f.read().splitlines()

all_traces = []
for line in trace_lines:
    parts = line.strip().split(",")
    if len(parts) < 2:
        continue
    trace_type = parts[-2]
    category = parts[-1]
    all_traces.append((trace_type, category))

with open(txt_file, "r") as f:
    txt_lines = f.read().splitlines()

dataset = []
current_story = []
trace_index = 0

for line in txt_lines:
    if not line.strip():
        continue

    if "\t" in line:
        parts = line.strip().split("\t")

        if len(parts) >= 3:
            question_line = parts[0]
            question = " ".join(question_line.split()[1:]) if len(question_line.split()) > 1 else question_line
            answer = parts[1]
            choices_raw = parts[2]
        else:
            question, answer, choices_raw = "", "", ""

        if trace_index < len(all_traces):
            q_type, category = all_traces[trace_index]
        else:
            q_type, category = "unknown", "unknown"

        trace_index += 1

        if q_type == "memory":
            current_story = []
            continue

        story_text = " ".join(
            [s.split(maxsplit=1)[1] if len(s.split()) > 1 else s for s in current_story]
        )

        story_phrases = [
            (phrase.strip() + ".") if not phrase.strip().endswith(".") else phrase.strip()
            for phrase in story_text.split(". ")
            if phrase.strip()
        ]

        correct_answer_text = answer.strip()


        dataset.append({
            "type": q_type,
            "category": category,
            "story": story_phrases,
            "question": question,
            "answer": correct_answer_text,
        })

        current_story = []

    else:
        current_story.append(line.strip())

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"Saved {len(dataset)} stories to {output_file}")