'''
Refactor BigToM to get a list of phrases and couple question-answers.
'''
import json
from datasets import load_dataset

def transform_and_save(input_path: str, output_path: str):
    ds = load_dataset(
        "json",
        data_files=input_path,
        split="train"
    )

    transformed_data = []

    for item in ds:
        story_phrases = [
            (phrase.strip() + ".") if not phrase.strip().endswith(".") else phrase.strip()
            for phrase in item["narrative"].split(". ")
            if phrase
        ]        

        transformed_item = {
            "q_type": item.get("q_type"),
            "narrative": story_phrases,
            "question": item.get("question"),
            "true_answer": item.get("true_answer"),
            "wrong_answer": item.get("wrong_answer"),
        }

        transformed_data.append(transformed_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/eval/BigToM/bigtom_all.json"
    output_file = "../data/eval/BigToM/bigtom.json"

    transform_and_save(input_file, output_file)