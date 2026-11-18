'''
Refactor ToMChallenges to get a list of phrases.
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
            for phrase in item["story"].split(". ")
            if phrase
        ]        

        transformed_item = {
            "story": story_phrases,
            "question": item.get("question"),
            "answer": item.get("answer"),
            "belief": item.get("belief"),
            "question_type": item.get("question_type"),
            "category": item.get("category"),
        }

        transformed_data.append(transformed_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/eval/ToMChallenges/Sally-Anne_original.json"
    output_file = "../data/eval/ToMChallenges/Sally-Anne.json"

    transform_and_save(input_file, output_file)



'''import json
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
            for phrase in item["story"].split(". ")
            if phrase
        ]        

        transformed_item = {
            "story": story_phrases,
            "question": item.get("question"),
            "answer": item.get("answer"),
            "question_type": item.get("question_type")
        }

        transformed_data.append(transformed_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/eval/ToMChallenges/Smarties_original.json"
    output_file = "../data/eval/ToMChallenges/Smarties.json"

    transform_and_save(input_file, output_file)'''