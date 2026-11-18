'''
Refactor Experts to get a list of phrases.
'''
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
            "answer": item.get("answer")
        }

        transformed_data.append(transformed_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/eval/expert_data/kosinski_original.json"
    output_file = "../data/eval/expert_data/kosinski.json"

    transform_and_save(input_file, output_file)'''



import json
import string
import re
from datasets import load_dataset


def transform_and_save(input_path: str, output_path: str):
    ds = load_dataset(
        "json",
        data_files=input_path,
        split="train"
    )

    transformed_data = []
    letters = list(string.ascii_uppercase)

    for item in ds:
        story_phrases = [
            (phrase.strip() + ".") if not phrase.strip().endswith(".") else phrase.strip()
            for phrase in item["story"].split(". ")
            if phrase
        ]  

        options = item.get("options", [])

        labeled_options = [
            f"{letters[i]}. {opt.strip()}" for i, opt in enumerate(options)
        ]

        letter_to_option = {
            letters[i]: opt.strip() for i, opt in enumerate(options)
        }

        correct_answer_text = item.get("answer", "").strip()

        correct_letter = None
        for letter, text in letter_to_option.items():
            if text.lower() == correct_answer_text.lower():
                correct_letter = letter
                break

        transformed_item = {
            "story": story_phrases,             # ["Sentence.", ...]
            "question": item.get("question"),
            "options": labeled_options,         # ["A. chocolate", ...]
            "answer": correct_answer_text,      # "chocolate"
            "gold": correct_letter              # "A"
        }

        transformed_data.append(transformed_item)

    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/eval/expert_data/ullman_original.json"
    output_file = "../data/eval/expert_data/ullman.json"

    transform_and_save(input_file, output_file)