'''
Refactor HiToM to get a list of phrases and a list of options.
Associate the right answer to the right letter
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
        choices_list = [choice.strip() for choice in item["choices"].split(",")]

        letter_to_option = {}
        for choice in choices_list:
            if ". " in choice:
                letter, text = choice.split(". ", 1)
                letter_to_option[letter.strip()] = text.strip()

        correct_answer_text = item.get("answer", "").strip()

        correct_letter = None
        for letter, text in letter_to_option.items():
            if text.strip().lower() == correct_answer_text.lower():
                correct_letter = letter
                break

        transformed_item = {
            "prompting_type": item.get("prompting_type"),
            "deception": item.get("deception"),
            "story_length": item.get("story_length"),
            "question_order": item.get("question_order"),
            "sample_id": item.get("sample_id"),
            "story": story_phrases, #["phrase1", ...]
            "question": item.get("question"),
            "options": choices_list,   #["A. red_basket", ...]
            "answer": correct_answer_text,
            "gold": correct_letter   # "A"or "B", ...
        }

        transformed_data.append(transformed_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    input_file = "../data/eval/HiToM/hitom_original.json"
    output_file = "../data/eval/HiToM/hitom.json"

    transform_and_save(input_file, output_file)