from datasets import load_dataset
from typing import Iterable, Dict



def load_HiToM(json_path: str) -> Iterable[Dict]:
    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        download_mode="force_redownload"
    )

    for item in ds:
        yield {
            #"prompting_type": item.get("prompting_type"),
            #"deception": item.get("deception"),
            #"story_length": item.get("story_length"),
            "question_order": item.get("question_order"),
            "sample_id": item.get("sample_id"),
            "story": item.get("story"),
            "question": item.get("question"),
            "options": item.get("options"),
            "answer": item.get("answer"),
            "gold": item.get("gold"),
        }




def load_BigToM(json_path: str) -> Iterable[Dict]:
    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        download_mode="force_redownload"
    )

    for item in ds:
        yield {
            "q_type": item.get("q_type"),
            "narrative": item.get("narrative"),
            "question": item.get("question"),
            "true_answer": item.get("true_answer"),
            "wrong_answer": item.get("wrong_answer"),
        }


def load_expert(json_path: str, expert_type: str) -> Iterable[Dict]:
    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        download_mode="force_redownload"
    )

    if expert_type in ["df", "k"]:
        for item in ds:
            story_sentences = [s.strip() for s in item["story"].split(". ") if s]

            yield {
                "story": story_sentences,
                "question": item.get("question"),
                "answer": item.get("answer")
            }
    else:
        for item in ds:
            story_sentences = [s.strip() for s in item["story"].split(". ") if s]

            yield {
                "story": story_sentences,
                "question": item.get("question"),
                "options": item.get("options"),
                "answer": item.get("answer")
            }


def load_ToMChallenges(json_path: str) -> Iterable[Dict]:
    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        download_mode="force_redownload"
    )

    
    for item in ds:
        yield {
            "story": item.get("story"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "question_type": item.get("question_type")
        }

def load_ToMi(json_path: str) -> Iterable[Dict]:
    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        download_mode="force_redownload"
    )

    
    for item in ds:
        yield {
            "category": item.get("category"),
            "story": item.get("story"),
            "question": item.get("question"),
            "answer": item.get("answer"),
        }

def load_Custom(json_path: str) -> Iterable[Dict]:
    ds = load_dataset(
        "json",
        data_files=json_path,
        split="train",
        download_mode="force_redownload"
    )

    
    for item in ds:
        yield {
            "id": item.get("id"),
            "category": item.get("category"),
            "story": item.get("story"),
            "memory_tracking": item.get("memory_tracking"),
            "question": item.get("question"),
            "answer": item.get("answer"),
        }
