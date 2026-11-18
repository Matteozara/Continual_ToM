'''
Not really better. I think the old one is better
'''

import os
import json
import spacy
import re
from tqdm import tqdm
from utils import set_seed

# 1. Activate GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FOLDER_OUTPUTS = "NER_Beliefs"
os.makedirs(FOLDER_OUTPUTS, exist_ok=True)

try:
    activated = spacy.prefer_gpu()
    print("✓ GPU activated." if activated else "ℹ Using CPU.")
except Exception as e:
    print(f"✗ GPU error: {e}")

# 2. Model Loading
MODEL_NAME = "en_core_web_lg"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    os.system(f"python -m spacy download {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)

def clean_text(text):
    if not text: return ""
    text = text.replace("{", "").replace("}", "").replace(";", ". ")
    return re.sub(r'\s+', ' ', text).strip()

GENERIC_NOUNS = {"way", "time", "thing", "day", "place", "kind",
                 "type", "lot", "bit", "fact", "idea", "noise",
                 "breath", "evening", "night", "morning", "village"}

def is_mentioned(entity, text):
    """Word-boundary safe substring check."""
    return bool(re.search(rf'\b{re.escape(entity)}\b', text))

def extract_core_from_story(story):
    """Extracts the ground-truth entity whitelist from the full story."""
    if isinstance(story, list):
        full_text = " ".join(story)
    else:
        full_text = story

    doc = nlp(clean_text(full_text))
    core = set()

    # Only PERSON and PRODUCT — no locations
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            core.add(ent.lemma_.lower())
        elif ent.label_ == "PRODUCT":
            core.add(ent.lemma_.lower())

    # Physical nouns that are syntactic subjects/objects
    for token in doc:
        if token.pos_ != "NOUN":
            continue
        if token.lemma_.lower() in GENERIC_NOUNS:
            continue
        if token.dep_ in {"nsubj", "nsubjpass", "dobj"}:
            if not any(anc.dep_ == "prep" for anc in token.ancestors):
                core.add(token.lemma_.lower())

    return core

def process_memories_optimized(data):
    all_memory_texts = [clean_text(log.get("new_memory", ""))
                        for entry in data for log in entry.get("memory_logs", [])]

    print(f"Analyzing {len(all_memory_texts)} memory steps...")
    doc_iter = iter(nlp.pipe(all_memory_texts, batch_size=256))

    results, traces = [], []
    m_total, m_wrong = 0, 0
    s_perfect, s_flawed = 0, 0

    for entry in tqdm(data, desc="Auditing Persistence"):
        story_id = entry.get("id")
        memory_logs = entry.get("memory_logs", [])

        core_entities = extract_core_from_story(entry.get("story", ""))

        story_trace = {
            "id": story_id,
            "core_entities_found": list(core_entities),
            "steps": [],
            "final_census_passed": False,
            "entities_dropped_prematurely": set()
        }

        # ← Fix: initialize to full core, not empty set
        previously_present = core_entities.copy()
        story_has_error = False

        for i, log in enumerate(memory_logs):
            try:
                current_doc = next(doc_iter)
            except StopIteration:
                break

            m_total += 1
            mem_content = current_doc.text.lower()

            # ← Fix: word-boundary matching
            currently_present = {e for e in core_entities if is_mentioned(e, mem_content)}
            dropped_this_step = previously_present - currently_present
            missing_from_core = core_entities - currently_present

            is_consistent = len(missing_from_core) == 0
            log["is_consistent"] = is_consistent
            log["missing_entities"] = list(missing_from_core)

            if not is_consistent:
                m_wrong += 1
                story_has_error = True
                story_trace["entities_dropped_prematurely"].update(dropped_this_step)

            story_trace["steps"].append({
                "phrase_number": log.get("phrase_number"),
                "dropped_this_step": list(dropped_this_step),
                "missing_from_core": list(missing_from_core)
            })

            previously_present = currently_present

            if i == len(memory_logs) - 1:
                story_trace["final_census_passed"] = is_consistent
                story_trace["missing_at_end"] = list(missing_from_core)

        if story_has_error:
            s_flawed += 1
        else:
            s_perfect += 1

        story_trace["entities_dropped_prematurely"] = list(
            story_trace["entities_dropped_prematurely"]
        )
        results.append(entry)
        traces.append(story_trace)

    # Stats Compilation
    total_stories = len(data)
    summary_stats = {
        "overall_results": {
            "total_stories": total_stories,
            "total_wrong_memories": m_wrong,
            "percentage_stories_perfect": round((s_perfect / total_stories * 100), 2) if total_stories > 0 else 0,
            "mean_errors_per_story": round(m_wrong / total_stories, 2) if total_stories > 0 else 0
        },
        "stories": traces
    }

    return results, summary_stats

def start_evaluation():
    set_seed(42)
    benchs = ["BigToM"] #["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
    models_gpu2 = ["gpt-20b"] #["gpt-20b","gemma2-27b","granite4-32b","llama3-70b", "qwen3.6-35b"]
    ebe_types = ["standard"] #["standard", "obj_location", "task_specific"]

    summary = {}
    for benchmark in benchs:
        for m in models_gpu2:
            for e in ebe_types:
                input_file = f"predictions/{benchmark}_{e}_EbE_{m}_raw.json"
                if os.path.exists(input_file):
                    print(f"Auditing Model: {m} | Bench: {benchmark}...")
                    with open(input_file, 'r') as f:
                        input_data = json.load(f)

                    enriched_data, trace_log = process_memories_optimized(input_data)

                    with open(f'{FOLDER_OUTPUTS}/{benchmark}_{e}_EbE_{m}_trace.json', 'w') as f:
                        json.dump(trace_log, f, indent=4)

                    summary[f"{m}_{benchmark}_{e}"] = trace_log["overall_results"]
    return summary

if __name__ == "__main__":
    final_summary = start_evaluation()
    with open(f'{FOLDER_OUTPUTS}/FINAL_SUMMARY.json', 'w') as f:
        json.dump(final_summary, f, indent=4)