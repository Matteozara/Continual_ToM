import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import spacy
import re
from tqdm import tqdm
from utils import set_seed

FOLDER_OUTPUTS = "NER_Beliefs_obj" #"NER_Beliefs_obj_gpu1" 
os.makedirs(FOLDER_OUTPUTS, exist_ok=True)

try:
    activated = spacy.prefer_gpu()
    print("✓ GPU activated." if activated else "ℹ Using CPU.")
except Exception as e:
    print(f"✗ GPU error: {e}")

MODEL_NAME = "en_core_web_lg"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    os.system(f"python -m spacy download {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)

def clean_text(text):
    text = text.replace("{", "").replace("}", "").replace(";", ". ")
    return re.sub(r'\s+', ' ', text).strip()

def extract_subjects(doc):
    subjects = set()
    
    for token in doc:
        #get nouns
        if token.pos_ == "PROPN":
            subjects.add(token.text.lower())
        
        if token.pos_ == "NOUN" and token.dep_ in ["nsubj", "nsubjpass", "dobj"]:
            subjects.add(token.text.lower())
            
    return subjects

def process_memories_optimized(data):
    all_texts = [clean_text(log.get("new_memory")) 
                 for entry in data for log in entry.get("memory_logs", [])]

    print(f"Analyzing {len(all_texts)} memory steps...")
    docs = list(nlp.pipe(all_texts, batch_size=256))
    doc_iter = iter(docs)
    
    results, traces = [], []
    
    m_total, m_wrong = 0, 0
    s_perfect, s_flawed = 0, 0
    total_subjects_lost = 0
    total_empty_memories = 0

    for entry in tqdm(data, desc="Auditing Memories"):
        story_id = entry.get("id")
        memory_logs = entry.get("memory_logs", [])
        
        running_subjects = set() 
        story_trace = {"id": story_id, "steps": [], "subjects_lost_count": 0, "empty_memories_count": 0}
        story_has_error = False
        story_lost_count = 0
        story_empty_count = 0
        
        for log in memory_logs:
            try:
                current_doc = next(doc_iter)
            except StopIteration: break
                
            m_total += 1
            current_subjects = extract_subjects(current_doc)
            
            lost_this_step = running_subjects - current_subjects
            is_empty = len(current_subjects) == 0
            
            is_consistent = (len(lost_this_step) == 0)# and (not is_empty)
            log["is_consistent"] = is_consistent
            
            if not is_consistent:
                m_wrong += 1
                story_has_error = True
                story_lost_count += len(lost_this_step)
                total_subjects_lost += len(lost_this_step)
                
            if is_empty:
                #story_empty_count += 1
                total_empty_memories += 1
            
            running_subjects.update(current_subjects)
            
            story_trace["steps"].append({
                "phrase_number": log.get("phrase_number"),
                "lost_subjects": list(lost_this_step),
                "current_subjects": list(current_subjects),
                "all_known_subjects": list(running_subjects),
                "is_empty_memory": is_empty
            })
            
        if story_has_error: s_flawed += 1
        else: s_perfect += 1
            
        story_trace["subjects_lost_count"] = story_lost_count
        story_trace["empty_memories_count"] = story_empty_count
        results.append(entry)
        traces.append(story_trace)

    total_stories = len(data)
    
    pct_wrong_memories = (m_wrong / m_total * 100) if m_total > 0 else 0
    pct_stories_perfect = (s_perfect / total_stories * 100) if total_stories > 0 else 0
    pct_stories_with_errors = (s_flawed / total_stories * 100) if total_stories > 0 else 0
    pct_empty_memories = (total_empty_memories / m_total * 100) if m_total > 0 else 0


    entity_trace_final = {
        "overall_results": {
            "total_stories": total_stories,
            "total_memories": m_total,
            "total_wrong_memories": m_wrong,
            "percentage_wrong_memories": round(pct_wrong_memories, 2),
            "total_stories_perfect": s_perfect,
            "percentage_stories_perfect": round(pct_stories_perfect, 2),
            "total_stories_with_errors": s_flawed,
            "percentage_stories_with_errors": round(pct_stories_with_errors, 2),
            "mean_wrong_updates_per_story": round(m_wrong / total_stories, 2) if total_stories > 0 else 0,
            "total_subjects_lost": total_subjects_lost,
            "mean_subjects_lost_per_story": round(total_subjects_lost / total_stories, 2) if total_stories > 0 else 0,
            "total_empty_memories": total_empty_memories,  # NEW STAT
            "percentage_empty_memories": pct_empty_memories,  # NEW STAT
            "mean_empty_memories_per_story": round(total_empty_memories / total_stories, 2) if total_stories > 0 else 0  # NEW STAT
        },
        "stories": traces
    }

    return results, entity_trace_final



def start_evaluation():
    print("Setting seeds ...")
    set_seed(42)
    
    benchs = ["ToMi_reality"] #["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test", "ToMi_reality", "HiToM_TABLE0"] #["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
    models_gpu1 = ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]
    models_gpu2 = ["gpt-20b"] #["gpt-20b","gemma2-27b","llama3-70b", "qwen3-30b", "qwen2.5-72b"] #"granite4-32b" #, "qwen3.6-35b"

    ebe_types = ["standard_CLOSEDSOURCE"] #["standard", "obj_location", "obj_location_NEW"] #["standard", "task_specific"] #"obj_location",
    closed_source = ["gpt-5.6-luna"]

        
    summary = {}

    for benchmark in benchs:
        print(f"\nEvaluating benchmark: {benchmark}")
        for m in closed_source:
            for e in ebe_types:
                print(f"Evaluating model: {m} on: {benchmark} EbE type: {e}")
                input_file = f"predictions/{benchmark}_{e}_EbE_{m}_raw.json" #f"predictions_gpu1/{benchmark}_{e}_EbE_{m}_raw.json" #f"predictions/{benchmark}_{e}_EbE_{m}_raw.json"

                if os.path.exists(input_file):
                    with open(input_file, 'r') as f:
                        input_data = json.load(f)

                    enriched_data, trace_log = process_memories_optimized(input_data)

                    with open(f'{FOLDER_OUTPUTS}/{benchmark}_{e}_EbE_{m}_trace.json', 'w') as f:
                        json.dump(enriched_data, f, indent=4)

                    with open(f'{FOLDER_OUTPUTS}/Overall_{benchmark}_{e}_EbE_{m}.json', 'w') as f:
                        json.dump(trace_log, f, indent=4)

                    summary[f"{m}_{benchmark}_{e}"] = trace_log["overall_results"]
                    print(f"\nProcessing Complete. Stats saved to {FOLDER_OUTPUTS}/Overall_{benchmark}_{e}_EbE_{m}.json")
    return summary




if __name__ == "__main__":
    summary = start_evaluation()

    with open(f'{FOLDER_OUTPUTS}/FINAL_SUMMARY_OBJ_GPT5.6_luna.json', 'w') as f:
        json.dump(summary, f, indent=4)
