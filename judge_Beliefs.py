import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import spacy
import re
from tqdm import tqdm

try:
    spacy.prefer_gpu()
except:
    pass

MODEL_NAME = "en_core_web_lg"
nlp = spacy.load(MODEL_NAME)

BELIEF_VERBS = {"believe", "think", "know", "see", "witness", "perceive", "understand", "look"}
'''
#to test
BELIEF_VERBS = {
    "believe", "think", "know", "see", "witness", 
    "perceive", "understand", "look", "realize", 
    "notice", "recognize", "suspect", "assume", 
    "guess", "suppose", "feel", "conclude"
}
'''

def clean_text(text):
    text = text.replace("{", "").replace("}", "").replace(";", ". ")
    return re.sub(r'\s+', ' ', text).strip()

def extract_beliefs(doc):
    beliefs = set()
    for token in doc:
        if token.lemma_ in BELIEF_VERBS:
            believer, target_object, location = None, None, None
            for child in token.lefts:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    believer = child.text.lower()
            for child in token.rights:
                if child.dep_ in ("ccomp", "xcomp", "dobj", "prep"):
                    for grandchild in child.subtree:
                        if grandchild.dep_ in ("nsubj", "nsubjpass", "attr") and grandchild.pos_ in ("NOUN", "PROPN"):
                            target_object = grandchild.text.lower()
                        if grandchild.dep_ == "pobj":
                            location = grandchild.text.lower()
            if believer and target_object:
                beliefs.add((believer, target_object, location))
    return beliefs

def process_beliefs_with_stats(data):
    all_texts = [clean_text(log.get("new_memory", "")) 
                 for entry in data for log in entry.get("memory_logs", [])]
    
    docs = list(nlp.pipe(all_texts, batch_size=256))
    doc_iter = iter(docs)
    
    #counters
    total_stories = len(data)
    stories_with_any_belief = 0
    stories_with_belief_drops = 0
    total_memories_processed = 0
    total_beliefs_detected = 0
    total_beliefs_dropped = 0
    
    processed_stories = []

    for entry in tqdm(data, desc="Auditing Belief Persistence"):
        memory_logs = entry.get("memory_logs", [])
        previous_step_beliefs = set()
        story_had_belief = False
        story_had_drop = False
        
        for log in memory_logs:
            try:
                current_doc = next(doc_iter)
            except StopIteration:
                break
            
            total_memories_processed += 1
            current_beliefs = extract_beliefs(current_doc)
            current_pairs = {(b, t) for (b, t, l) in current_beliefs}
            
            dropped = set()
            for (prev_b, prev_t, prev_l) in previous_step_beliefs:
                if (prev_b, prev_t) not in current_pairs:
                    dropped.add((prev_b, prev_t, prev_l))

            #dropped = previous_step_beliefs - current_beliefs
            
            if len(current_beliefs) > 0:
                story_had_belief = True
                total_beliefs_detected += len(current_beliefs)
                
            if len(dropped) > 0 and len(previous_step_beliefs) > 0:
                story_had_drop = True
                total_beliefs_dropped += len(dropped)
            
            log["beliefs_found"] = [list(b) for b in current_beliefs]
            log["beliefs_dropped_here"] = [list(b) for b in dropped]
            previous_step_beliefs = current_beliefs

        if story_had_belief:
            stories_with_any_belief += 1
        if story_had_drop:
            stories_with_belief_drops += 1
            
        processed_stories.append(entry)

    #calculations
    perc_stories_with_belief = (stories_with_any_belief / total_stories * 100) if total_stories > 0 else 0
    perc_stories_without_belief = 100 - perc_stories_with_belief
    
    perc_belief_instability = (stories_with_belief_drops / stories_with_any_belief * 100) if stories_with_any_belief > 0 else 0
    
    perc_total_stories_dropped = (stories_with_belief_drops / total_stories * 100) if total_stories > 0 else 0

    print("\n" + "="*50)
    print("           BELIEF CONSISTENCY & PERCENTAGE REPORT")
    print("="*50)
    print(f"Total Stories Analyzed:           {total_stories}")
    print(f"Stories WITH Beliefs:             {stories_with_any_belief} ({perc_stories_with_belief:.1f}%)")
    print(f"Stories WITHOUT Beliefs:          {total_stories - stories_with_any_belief} ({perc_stories_without_belief:.1f}%)")
    print("-" * 50)
    print(f"Stories with Belief DROPS:        {stories_with_belief_drops} ({perc_total_stories_dropped:.1f}% of total)")
    print(f"Belief Instability Rate*:         {perc_belief_instability:.1f}%")
    print(f"(*% of stories with beliefs that failed to keep them)")
    print("-" * 50)
    print(f"Total Individual Beliefs Found:   {total_beliefs_detected}")
    print(f"Total Individual Beliefs Dropped: {total_beliefs_dropped}")
    print("="*50 + "\n")

    final_output = {
        "stats": {
            "total_stories": total_stories,
            "stories_with_beliefs": stories_with_any_belief,
            "percentage_stories_with_beliefs": perc_stories_with_belief,
            "stories_without_beliefs": total_stories - stories_with_any_belief,
            "percentage_stories_without_beliefs": perc_stories_without_belief,
            "stories_with_dropped_beliefs": stories_with_belief_drops,
            "percentage_total_stories_with_drops": perc_total_stories_dropped,
            "belief_instability_rate": perc_belief_instability,
            "total_beliefs_detected": total_beliefs_detected,
            "total_beliefs_dropped": total_beliefs_dropped
        },
        "data": processed_stories
    }
    
    return final_output

NER_DIR = "NER_Beliefs/" #"NER_Beliefs_gpu1/"
RES_DIR_pred = f"{NER_DIR}/beliefs"
os.makedirs(RES_DIR_pred, exist_ok=True)

benchs = ["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
models_gpu1 = ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]
models_gpu2 = ["gpt-20b","gemma2-27b","llama3-70b", "qwen3.6-35b", "qwen3-30b", "qwen2.5-72b"] #"granite4-32b"

ebe_types = ["standard", "task_specific"] #"obj_location",


    
summary = {}

for benchmark in benchs:
    print(f"\nEvaluating benchmark: {benchmark}")
    for m in models_gpu2:
        for e in ebe_types:
            print(f"Evaluating model: {m} on: {benchmark} EbE type: {e}")
            input_file = f'{NER_DIR}/{benchmark}_{e}_EbE_{m}_trace.json'

            if os.path.exists(input_file):
                with open(input_file, 'r') as f:
                    input_data = json.load(f)

                final_result = process_beliefs_with_stats(input_data)

                with open(f'{RES_DIR_pred}/{benchmark}_{e}_EbE_{m}.json', 'w') as f:
                    json.dump(final_result, f, indent=4)

                summary[f"{m}_{benchmark}_{e}"] = final_result["stats"]
                print(f"\nProcessing Complete. Stats saved to {RES_DIR_pred}/{benchmark}_{e}_EbE_{m}.json")



    
with open(f'{RES_DIR_pred}/FINAL_SUMMARY.json', 'w') as f:
    json.dump(summary, f, indent=4)

