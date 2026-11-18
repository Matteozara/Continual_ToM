import json
import re
import os

def get_perc(m, t): 
    return (m / t * 100) if t > 0 else 100.0
    
def calculate_logic_score(ground_truth_dict: dict, predicted_str: str) -> dict:
    results = {"loc_m": 0, "loc_t": 0, "b1_m": 0, "b1_t": 0, "bh_m": 0, "bh_t": 0}
    if not ground_truth_dict: return results
    
    pred_lower = predicted_str.lower()
    sentences = [s.strip() for s in re.split(r'[.!?|;]', pred_lower) if s.strip()]
    
    # Expanded keywords for better coverage
    mental_keywords = [
        "believ", "know", "think", "thought", "knew", 
        "realiz", "notic", "awar", "recogniz", "understand",
        "suspect", "assum", "guess", "suppos"
    ]
    verb_pattern = "|".join(mental_keywords)

    # --- 1. LOCATIONS ---
    locs = ground_truth_dict.get("locations", {})
    results["loc_t"] = len(locs)
    for entity, loc in locs.items():
        entity_l = entity.lower()
        loc_words = loc.lower().replace("outside ", "").split()
        for sent in sentences:
            if entity_l in sent and all(w in sent for w in loc_words):
                if "outside" in loc.lower() and "outside" not in sent:
                    continue
                results["loc_m"] += 1
                break

    # --- 2. 1ST ORDER ---
    b1 = ground_truth_dict.get("beliefs", {}).get("1-order", {})
    results["b1_t"] = len(b1)
    for char, _ in b1.items():
        pattern = rf"\b{re.escape(char.lower())}\b.*({verb_pattern})"
        if any(re.search(pattern, sent) for sent in sentences):
            results["b1_m"] += 1

    # --- 3. HIGHER ORDER (Improved Segmentation Logic) ---
    bh = ground_truth_dict.get("beliefs", {}).get("higher-order", {})
    results["bh_t"] = len(bh)
    all_char_names = [n.lower() for n in locs.keys()]
    
    for char, target_content in bh.items():
        char_l = char.lower()
        other_chars = [re.escape(oc) for oc in all_char_names if oc != char_l]
        if not other_chars: continue
        others_pattern = "|".join(other_chars)
        
        # Extract keywords from the gold belief to ensure the model isn't just hallucinating names
        # e.g., if target is "the ball is in the attic", we look for "attic"
        content_keywords = [w for w in target_content.lower().split() if len(w) > 3]

        for sent in sentences:
            # Step A: Find the first character and their mental state
            # Example: "John thinks..."
            match_start = re.search(rf"\b{re.escape(char_l)}\b.*?({verb_pattern})", sent)
            
            if match_start:
                # Step B: Look at everything AFTER the first mental verb
                remainder = sent[match_start.end():]
                
                # Step C: Check for another character + another mental verb in that remainder
                # Example: "...that Mary believes..."
                nested_match = re.search(rf"\b({others_pattern})\b.*?({verb_pattern})", remainder)
                
                if nested_match:
                    # Step D: Optional - Verify the location content is also in this sentence
                    if any(kw in sent for kw in content_keywords):
                        results["bh_m"] += 1
                        break # Found it for this character, move to next bh item
    
    return results

def compare_and_report(json1_path, json2_path, output_file="comparison_results.txt"):
    # ... (Rest of your compare_and_report remains largely the same)
    # Ensure you keep the logic that reads the files and iterates through stories
    with open(json1_path, 'r') as f1, open(json2_path, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    lookup1 = {item['id']: item for item in data1}
    stats = {"loc_m": 0, "loc_t": 0, "b1_m": 0, "b1_t": 0, "bh_m": 0, "bh_t": 0}
    total_steps, total_step_matches, total_stories, perfect_stories_count = 0, 0, 0, 0

    with open(output_file, 'w', encoding='utf-8') as f:
        def dual_print(text):
            print(text); f.write(text + "\n")

        dual_print("--- Detailed Category Logic Log ---")

        for story2 in data2:
            story_id = story2.get('id')
            if story_id not in lookup1: continue
            
            total_stories += 1
            story_has_mismatch = False
            logs = story2.get('memory_logs', [])
            tracking = lookup1[story_id].get('memory_tracking', [])

            for i in range(min(len(logs), len(tracking))):
                total_steps += 1
                text_predicted = logs[i].get('new_memory', "").strip("{}")
                step_results = calculate_logic_score(tracking[i], text_predicted)
                
                for key in stats: stats[key] += step_results[key]
                
                step_total = step_results["loc_t"] + step_results["b1_t"] + step_results["bh_t"]
                step_matches = step_results["loc_m"] + step_results["b1_m"] + step_results["bh_m"]
                
                if step_total > 0 and step_matches == step_total:
                    total_step_matches += 1
                elif step_total > 0:
                    story_has_mismatch = True
                    dual_print(f"  [Step {i}] MISMATCH")
                    dual_print(f"    - Locations:    {step_results['loc_m']}/{step_results['loc_t']}")
                    dual_print(f"    - 1st Order:    {step_results['b1_m']}/{step_results['b1_t']}")
                    dual_print(f"    - Higher Order: {step_results['bh_m']}/{step_results['bh_t']}")
                    dual_print(f"    Predicted: {text_predicted}")
                    dual_print(f"    GOLD: {tracking[i]}")
                    story_has_mismatch = True
                    dual_print(f"  [Story {story_id} - Step {i}] MISMATCH")
                    dual_print(f"    Pred: {text_predicted}\n    Gold: {tracking[i]}")

            if not story_has_mismatch: perfect_stories_count += 1

        # Summary Printing...
        # (Append your summary string formatting here)

        summary = f"""
{'='*50}
OVERALL LOGIC STATISTICS (BY CATEGORY)
{'='*50}
LOCATIONS:      {stats['loc_m']}/{stats['loc_t']} ({get_perc(stats['loc_m'], stats['loc_t']):.2f}%)
1ST ORDER BEL:  {stats['b1_m']}/{stats['b1_t']} ({get_perc(stats['b1_m'], stats['b1_t']):.2f}%)
HIGHER ORDER:   {stats['bh_m']}/{stats['bh_t']} ({get_perc(stats['bh_m'], stats['bh_t']):.2f}%)

TOTAL STEP SUCCESS RATE: {total_step_matches}/{total_steps} ({get_perc(total_step_matches, total_steps):.2f}%)
TOTAL PERFECT STORIES:   {perfect_stories_count}/{total_stories} ({get_perc(perfect_stories_count, total_stories):.2f}%)
{'='*50}
"""
        dual_print(summary)
    
    return summary

# Define your paths and call compare_and_report here



# --- Configuration ---
pred_folder = "./custom_predictions" #"./custom_predictions_gpu1"
GOLD_MEMORY = "./../data/custom/memory_dataset_OQ.json"
OUT_DIR = "./res_memory" #"./res_memory_gpu1"
os.makedirs(OUT_DIR, exist_ok=True)
benchmarks = ["Custom"] #HiToM_TABLE0 #["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
models_gpu1 = ["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
models_gpu2 = ["gpt-20b", "gemma2-27b", "llama3-70b", "qwen3.6-35b", "qwen3-30b", "qwen2.5-72b"] #"granite4-32b",
all_models =  models_gpu2 #models_gpu2 #models_gpu1 +


ebe_type = ["standard_ebe", "task_specific_ebe"] #, "standard_gold_memory"

# This will store our data as: final_results[model_config] = [acc1, acc2, acc3...]
final_results = {}

# --- Processing ---
for model in all_models:
    # 1. Process Standard/CoT/Complete Runs
    for run in ebe_type:
        code = f"{model}_{run}" if run else f"{model}_VP"
        accuracies = []
        
        for bench in benchmarks:
            # Match the file naming convention from your loop
            path = f'{pred_folder}/{bench}_{run}_EbE_{model}_raw.json'
            out_file = f"{OUT_DIR}/{code}.txt"
            final_results[code] = compare_and_report(GOLD_MEMORY, path, out_file)
            


with open(f"{OUT_DIR}/FINAL_SUMMARY.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)