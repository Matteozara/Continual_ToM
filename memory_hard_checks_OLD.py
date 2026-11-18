import json
import os
import argparse
from utils import MODELS_MAP, BENCHMARKS
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein
from tqdm import tqdm
from fastembed import TextEmbedding

MEMORY_FOLDER = "memory_stats/"

embedding_model = TextEmbedding()

# Initialize Embedding Model once (it's very lightweight and fast)
'''embedding_model = TextEmbedding()

def calculate_sequential_metrics(memories):
    """
    Calculates Levenshtein, Jaccard, and Semantic similarity between 
    consecutive memories.
    """
    if len(memories) < 2:
        return None  # Will handle in the main loop
    
    lev_scores = []
    jac_scores = []
    sem_scores = []
    stagnant_updates = 0

    # Batch embed the entire story's memories for speed
    # We convert to list to index them easily
    memory_embeddings = list(embedding_model.embed(memories))
    
    for i in range(len(memories) - 1):
        m_curr, m_next = memories[i], memories[i+1]
        
        # 1. Levenshtein (Character-level edit distance)
        lev = Levenshtein.ratio(m_curr, m_next)
        lev_scores.append(lev)
        
        # 2. Jaccard (Word-set overlap)
        set_a = set(m_curr.lower().split())
        set_b = set(m_next.lower().split())
        if not set_a and not set_b:
            jac = 1.0
        else:
            jac = len(set_a & set_b) / len(set_a | set_b)
        jac_scores.append(jac)

        # 3. Semantic Cosine Similarity
        v1, v2 = memory_embeddings[i], memory_embeddings[i+1]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        sem_scores.append(cos_sim)
        
        # Threshold for 'stagnant' (using Levenshtein as the primary anchor)
        if lev > 0.95:
            stagnant_updates += 1
                
    return {
        "lev": np.mean(lev_scores),
        "jac": np.mean(jac_scores),
        "sem": np.mean(sem_scores),
        "stagnant": stagnant_updates
    }

def process_subset(subset_data, subset_name, model_name, benchmark):
    all_char_lengths = []
    total_empty_count = 0
    stories_with_empty_count = 0
    total_memories_count = 0
    total_stories = len(subset_data)
    story_mean_tokens = []
    
    # Tracking New Sequential Metrics
    story_lev_scores = []
    story_jac_scores = []
    story_sem_scores = []
    total_stagnant_updates = 0

    for story_item in tqdm(subset_data, desc=f"Processing {subset_name}", leave=False):
        logs = story_item.get('memory_logs', [])
        new_memories = [log.get('new_memory', '') for log in logs if log.get('new_memory')]
        
        clean_memories = [m for m in new_memories if m.strip() not in ["", "{}"]]
        total_memories_count += len(new_memories)
        
        # Calculate Sequential Metrics
        if len(clean_memories) >= 2:
            metrics = calculate_sequential_metrics(clean_memories)
            story_lev_scores.append(metrics["lev"])
            story_jac_scores.append(metrics["jac"])
            story_sem_scores.append(metrics["sem"])
            total_stagnant_updates += metrics["stagnant"]
        
        # Metadata tracking
        num_empty = len(new_memories) - len(clean_memories)
        total_empty_count += num_empty
        if num_empty > 0:
            stories_with_empty_count += 1
            
        token_counts = [len(str(m).split()) for m in new_memories]
        all_char_lengths.extend([len(str(m)) for m in new_memories])
        if token_counts:
            story_mean_tokens.append(sum(token_counts) / len(token_counts))

    # --- Binning & Plotting Logic ---
    fixed_bins = {"0-10": 0, "10-20": 0, "20-30": 0, "30-40": 0, "40-50": 0, "50-60": 0, "60+": 0}
    for t in story_mean_tokens:
        if t < 10: fixed_bins["0-10"] += 1
        elif t < 20: fixed_bins["10-20"] += 1
        elif t < 30: fixed_bins["20-30"] += 1
        elif t < 40: fixed_bins["30-40"] += 1
        elif t < 50: fixed_bins["40-50"] += 1
        elif t < 60: fixed_bins["50-60"] += 1
        else: fixed_bins["60+"] += 1

    if story_mean_tokens:
        labels = list(fixed_bins.keys())
        counts = [fixed_bins[l] for l in labels]
        plt.figure(figsize=(10, 6))
        color = 'teal' if subset_name == "Global" else ('green' if subset_name == "Correct" else 'crimson')
        bars = plt.bar(np.arange(len(labels)), counts, width=0.8, color=color, alpha=0.7, edgecolor='white')
        plt.xlabel('Mean Token Range')
        plt.ylabel('Number of Stories')
        plt.title(f'{subset_name} Distribution: {model_name} ({benchmark})')
        plt.xticks(np.arange(len(labels)), labels)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{MEMORY_FOLDER}{model_name}_{benchmark}_{subset_name.lower()}_distribution.png")
        plt.close()

    pct_empty = (total_empty_count / total_memories_count * 100) if total_memories_count > 0 else 0
    pct_stories_empty = (stories_with_empty_count / total_stories * 100) if total_stories > 0 else 0
    
    return {
        "avg_char_per_memory": round(float(np.mean(all_char_lengths)), 2) if all_char_lengths else 0,
        "avg_seq_levenshtein": round(float(np.mean(story_lev_scores)), 4) if story_lev_scores else 0,
        "avg_seq_jaccard": round(float(np.mean(story_jac_scores)), 4) if story_jac_scores else 0,
        "avg_seq_semantic": round(float(np.mean(story_sem_scores)), 4) if story_sem_scores else 0,
        "total_stagnant_updates": total_stagnant_updates,
        "total_memories": total_memories_count,
        "total_empty": total_empty_count,
        "percentage_empty": f"{round(pct_empty, 2)}%",
        "total_stories": total_stories,
        "percentage_stories_w_empty": f"{round(pct_stories_empty, 2)}%",
        "classes": fixed_bins
    }'''

import re

# ToM-specific anchors: Locations and Mental States
TOM_Pivots = {
    "locations": {"in", "inside", "on", "at", "under", "next", "behind", "into", "from"},
    "beliefs": {"thinks", "knows", "believes", "realizes", "expects", "saw", "heard", "unaware"}
}

def calculate_sequential_metrics(memories):
    if len(memories) < 2:
        return None
    
    lev_scores, sem_scores, pivot_scores = [], [], []
    stagnant_updates = 0
    total_comparisons = len(memories) - 1

    memory_embeddings = list(embedding_model.embed(memories))
    
    for i in range(total_comparisons):
        m_curr, m_next = memories[i].lower(), memories[i+1].lower()
        
        # 1. Levenshtein (Literal Change)
        lev = Levenshtein.ratio(m_curr, m_next)
        lev_scores.append(lev)
        
        # 2. Semantic Similarity (Meaning)
        v1, v2 = memory_embeddings[i], memory_embeddings[i+1]
        sem = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        sem_scores.append(sem)

        # 3. ToM Pivot Score (Did locations or beliefs actually change?)
        # We look at the symmetric difference of ToM-related words
        words_curr = set(re.findall(r'\b\w+\b', m_curr))
        words_next = set(re.findall(r'\b\w+\b', m_next))
        
        relevant_words_curr = words_curr.intersection(TOM_Pivots["locations"] | TOM_Pivots["beliefs"])
        relevant_words_next = words_next.intersection(TOM_Pivots["locations"] | TOM_Pivots["beliefs"])
        
        # Jaccard-style distance specifically for ToM keywords
        if not relevant_words_curr and not relevant_words_next:
            pivot_dist = 1.0 # No ToM words present in either
        else:
            intersection = len(relevant_words_curr & relevant_words_next)
            union = len(relevant_words_curr | relevant_words_next)
            pivot_dist = intersection / union
        
        pivot_scores.append(pivot_dist)
        
        # Threshold for 'stagnant'
        if lev > 0.95:
            stagnant_updates += 1
                
    return {
        "lev": np.mean(lev_scores),
        "sem": np.mean(sem_scores),
        "tom_pivot": np.mean(pivot_scores), # Lower is better (means beliefs/locations changed)
        "stagnant_count": stagnant_updates,
        "total_comps": total_comparisons
    }

def process_subset(subset_data, subset_name, model_name, benchmark):
    all_char_lengths = []
    total_empty_count = 0
    stories_with_empty_count = 0
    total_memories_count = 0
    total_stories = len(subset_data)
    story_mean_tokens = []
    
    # Trackers
    story_lev, story_sem, story_tom = [], [], []
    grand_total_stagnant = 0
    grand_total_comparisons = 0

    for story_item in tqdm(subset_data, desc=f"Processing {subset_name}", leave=False):
        logs = story_item.get('memory_logs', [])
        new_memories = [log.get('new_memory', '') for log in logs if log.get('new_memory')]
        clean_memories = [m for m in new_memories if m.strip() not in ["", "{}"]]
        
        total_memories_count += len(new_memories)
        
        if len(clean_memories) >= 2:
            metrics = calculate_sequential_metrics(clean_memories)
            story_lev.append(metrics["lev"])
            story_sem.append(metrics["sem"])
            story_tom.append(metrics["tom_pivot"])
            grand_total_stagnant += metrics["stagnant_count"]
            grand_total_comparisons += metrics["total_comps"]
        
        # Metadata logic (Same as original)
        num_empty = len(new_memories) - len(clean_memories)
        total_empty_count += num_empty
        if num_empty > 0: stories_with_empty_count += 1
        token_counts = [len(str(m).split()) for m in new_memories]
        all_char_lengths.extend([len(str(m)) for m in new_memories])
        if token_counts: story_mean_tokens.append(sum(token_counts) / len(token_counts))

    # --- Calculations ---
    stagnant_pct = (grand_total_stagnant / grand_total_comparisons * 100) if grand_total_comparisons > 0 else 0

    return {
        "avg_char_per_memory": round(float(np.mean(all_char_lengths)), 2) if all_char_lengths else 0,
        "avg_seq_levenshtein": round(float(np.mean(story_lev)), 4) if story_lev else 0,
        "avg_seq_semantic": round(float(np.mean(story_sem)), 4) if story_sem else 0,
        "avg_tom_belief_stability": round(float(np.mean(story_tom)), 4) if story_tom else 0,
        "total_stagnant_updates": grand_total_stagnant,
        "percentage_stagnant_updates": f"{round(stagnant_pct, 2)}%",
        "total_memories": total_memories_count,
        "percentage_empty_memories": f"{round((total_empty_count/total_memories_count)*100, 2)}%" if total_memories_count > 0 else "0%"
    }

def analyze_memories_from_file(file_path, benchmark, model_name):
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."

    with open(file_path, 'r') as f:
        data = json.load(f)

    # --- Split data into Correct and Wrong categories ---
    correct_stories = [item for item in data if item.get('gold') in item.get('predicted')]
    wrong_stories = [item for item in data if item.get('gold') not in item.get('predicted')]

    # Execute analysis for all three groups
    global_stats = process_subset(data, "Global", model_name, benchmark)
    correct_stats = process_subset(correct_stories, "Correct", model_name, benchmark)
    wrong_stats = process_subset(wrong_stories, "Wrong", model_name, benchmark)

    # Combine results
    final_results = {
        "global_analysis": global_stats,
        "correct_predictions_analysis": correct_stats,
        "wrong_predictions_analysis": wrong_stats,
        "summary": {
            "total_correct": len(correct_stories),
            "total_wrong": len(wrong_stories),
            "accuracy": f"{round(len(correct_stories)/len(data)*100, 2)}%" if data else "0%"
        }
    }

    # Save JSON
    try:
        output_path = f"{MEMORY_FOLDER}{model_name}{benchmark}_memory_check.json"
        with open(output_path, 'w') as out_file:
            json.dump(final_results, out_file, indent=4)
        print(f"Analysis saved to: {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    #return final_results




def start_evaluation(args):
    ebe_type = args.ebe_type
    if args.data == "all":
        benchs = BENCHMARKS
    else:
        benchs = [args.data]
    
    summary = {}
    print("benchmarks: ", benchs)

    for benchmark in benchs:
        print(f"Current bench: {benchmark}")
        if benchmark == "ToMChallenges":
            for split in ["Sally-Anne", "Smarties"]:
                bench = f"{benchmark}_{split}"
                file_path = f"predictions/{bench}_{ebe_type}_EbE_{args.model}_raw.json"
                analyze_memories_from_file(file_path, bench, args.model)


        elif benchmark == "ToMi":
            for split in ["val", "test"]:
                bench = f"{benchmark}_{split}"
                file_path = f"predictions/{bench}_{ebe_type}_EbE_{args.model}_raw.json"
                analyze_memories_from_file(file_path, bench, args.model)
        else:
            file_path = f"predictions/{benchmark}_{ebe_type}_EbE_{args.model}_raw.json"
            analyze_memories_from_file(file_path, benchmark, args.model)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", choices=BENCHMARKS + ["all"], default="all", help="Benchamrk in which test the model (default: all)")
    parser.add_argument("--model", choices=list(MODELS_MAP.keys()), default="qwen2.5-7b", help="Model to evaluate (default: qwen2.5-7b)")
    parser.add_argument("--ebe_type", choices=["standard", "obj_location", "task_specific"], default="standard", help="Type of inference to run EbE (with ablation variants of the standard version) (default: 'standard')")

       
    args = parser.parse_args()

    start_evaluation(args)