import os
# Force vLLM to utilize the classic stable engine for pooling/classification architectures
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import json
import torch
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from vllm import LLM, SamplingParams

# 1. Pipeline path setup
current_file = Path(__file__).resolve()
parent_directory = current_file.parent.parent
sys.path.append(str(parent_directory))

from utils import set_seed

def judge_memory_update_vllm_nli(vllm_engine, diff_ids, data_file, output_path="judge_results.json", raw_path="judge_results_raw.json"):
    all_premises = []
    all_hypotheses = []
    metadata = []

    with open(data_file, 'r') as f1, open(diff_ids, 'r') as f2:
        data_extracted = json.load(f1)
        ids_diff = json.load(f2)

    target_ids = ids_diff.get("vp_right_ebe_wrong", {}).get("ids", [])
    target_ids_set = set(target_ids)
    matched_stories = [item for item in data_extracted if item.get("id") in target_ids_set]

    # For NLI via ModernBERT, inputs are paired sequences
    paired_inputs = []
    
    for i, story_data in enumerate(matched_stories):
        last_memory = story_data.get('memory_logs')[-1].get("new_memory")
        whole_story = story_data.get('story')
        
        all_premises.append(whole_story)
        all_hypotheses.append(last_memory)
        
        # Structure the pair explicitly for the Cross-Encoder pipeline
        paired_inputs.append({"text": whole_story, "text_pair": last_memory})
        
        metadata.append({
            "id": story_data.get('id'), 
            "phrase_idx": i,
            "memory": last_memory,
            "whole_story": whole_story
        })
    
    if not paired_inputs:
        print("No target matching cases found to execute.")
        return {}

    # Execute batch text pair classification natively inside vLLM
    # Note: ModernBERT output via vLLM returns raw un-normalized hidden logits
    outputs = vllm_engine.classify(paired_inputs)
    
    # Dynamically extract labels mapping configurations from the engine's internal setup
    # Typically: 0 -> contradiction, 1 -> neutral, 2 -> entailment
    model_config = vllm_engine.get_tokenizer().model_config if hasattr(vllm_engine.get_tokenizer(), 'model_config') else None
    
    # Default fallback mapping matching standard MNLI configurations if metadata attributes shift
    entail_idx = 2
    contradict_idx = 0
    
    results = []
    for idx, output in enumerate(outputs):
        # Convert classification probabilities using Softmax across logits
        logits = torch.tensor(output.outputs.probs)
        probs = torch.softmax(logits, dim=-1)
        
        entail_prob = probs[entail_idx].item()
        contradict_prob = probs[contradict_idx].item()
        
        score = entail_prob 

        results.append({
            **metadata[idx],
            "entailment_prob": entail_prob,
            "contradiction_prob": contradict_prob,
            "score": score,
            "output": "entailment" if entail_prob > contradict_prob else "contradiction"
        })

    story_scores = defaultdict(list)
    all_scores = []

    for res in results:
        s_id = res["id"]
        score = res["score"]
        story_scores[s_id].append(score) 
        all_scores.append(score)
    
    total = len(results)
    total_stories = len(story_scores)
    story_means = {s_id: float(np.mean(scores)) for s_id, scores in story_scores.items()}
    story_level_mean = float(np.mean(list(story_means.values()))) if story_means else 0
    overall_mean = float(np.mean(all_scores)) if all_scores else 0
    
    print(f"Overall vLLM ModernBERT Entailment Mean: {overall_mean:.4f}")

    metrics = {
        "summary_single_memory": {
            "Total samples": total,
            "Average score (Entailment Prob)": overall_mean,
        },
        "summary_whole_story": {
            "Total stories": total_stories,
            "Mean of story scores": story_level_mean,
            "Average score per story": story_means,
        }
    }

    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return metrics

def run_judge(): 
    RES_DIR = "Res_VP_vs_EbE_res/"
    os.makedirs(RES_DIR, exist_ok=True)
    RES_DIR_pred = "Res_VP_vs_EbE_pred/"
    os.makedirs(RES_DIR_pred, exist_ok=True)
    DIFF_DIR = "diff_VP_EbE/"
    PRED_DIR = "./../predictions/"
    
    print("Setting seeds ...")
    set_seed(42)

    model_name = "MoritzLaurer/ModernBERT-large-zeroshot-v2" 
    print(f"Initializing vLLM Engine for Text-Pair Classification: {model_name}...")
    
    # CRITICAL: Specify task="classify" so vLLM sets up the sequence classification head
    vllm_engine = LLM(
        model=model_name, 
        task="classify", 
        max_model_len=2048, # Maximize ModernBERT's expanded context capacity
        tensor_parallel_size=1
    )
    
    benchs = ["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
    all_models = ["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
    
    summary = {}

    for model in all_models:
        for benchmark in benchs:
            print(f"Evaluating via vLLM ModernBERT NLI - model: {model} benchmark: {benchmark}")
            ids_diff = f"{DIFF_DIR}diff_{benchmark}_{model}.json"
            data_file = f"{PRED_DIR}{benchmark}_standard_EbE_{model}_raw.json"
            output_file = f"{RES_DIR}/{benchmark}_{model}.json"
            raw_io_file = f"{RES_DIR_pred}/{benchmark}_{model}_raw.json"
            
            summary[f"{model}_{benchmark}"] = judge_memory_update_vllm_nli(
                vllm_engine=vllm_engine, 
                output_path=output_file, 
                raw_path=raw_io_file, 
                diff_ids=ids_diff, 
                data_file=data_file
            )

    return summary

if __name__ == "__main__":
    summary = run_judge()