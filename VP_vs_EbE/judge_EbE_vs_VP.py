import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import torch

import sys
from pathlib import Path

# 1. Get the directory of my_script.py
current_file = Path(__file__).resolve()

# 2. Get the parent directory (where main_utils.py lives)
parent_directory = current_file.parent.parent

# 3. Add that parent directory to Python's search list
sys.path.append(str(parent_directory))


from utils import load_model, set_seed, MAX_TOKENS, MODELS_MAP
from extract_answers import extract_answer_judge_Custom
from prompts import build_prompt_Judge_VP_vs_EbE
import argparse
import json
import torch
import time
import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
from dotenv import load_dotenv
from collections import defaultdict

from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in environment.")






def judge_memory_update(judge, tokenizer, diff_ids, data_file, output_path="judge_results.json", raw_path="judge_results_raw.json", batch_size=32, engine_type="vllm"):
    if engine_type == "vllm":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=42)
    all_prompts = []
    metadata = []

    with open(data_file, 'r') as f1, open(diff_ids, 'r') as f2:
        data_extracted = json.load(f1)
        ids_diff = json.load(f2)

    target_ids = ids_diff.get("vp_right_ebe_wrong", {}).get("ids", [])

    target_ids_set = set(target_ids)
    matched_stories = [item for item in data_extracted if item.get("id") in target_ids_set]

    for i, story_data in enumerate(matched_stories):
        last_memory = story_data.get('memory_logs')[-1].get("new_memory")
        whole_story = story_data.get('story')
        prompt = build_prompt_Judge_VP_vs_EbE(last_memory, whole_story)
        formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        all_prompts.append(formatted_prompt)
        metadata.append({
            "id": story_data.get('id'), 
            "phrase_idx": i,
            "memory": last_memory,
            "whole_story": whole_story
        })
    

    results = []

    if engine_type == "vllm":
        outputs = judge.generate(all_prompts, sampling_params)
        
        for idx, output in enumerate(outputs):
            raw_response = output.outputs[0].text.strip()
            results.append({
                **metadata[idx],
                "full_prompt": all_prompts[idx],
                "raw_prediction": raw_response,
            })

    else: # Hugging Face
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Judging Memories... "):
            batch = all_prompts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(judge.device)
            
            with torch.no_grad():
                output_ids = judge.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_len = inputs.input_ids.shape[1]
            responses = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

            for idx, resp in enumerate(responses):
                raw_response = resp.strip()
                results.append({
                    **metadata[i + idx],
                    "full_prompt": batch[idx],
                    "raw_prediction": raw_response,
                })


    correct_count = 0
    incorrect_count = 0
    not_answered = 0
    memory_token_counts = []
    story_scores = defaultdict(list)
    all_scores = []


    for res in results:
        pred = extract_answer_judge_Custom(res["raw_prediction"])
        res["output"] = pred
        s_id = res["id"]
        
        try:
            score = float(pred)
            res["score"] = score
            story_scores[s_id].append(score) 
            all_scores.append(score)
        except (ValueError, TypeError):
            continue
        
        


        mem_tokens = tokenizer.encode(res["memory"])
        res["memory_token_count"] = len(mem_tokens)
        memory_token_counts.append(len(mem_tokens))
    

    
    
    total = len(results)
    total_stories = len(story_scores)
    story_means = {s_id: sum(scores) / len(scores) for s_id, scores in story_scores.items()}
    story_level_mean = sum(story_means.values()) / len(story_means) if story_means else 0
    overall_mean = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"Overall Mean: {overall_mean:.2f}")
    print("Means per Story:", story_means)

    


    metrics = {
        "summary_single_memory": {
            "Total samples": total,
            "Average score": overall_mean,
            "Avg predcited_memory tokens": float(np.mean(memory_token_counts)) if memory_token_counts else 0,
            "Max predcited_memory tokens": int(np.max(memory_token_counts)) if memory_token_counts else 0,
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

    RES_DIR = "Res_VP_vs_EbE_res_gpu1/" #"Res_VP_vs_EbE_res/"
    os.makedirs(RES_DIR, exist_ok=True)
    RES_DIR_pred = "Res_VP_vs_EbE_pred_gpu1/" #"Res_VP_vs_EbE_pred/"
    os.makedirs(RES_DIR_pred, exist_ok=True)
    DIFF_DIR = "diff_VP_EbE_gpu1/" #"diff_VP_EbE/"
    PRED_DIR = "./../predictions_gpu1/" #"./../predictions/"
    
    print("Setting seeds ...")
    set_seed(42)
    print("Setting vllm ...")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    #os.environ["VLLM_BATCH_INVARIANT"] = "1"

    print("Loading judge ...")
    judge = "llama3-70b"
    engine = "vllm"
    max_samples = None 
    batch_size = 4


    tokenizer, judge = load_model(MODELS_MAP[judge], engine_type=engine)
    
    benchs = ["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]

    models_gpu1 = ["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
    models_gpu2 = ["gpt-20b", "gemma2-27b", "llama3-70b", "qwen3.6-35b"] #"granite4-32b", 
    all_models = models_gpu1
    
    summary = {}

    for model_name in all_models:
        for benchmark in benchs:
            print(f"Evaluating VP vs EbE - model: {model_name} benchmark: {benchmark}")
            ids_diff = f"{DIFF_DIR}diff_{benchmark}_{model_name}.json"
            data_file = f"{PRED_DIR}{benchmark}_standard_EbE_{model_name}_raw.json"
            output_file = f"{RES_DIR}/{benchmark}_{model_name}.json"
            raw_io_file = f"{RES_DIR_pred}/{benchmark}_{model_name}_raw.json"
            summary[f"{model_name}_{benchmark}"] = judge_memory_update(judge=judge, tokenizer=tokenizer, output_path=output_file, raw_path=raw_io_file, diff_ids=ids_diff, data_file=data_file, batch_size=batch_size, engine_type=engine)


        print(f"Final summary for {model_name}: {summary[f'{model_name}_{benchmark}']}")


    return summary



############
#   Main   #
############

if __name__ == "__main__":
    summary = run_judge()
    summary_out_path = "Res_VP_vs_EbE_res_gpu1/FINAL_SUMMARY_JUDGE.json" #"Res_VP_vs_EbE_res/FINAL_SUMMARY_JUDGE.json"
    with open(summary_out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(summary)