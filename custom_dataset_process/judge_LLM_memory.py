import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
from prompts import build_prompt_Judge_Custom
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




#set_seed(42)

FOLDER_PATH = "./predictions/"


def format_memory_to_string(memory_entry: dict) -> str:
    """
    Converts the structured JSON memory into a string for the prompt,
    explicitly including names for every category.
    """
    if not memory_entry:
        return ""
    
    locs = memory_entry.get("locations", {})
    beliefs_1 = memory_entry.get("beliefs", {}).get("1-order", {})
    beliefs_h = memory_entry.get("beliefs", {}).get("higher-order", {})
    
    # 1. Locations
    mem_parts = []
    loc_text = ", ".join([f"{name} is in the {place}" for name, place in locs.items()])
    mem_parts.append(loc_text)
    
    # 2. First-order Beliefs (e.g., "Jackson: Believes boots are in the bathtub.")
    if beliefs_1:
        b1_text = "\n".join([f"{name}: {belief}" for name, belief in beliefs_1.items()])
        mem_parts.append(b1_text)
        
    # 3. Higher-order Beliefs (e.g., "Jackson: Knows Chloe believes boots are in the bathtub.")
    if beliefs_h:
        bh_text = "\n".join([f"- {name}: {belief}" for name, belief in beliefs_h.items()])
        mem_parts.append(bh_text)
        
    return "\n".join(mem_parts)





def judge_memory_update(judge, tokenizer, json_predictions, json_gold, output_path="judge_results.json", raw_path="judge_results_raw.json", batch_size=32, engine_type="vllm"):
    if engine_type == "vllm":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=42)
    all_prompts = []
    metadata = []

    with open(json_gold, 'r') as f1, open(json_predictions, 'r') as f2:
        memory_gold = json.load(f1)
        memory_predictions = json.load(f2)

    data_gold = {item['id']: item for item in memory_gold}

    for data_predictions in memory_predictions:
        story_id = data_predictions.get('id')
        if story_id not in data_gold: continue
        
        logs_memory = data_predictions.get('memory_logs', [])
        ground_truth_memory = data_gold[story_id].get('memory_tracking', [])

        for i in range(min(len(logs_memory), len(ground_truth_memory))):
            gold_text_memory = format_memory_to_string(ground_truth_memory[i])
            prompt = build_prompt_Judge_Custom(logs_memory[i].get("new_memory"), gold_text_memory)
            formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            all_prompts.append(formatted_prompt)
            metadata.append({
                "id": story_id, 
                "phrase_idx": i,
                "predcited_memory": logs_memory[i].get("new_memory"),
                "gold_memory": gold_text_memory
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
        
        


        mem_tokens = tokenizer.encode(res["predcited_memory"])
        res["predcited_memory_token_count"] = len(mem_tokens)
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







def run_judge(args): 

    output_dir = "./custom_judgeLLM_res"
    res_dir = "./custom_judgeLLM_pred"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    print("Setting seeds ...")
    set_seed(42)
    if args.engine == "vllm":
        print("Setting vllm ...")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        #os.environ["VLLM_BATCH_INVARIANT"] = "1"

    print("Loading judge ...")


    tokenizer, judge = load_model(MODELS_MAP[args.judge], engine_type=args.engine)
    method = "Custom"
    GOLD_MEMORY = "./../data/custom/memory_dataset_OQ.json"
    FOLDER_PREDICTED_MEMORY = "custom_predictions/"
    ebe_type = args.ebe_type
    models_gpu1 = ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]  # ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]
    models_gpu2 = ["gpt-20b","gemma2-27b","llama3-70b", "qwen3.6-35b", "qwen3-30b", "qwen2.5-72b"]
    closed_models = ["gpt-5.6-luna"]
    
    summary = {}

    for model_name in closed_models:
        pred_path = FOLDER_PREDICTED_MEMORY+f"{method}_{ebe_type}_ebe_EbE_{model_name}_raw.json"
        output_file = f"{output_dir}/{method}_EbE_{model_name}.json"
        raw_io_file = f"{res_dir}/{method}_EbE_{model_name}_raw.json"
        summary[f"{model_name}"] = judge_memory_update(judge=judge, tokenizer=tokenizer, output_path=output_file, raw_path=raw_io_file, json_predictions=pred_path, json_gold=GOLD_MEMORY, batch_size=args.batch_size, engine_type=args.engine)


        print(f"Final summary for {model_name}: {summary}")


    return summary



############
#   Main   #
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--judge", choices=list(MODELS_MAP.keys()), default="llama3-70b", help="Model to use to judge the memory updates (default: llama3-70b)")
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm", help="Engine to use for inference (default: vllm)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use (default: 4)")
    parser.add_argument("--ebe_type", choices=["standard", "location", "belief", "obj_location", "task_specific"], default="standard", help="Type of inference to run EbE (with ablation variants of the standard version) (default: 'standard')")

    
    args = parser.parse_args()

    run_judge(args)