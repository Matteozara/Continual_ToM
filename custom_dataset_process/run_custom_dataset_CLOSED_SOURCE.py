
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import re
import asyncio

from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
import time
import json
import torch
import random
from tqdm import tqdm
from vllm import SamplingParams
import sys
from pathlib import Path

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

load_dotenv()


from openai import AsyncOpenAI
from openai import OpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))

# Limit concurrent OpenAI requests
CONCURRENT_REQUESTS = 20
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)


# 1. Get the directory of my_script.py
current_file = Path(__file__).resolve()

# 2. Get the parent directory (where main_utils.py lives)
parent_directory = current_file.parent.parent

# 3. Add that parent directory to Python's search list
sys.path.append(str(parent_directory))

from load_data import load_Custom
from prompts import build_prompt_ToMi_EbE
from extract_answers import extract_answer_MC, extract_answer_OQ, extract_memory
from utils import create_folders_Custom, save_results_EbE, MAX_TOKENS, load_model, set_seed, MODELS_MAP, BENCHMARKS

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in environment.")

PATH_CUSTOM = "./../data/custom/memory_dataset_OQ.json"



@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type(
        (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            APIError,
        )
    ),
    reraise=True,
)
async def generate_single(messages, model_name):

    async with semaphore:

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        return (
            response.choices[0].message.content,
            (
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            ),
        )

async def generate_openai(messages_batch, model_name):

    tasks = [
        generate_single(messages, model_name)
        for messages in messages_batch
    ]

    results = await asyncio.gather(*tasks)

    outputs = [r[0] for r in results]
    token_counts = [r[1] for r in results]

    return outputs, token_counts




def format_memory_to_string(memory_entry: dict) -> str:
    '''if not memory_entry:
        return ""
    
    locs = memory_entry.get("locations", {})
    beliefs_1 = memory_entry.get("beliefs", {}).get("1-order", {})
    beliefs_h = memory_entry.get("beliefs", {}).get("higher-order", {})
    
    # 1. Locations
    mem_parts = []
    loc_text = "Physical locations of people and objects: " + ", ".join([f"{name} is in the {place}" for name, place in locs.items()])
    mem_parts.append(loc_text)
    
    # 2. First-order Beliefs (e.g., "Jackson: Believes boots are in the bathtub.")
    if beliefs_1:
        b1_text = "Individual Beliefs:\n" + "\n".join([f"{name}: {belief}" for name, belief in beliefs_1.items()])
        mem_parts.append(b1_text)
        
    # 3. Higher-order Beliefs (e.g., "Jackson: Knows Chloe believes boots are in the bathtub.")
    if beliefs_h:
        bh_text = "Higher-order Beliefs:\n" + "\n".join([f"{name}: {belief}" for name, belief in beliefs_h.items()])
        mem_parts.append(bh_text)
        
    return "\n".join(mem_parts)'''
    if not memory_entry:
        return ""
    
    locs = memory_entry.get("locations", {})
    beliefs_1 = memory_entry.get("beliefs", {}).get("1-order", {})
    beliefs_h = memory_entry.get("beliefs", {}).get("higher-order", {})
    
    # 1. Locations
    mem_parts = []
    loc_text = "Physical locations of people and objects: " + ", ".join([f"{name} is in the {place}" for name, place in locs.items()])
    mem_parts.append(loc_text)
    
    # 2. First-order Beliefs (e.g., "Jackson: Believes boots are in the bathtub.")
    if beliefs_1:
        b1_text = "\n".join([f"Individual Beliefs of {name}: {belief}" for name, belief in beliefs_1.items()])
        mem_parts.append(b1_text)
        
    # 3. Higher-order Beliefs (e.g., "Jackson: Knows Chloe believes boots are in the bathtub.")
    if beliefs_h:
        bh_text = "\n".join([f"Higher-order Beliefs of {name}: {belief}" for name, belief in beliefs_h.items()])
        mem_parts.append(bh_text)
        
    return "\n".join(mem_parts)

async def eval_tom_benchmarks_Custom(tokenizer, model, model_name="", benchmark="Custom", data_path=None, max_samples=None, batch_size=64, engine_type="vllm", ebe_type="standard"):
    
    eval_methods = ["ebe", "gold_memory"]
    all_results = {}

    for method in eval_methods:
        base_name = f"{benchmark}_{ebe_type}_{method}"
        output_file, raw_io_file = create_folders_Custom(base_name, model_name)

        if not os.path.exists(data_path): continue
        
        full_data = list(load_Custom(data_path))
        if max_samples: full_data = full_data[:max_samples]

        states = []
        for i, ex in enumerate(full_data):
            raw_mem_list = ex.get("memory_tracking", [])
            last_mem_struct = raw_mem_list[-1] if raw_mem_list else {}

            
            gold = ex["answer"].strip().lower().replace(".", "")
            states.append({
                "id": ex.get("id", i),
                "story": ex.get("story"), 
                "question": ex.get("question"),
                "options": "",
                "gold": gold,
                "memory": "",
                "finished": False,
                "memory_logs": [],
                "gold_memory_str": format_memory_to_string(last_mem_struct),
                "final_res": None,
                "category": ex.get("category", "unknown"),
                "model_time": 0.0,
                "in_tokens": 0,
                "out_tokens": 0
            })

        start_time = time.time()
        if method == "ebe":
            max_story_len = max(len(s["story"]) for s in states)
        
            for step_idx in tqdm(range(max_story_len + 1), desc=f"Evaluating EbE Running {method}... "):
                active_indices = [idx for idx, s in enumerate(states) if not s["finished"]]
                if not active_indices: break
                
                inner_batch_size = len(active_indices) if engine_type == "vllm" else batch_size
                
                for b_start in range(0, len(active_indices), inner_batch_size):
                    batch_indices = active_indices[b_start : b_start + inner_batch_size]
                    prompts = []
                    
                    for idx in batch_indices:
                        s = states[idx]
                        if step_idx < len(s["story"]):
                            prompt_content = build_prompt_ToMi_EbE(s["story"][step_idx], s["memory"], ebe_type=ebe_type)
                        else:
                            prompt_content = build_prompt_ToMi_EbE("", s["memory"], s["question"], ebe_type=ebe_type)
                                                
                        prompts.append(prompt_content)

                        

                    # --- 2. Inference ---
                    inference_start = time.time()

                    raw_outputs, token_counts = await generate_openai(
                        prompts,
                        model_name=model_name
                    )

                    inference_duration = (
                        time.time() - inference_start
                    ) / len(prompts)




                    # --- 3. State Update ---
                    for (raw_text, (in_t, out_t)), idx in zip(zip(raw_outputs, token_counts), batch_indices):
                        s = states[idx]
                        s["model_time"] += inference_duration
                        s["in_tokens"] += in_t
                        s["out_tokens"] += out_t

                        if step_idx < len(s["story"]):
                            past_mem = s["memory"]
                            new_mem = extract_memory(raw_text)
                            s["memory"] = new_mem if new_mem else s["memory"]
                            s["memory_logs"].append({
                                "phrase_number": step_idx,
                                "event_text": s["story"][step_idx],
                                "past_memory": past_mem,
                                "predicted_raw": raw_text,
                                "new_memory": s["memory"]
                            })
                        else:
                            # Final answer extraction
                            s["final_res"] = extract_answer_OQ(raw_text).lower().strip()
                            s["finished"] = True
            
        elif method == "gold_memory":
            prompts = []
            for s in states:
                # Use the 'memory_tracking' from dataset as the memory input
                prompt_content = build_prompt_ToMi_EbE("", s["gold_memory_str"], s["question"], ebe_type=ebe_type)
                prompts.append(prompt_content)

               

            # --- 2. Inference ---
            inference_start = time.time()

            raw_outputs, token_counts = await generate_openai(
                prompts,
                model_name=model_name
            )

            inference_duration = (
                time.time() - inference_start
            ) / len(prompts)
            
            
            for r_text, (in_t, out_t), s in zip(raw_outputs, token_counts, states):
                s["model_time"] = inference_duration
                s["in_tokens"] = in_t
                s["out_tokens"] = out_t
                s["final_res"] = extract_answer_OQ(r_text).lower().strip()
                s["memory_logs"] = [{"mode": "gold_memory_last_step", "used_memory": s["gold_memory_str"]}]
                s["finished"] = True
            
                


        # --- FINAL AGGREGATION & SAVING ---
        results = []
        all_raw_data = []
        group_stats = {}
        
        for s in states:
            pred = s["final_res"] if s["final_res"] is not None else ""
            is_correct = (s["gold"] in pred)
            not_answered = pred == ""

            res_entry = {
                "id": s["id"], 
                "predicted": pred, 
                "gold": s["gold"],
                "correct": is_correct, 
                "category": s["category"], 
                "memory_logs": s["memory_logs"],
                "not_answered": not_answered,
                "model_time_event": s["model_time"] / (len(s["story"]) + 1),
                "input_tokens_event": s["in_tokens"] / (len(s["story"]) + 1),
                "output_tokens_event": s["out_tokens"] / (len(s["story"]) + 1),
                "total_time": s["model_time"],
                "model_time_total": s["model_time"],
                "input_tokens_total": s["in_tokens"],
                "output_tokens_total": s["out_tokens"],
                "total_tokens": s["in_tokens"] + s["out_tokens"],
                "in_tokens": s["in_tokens"],
                "out_tokens": s["out_tokens"]
            }
            results.append(res_entry)

            all_raw_data.append({
                "id": s["id"],
                "story": s["story"],
                "question": s["question"],
                "gold": s["gold"],
                "predicted": s["final_res"],
                "memory_logs": s["memory_logs"]
            })

            # Update Per-Category Summary
            cat = s["category"]
            if cat not in group_stats:
                group_stats[cat] = {"total": 0, "correct": 0, "not_answered": 0, "model_time": 0.0}
            
            group_stats[cat]["total"] += 1
            if is_correct: group_stats[cat]["correct"] += 1
            if not_answered: group_stats[cat]["not_answered"] += 1
            group_stats[cat]["model_time"] += s["model_time"]
        
        save_results_EbE(results, time.time() - start_time, group_stats, output_file, "Category")
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(all_raw_data, f, indent=4)
        
        all_results[base_name] = results

    return all_results


async def start_evaluation(args):
    print("Setting seeds ...")
    set_seed(42)
    if args.engine == "vllm":
        print("Setting vllm ...")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        #os.environ["VLLM_BATCH_INVARIANT"] = "1"

    print("Loading model ...")
    tokenizer = None
    model = None

    model_name = "gpt-5.6-luna"
    
    summary = {}

    summary = await eval_tom_benchmarks_Custom(tokenizer, model, model_name=model_name, data_path=PATH_CUSTOM, benchmark="Custom", max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine, ebe_type=args.ebe_type)
        
    return summary



############
#   Main   #
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ebe_type", choices=["standard", "location", "belief", "obj_location", "task_specific"], default="standard", help="Type of inference to run EbE (with ablation variants of the standard version) (default: 'standard')")
    parser.add_argument("--model", choices=list(MODELS_MAP.keys()), default="qwen2.5-7b", help="Model to evaluate (default: qwen2.5-7b)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to evaluate (default: None -> all the samples)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use (default: 4)")
    parser.add_argument("--engine", choices=["vllm", "hf", "openai"], default="openai", help="Engine to use for inference (default: vllm)")
    
    args = parser.parse_args()

    asyncio.run(start_evaluation(args))
    #start_evaluation(args)