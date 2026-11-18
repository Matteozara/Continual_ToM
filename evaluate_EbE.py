'''

Implememted the batching phase

'''


import os
import time
import json
import torch
import random
from tqdm import tqdm
from vllm import SamplingParams
from load_data import load_HiToM, load_BigToM, load_ToMChallenges, load_ToMi
from prompts import (
    build_prompt_HiToM_EbE, 
    build_prompt_BigToM_EbE, 
    build_prompt_ToMChallenges_EbE, 
    build_prompt_ToMi_EbE
)
from extract_answers import extract_answer_MC, extract_answer_OQ, extract_memory
from utils import create_folders_EbE, save_results_EbE, MAX_TOKENS

PROMPT_BUILDERS = {
    "HiToM": build_prompt_HiToM_EbE,
    "HiToM0": build_prompt_HiToM_EbE,
    "BigToM": build_prompt_BigToM_EbE,
    "ToMChallenges": build_prompt_ToMChallenges_EbE,
    "ToMi": build_prompt_ToMi_EbE,
    "ToMi_reality": build_prompt_ToMi_EbE
}

benchmark_group_keys = {
    "HiToM": "Question Order",
    "HiToM0": "Question Order",
    "BigToM": "Question Type",
    "ToMChallenges": "Question Type",
    "ToMi": "Category",
    "ToMi_reality": "Category"
}

def eval_tom_benchmarks_EbE(tokenizer, model, model_name="", benchmark="HiToM", data_path=None, max_samples=None, batch_size=64, engine_type="vllm", ebe_type="standard"):

    if engine_type == "vllm":
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
    
    
    # 1. Handle Splits
    if benchmark == "ToMChallenges": splits = ["Sally-Anne", "Smarties"]
    elif benchmark == "ToMi": splits = ["val", "test"]
    else: splits = [""]
    
    all_results = {}
    prompt_fn = PROMPT_BUILDERS.get(benchmark)

    for split in splits:
        current_name = f"{benchmark}_{split}" if split else benchmark #f"{benchmark}_TABLE0"
        current_name += f"_{ebe_type}_NEW" #f"_{ebe_type}" # f"_{ebe_type}_NEW"
        current_path = os.path.join(data_path, f"{split}.json") if split else data_path
        
        if not os.path.exists(current_path): continue

        output_file, raw_io_file = create_folders_EbE(current_name, model_name)
        
        loader_map = {"HiToM": load_HiToM, "HiToM0": load_HiToM, "BigToM": load_BigToM, "ToMChallenges": load_ToMChallenges, "ToMi": load_ToMi, "ToMi_reality": load_ToMi}
        full_data = list(loader_map[benchmark](current_path))
        if max_samples: full_data = full_data[:max_samples]

        states = []
        for i, ex in enumerate(full_data):
            options = ""
            if benchmark in ("HiToM", "HiToM0"):
                gold = ex.get("gold", "").upper().strip()
                options = ex.get("options", "")
                story_name = "story"
                cat = "question_order"
            elif benchmark == "BigToM":
                t_ans, w_ans = ex["true_answer"].strip(), ex["wrong_answer"].strip()
                ans_list = [t_ans, w_ans]
                random.seed(42 + i)
                random.shuffle(ans_list)
                options = ans_list
                gold = "A" if ans_list[0] == t_ans else "B"
                story_name = "narrative"
                cat = "q_type"
            elif benchmark == "ToMChallenges":
                gold = ex["answer"].strip().lower().replace(".", "")
                story_name = "story"
                cat = "question_type"
            else: # ToMi
                gold = ex["answer"].strip().lower()
                story_name = "story"
                cat = "category"
            

            states.append({
                "id": ex.get("id", i),
                "story": ex.get(story_name), 
                "question": ex.get("question"),
                "options": options,
                "gold": gold,
                "memory": "",
                "finished": False,
                "memory_logs": [],
                "final_res": None,
                "category": ex.get(cat, "unknown"),
                "model_time": 0.0,
                "in_tokens": 0,
                "out_tokens": 0
            })

        max_story_len = max(len(s["story"]) for s in states)
        start_time = time.time()
        

        # --- SEQUENTIAL STEP LOOP ---
        #loop through steps (inside each step process ALL available stories)
        for step_idx in tqdm(range(max_story_len + 1), desc=f"Evaluating EbE {current_name}... "):
            active_indices = [idx for idx, s in enumerate(states) if not s["finished"]]
            if not active_indices: break
            
            inner_batch_size = len(active_indices) if engine_type == "vllm" else batch_size
            
            for b_start in range(0, len(active_indices), inner_batch_size):
                batch_indices = active_indices[b_start : b_start + inner_batch_size]
                prompts = []
                
                for idx in batch_indices:
                    s = states[idx]
                    if step_idx < len(s["story"]):
                        prompt_content = prompt_fn(s["story"][step_idx], s["memory"], ebe_type=ebe_type)
                    else:
                        # Question phase
                        if benchmark in ["ToMChallenges", "ToMi", "ToMi_reality"]:
                            prompt_content = prompt_fn("", s["memory"], s["question"], ebe_type=ebe_type)
                        else:
                            prompt_content = prompt_fn("", s["memory"], s["question"], s["options"], ebe_type=ebe_type)
                    
                    # Formatting
                    if isinstance(prompt_content, list):
                        #print(f"Prompt to format: {prompt_content}")####################################################
                        prompt_to_format = [dict(m) for m in prompt_content] # Create a copy
                        if "gemma2" in model_name.lower() and prompt_to_format[0]["role"] == "system":
                            sys_msg = prompt_to_format.pop(0)["content"]
                            prompt_to_format[0]["content"] = f"{sys_msg}\n\n{prompt_to_format[0]['content']}"
                        
                        if "granite" in model_name.lower():
                            formatted_p = ""
                            for msg in prompt_to_format:
                                content = msg["content"]
                                formatted_p += f"{content}\n"
                        else:
                            # Standard Chat Template for all other models
                            formatted_p = tokenizer.apply_chat_template(prompt_to_format, tokenize=False, add_generation_prompt=True)


                    else:
                        formatted_p = str(prompt_content)
                    prompts.append(formatted_p if formatted_p.strip() else " ")

                # --- 2. Inference ---
                if engine_type == "vllm":
                    inference_start = time.time()
                    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
                    inference_duration = (time.time() - inference_start) / len(prompts)
                    raw_outputs = [o.outputs[0].text for o in outputs]
                    token_counts = [(len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outputs]
                else:
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
                    inference_start = time.time()
                    with torch.no_grad():
                        gen_tokens = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                    inference_duration = (time.time() - inference_start) / len(prompts)
                    raw_outputs = tokenizer.batch_decode(gen_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    token_counts = [(inputs.input_ids.shape[1], gen_tokens.shape[1] - inputs.input_ids.shape[1]) for _ in raw_outputs]

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
                        s["final_res"] = extract_answer_OQ(raw_text).lower().strip() if benchmark in ["ToMChallenges", "ToMi", "ToMi_reality"] else extract_answer_MC(raw_text).upper().strip()
                        s["finished"] = True
            

        # --- FINAL AGGREGATION & SAVING ---
        results = []
        all_raw_data = []
        group_stats = {}
        
        for s in states:
            pred = s["final_res"] if s["final_res"] is not None else ""
            is_correct = (s["gold"] in pred) if benchmark in ["ToMi", "ToMChallenges", "ToMi_reality"] else (s["gold"] == pred)
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
        
        save_results_EbE(results, time.time() - start_time, group_stats, output_file, benchmark_group_keys.get(benchmark, "Question Type"))
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(all_raw_data, f, indent=4)
        
        all_results[current_name] = results

    return all_results