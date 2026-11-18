'''

Implememted the new set up to evaluate the model understanding/reasoning giving all the history (past events)

'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import time
import json
import torch
import random
from tqdm import tqdm
from vllm import SamplingParams
from load_data import load_HiToM, load_BigToM, load_ToMChallenges, load_ToMi
from prompts import (
    build_prompt_HiToM_wH, 
    build_prompt_BigToM_wH, 
    build_prompt_ToMChallenges_wH, 
    build_prompt_ToMi_wH
)
from utils import load_model
from extract_answers import extract_answer_MC, extract_answer_OQ, extract_memory
from utils import create_folders_wH, save_results_EbE, MAX_TOKENS

PROMPT_BUILDERS = {
    "HiToM": build_prompt_HiToM_wH,
    "BigToM": build_prompt_BigToM_wH,
    "ToMChallenges": build_prompt_ToMChallenges_wH,
    "ToMi": build_prompt_ToMi_wH
}

benchmark_group_keys = {
    "HiToM": "Question Order",
    "BigToM": "Question Type",
    "ToMChallenges": "Question Type",
    "ToMi": "Category"
}

BENCHMARKS = ["HiToM", "BigToM", "ToMChallenges", "ToMi"]

PATH_HiToM = "./data/eval/HiToM/hitom.json"
PATH_BigToM = "./data/eval/BigToM/bigtom.json"
PATH_ToMi = "./data/eval/ToMi/"
PATH_ToMChallenges = "./data/eval/ToMChallenges/"


MODELS_MAP = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507", #run
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct", #run
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "gpt-20b": "openai/gpt-oss-20b", #needs H100 for vllm
    "gemma2-27b": "google/gemma-2-27b-it",
    "granite4-32b": "ibm-granite/granite-4.0-h-small-base"
    "qwen3-32b": "Qwen/Qwen3-32B",  #maybe no (too many qwens)
    "llama3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    #"gemma2-9b": "google/gemma-2-9b-it", #needs H100 for vllm
    #"qwen3.5-9b": "Qwen/Qwen3.5-9B",
    #"qwen3-80b": "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF"
    #"deepseek-dist8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    #"llama4-17b": "meta-llama/Llama-4-Scout-17B-16E-Instruct-Original",
    }



def eval_tom_benchmarks_wH(tokenizer, model, model_name="", benchmark="HiToM", data_path=None, max_samples=None, batch_size=64, engine_type="vllm"):

    if engine_type == "vllm":
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
    
    
    # 1. Handle Splits
    if benchmark == "ToMChallenges": splits = ["Sally-Anne", "Smarties"]
    elif benchmark == "ToMi": splits = ["val", "test"]
    else: splits = [""]
    
    all_results = {}
    prompt_fn = PROMPT_BUILDERS.get(benchmark)

    for split in splits:
        current_name = f"{benchmark}_{split}" if split else benchmark
        current_path = os.path.join(data_path, f"{split}.json") if split else data_path
        
        if not os.path.exists(current_path): continue

        output_file, raw_io_file = create_folders_wH(current_name, model_name)
        
        loader_map = {"HiToM": load_HiToM, "BigToM": load_BigToM, "ToMChallenges": load_ToMChallenges, "ToMi": load_ToMi}
        full_data = list(loader_map[benchmark](current_path))
        if max_samples: full_data = full_data[:max_samples]

        states = []
        for i, ex in enumerate(full_data):
            options = ""
            if benchmark == "HiToM":
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
        # We loop through steps, but inside each step, we process ALL available stories.
        for step_idx in tqdm(range(max_story_len + 1), desc=f"Evaluating with History {current_name}... "):
            active_indices = [idx for idx, s in enumerate(states) if not s["finished"]]
            if not active_indices: break
            
            inner_batch_size = len(active_indices) if engine_type == "vllm" else batch_size
            
            for b_start in range(0, len(active_indices), inner_batch_size):
                batch_indices = active_indices[b_start : b_start + inner_batch_size]
                prompts = []
                
                for idx in batch_indices:
                    s = states[idx]
                    if step_idx < len(s["story"]):
                        #join all phrases seen so far (not including the current one)   ###############################################################
                        history = "\n".join(s["story"][:step_idx]) if step_idx > 0 else "No prior events."
                        current_event = s["story"][step_idx]
                        prompt_content = prompt_fn(current_event, s["memory"], history=history)
                    else:
                        # Question phase
                        if benchmark in ["ToMChallenges", "ToMi"]:
                            prompt_content = prompt_fn("", s["memory"], "", s["question"])
                        else:
                            prompt_content = prompt_fn("", s["memory"], "", s["question"], s["options"])
                    
                    # Formatting
                    if isinstance(prompt_content, list):
                        prompt_to_format = [dict(m) for m in prompt_content] # Create a copy
                        if "gemma" in model_name.lower() and prompt_to_format[0]["role"] == "system":
                            sys_msg = prompt_to_format.pop(0)["content"]
                            prompt_to_format[0]["content"] = f"{sys_msg}\n\n{prompt_to_format[0]['content']}"
                        
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
                        s["final_res"] = extract_answer_OQ(raw_text).lower().strip() if benchmark in ["ToMChallenges", "ToMi"] else extract_answer_MC(raw_text).upper().strip()
                        s["finished"] = True
            

        # --- FINAL AGGREGATION & SAVING ---
        results = []
        all_raw_data = []
        group_stats = {}
        
        for s in states:
            pred = s["final_res"] if s["final_res"] is not None else ""
            is_correct = (s["gold"] in pred) if benchmark in ["ToMi", "ToMChallenges"] else (s["gold"] == pred)
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







def start_evaluation(args):

    tokenizer, model = load_model(MODELS_MAP[args.model], engine_type=args.engine)
    if args.data == "all":
        benchs = BENCHMARKS
        
    else:
        benchs = [args.data]
    
    summary = {}

    for benchmark in benchs:
        print(f"\nEvaluating benchmark: {benchmark}")

        if benchmark == "HiToM":
            summary["HiToM_wH"] = eval_tom_benchmarks_wH(tokenizer, model, model_name=args.model, data_path=PATH_HiToM, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine)
            
        elif benchmark == "BigToM":
            summary["BigToM_wH"] = eval_tom_benchmarks_wH(tokenizer, model, model_name=args.model, data_path=PATH_BigToM, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine)

        elif benchmark == "ToMChallenges":
            summary["ToMChallenges_wH"] = eval_tom_benchmarks_wH(tokenizer, model, model_name=args.model, data_path=PATH_ToMChallenges, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine)

        elif benchmark == "ToMi":
            summary["ToMi_wH"] = eval_tom_benchmarks_wH(tokenizer, model, model_name=args.model, data_path=PATH_ToMi, benchmark=benchmark, max_samples=args.max_samples, batch_size=args.batch_size, engine_type=args.engine)

        else:
            continue

    return summary



############
#   Main   #
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", choices=BENCHMARKS + ["all"], default="all", help="Benchamrk in which test the model (default: all)")
    parser.add_argument("--model", choices=list(MODELS_MAP.keys()), default="qwen2.5-7b", help="Model to evaluate (default: qwen2.5-7b)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to evaluate (default: None -> all the samples)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use (default: 4)")
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm", help="Engine to use for inference (default: vllm)")
    
    args = parser.parse_args()

    start_evaluation(args)