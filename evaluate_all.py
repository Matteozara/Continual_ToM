import os
import time
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
from load_data import load_HiToM, load_BigToM, load_ToMChallenges, load_ToMi
from prompts import (
    build_prompt_HiToM, build_prompt_HiToM_CoT, build_prompt_HiToM_Complete,
    build_prompt_BigToM, build_prompt_BigToM_CoT, build_prompt_BigToM_Complete,
    build_prompt_ToMChallenges, build_prompt_ToMChallenges_CoT, build_prompt_ToMChallenges_Complete,
    build_prompt_ToMi, build_prompt_ToMi_CoT, build_prompt_ToMi_Complete
)
from utils import create_folders, save_results, MAX_TOKENS
from extract_answers import extract_answer_MC, extract_answer_OQ

benchmark_group_keys = {
    "HiToM": "Question Order",
    "BigToM": "Question Type",
    "ToMChallenges": "Question Type",
    "ToMi": "Category",
    "ToMi_reality": "Category"
}

def eval_tom_benchmarks(tokenizer, model, model_name="", benchmark="HiToM", data_path=None, max_samples=None, suffix="", batch_size=64, engine_type="vllm", type_variation="standard"):
    if engine_type == "vllm":
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
        

    

    if benchmark == "ToMChallenges":
        splits = ["Sally-Anne", "Smarties"]
    elif benchmark == "ToMi":
        splits = ["val", "test"]
    else:
        splits = [""]
    
    all_results = {}

    for split in splits:
        current_name = f"{benchmark}_{split}" if split else benchmark #f"{benchmark}_TABLE0"
        current_name += f"_{type_variation}"
        
        if data_path is None:
            print(f"Skipping {current_name}: The path provided is None")
            continue
        else:
            if benchmark in ["ToMChallenges", "ToMi"]:
                current_path = data_path + f"{split}.json"
            else:
                current_path = data_path

        if not os.path.exists(current_path):
            print(f"Skipping {current_name}: File not found at {current_path}")
            continue

        suffix_task=""
        if suffix=="cot": suffix_task="CoT"
        elif suffix=="complete": suffix_task="Complete"
        else: suffix_task=""
        output_file, raw_io_file = create_folders(current_name, model_name, suffix_task)
        
        group_key = benchmark_group_keys.get(benchmark, "Question Type")

        if benchmark == "HiToM": loader = load_HiToM(current_path)
        elif benchmark == "BigToM": loader = load_BigToM(current_path)
        elif benchmark == "ToMChallenges": loader = load_ToMChallenges(current_path)
        elif benchmark == "ToMi": loader = load_ToMi(current_path)
        elif benchmark == "ToMi_reality": loader = load_ToMi(current_path)
        else: 
            print(f"Skipping {benchmark}: Not implemented")
            continue
        
        data = list(loader)
        if max_samples: data = data[:max_samples]

        all_prompts, all_meta = [], []
        results, simple_predictions = [], []
        group_stats = defaultdict(lambda: {"correct": 0, "total": 0, "model_time": 0.0, "not_answered": 0, "in_tokens": 0, "out_tokens": 0})

        #Prompts
        for count, ex in enumerate(data):
            if benchmark == "HiToM":
                if suffix=="cot": prompt_content = build_prompt_HiToM_CoT(ex, type_variation)
                elif suffix=="complete": prompt_content = build_prompt_HiToM_Complete(ex)
                else: prompt_content = build_prompt_HiToM(ex, type_variation)
                gold = ex.get("gold", "").upper().strip()
                cat = ex.get("question_order", "unknown")
            elif benchmark == "BigToM":
                t_ans, w_ans = ex["true_answer"].strip(), ex["wrong_answer"].strip()
                ans_list = [t_ans, w_ans]
                random.seed(42 + count)
                random.shuffle(ans_list)
                gold = "A" if ans_list[0] == t_ans else "B"
                if suffix=="cot": prompt_content = build_prompt_BigToM_CoT(ex, type_variation, ans_list)
                elif suffix=="complete": prompt_content = build_prompt_BigToM_Complete(ex, ans_list)
                else: prompt_content = build_prompt_BigToM(ex, type_variation, ans_list)
                cat = ex.get("q_type", "unknown")
            elif benchmark == "ToMChallenges":
                gold = ex["answer"].strip().lower().replace(".", "")
                if suffix=="cot": prompt_content = build_prompt_ToMChallenges_CoT(ex, type_variation)
                elif suffix=="complete": prompt_content = build_prompt_ToMChallenges_Complete(ex)
                else: prompt_content = build_prompt_ToMChallenges(ex, type_variation)
                cat = ex.get("question_type", "unknown")
            else: # ToMi
                gold = ex["answer"].strip().lower()
                if suffix=="cot": prompt_content = build_prompt_ToMi_CoT(ex, type_variation)
                elif suffix=="complete": prompt_content = build_prompt_ToMi_Complete(ex)
                else: prompt_content = build_prompt_ToMi(ex, type_variation)
                cat = ex.get("category", "unknown")

            #Fix for Gemma
            if "gemma2" in model_name.lower() and prompt_content[0]["role"] == "system":
                sys_msg = prompt_content.pop(0)["content"]
                prompt_content[0]["content"] = f"{sys_msg}\n\n{prompt_content[0]['content']}"
            
            if "granite" in model_name.lower():
                formatted_prompt = ""
                for msg in prompt_content:
                    #role = msg["role"].upper()
                    content = msg["content"]
                    formatted_prompt += f"{content}\n"
            else:
                formatted_prompt = tokenizer.apply_chat_template(prompt_content, tokenize=False, add_generation_prompt=True)

            all_prompts.append(formatted_prompt)
            all_meta.append({"gold": gold, "category": cat, "id": ex.get("id", count)})

        overall_start_time = time.time()

        #Inference
        if engine_type == "vllm":
            #sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, seed=42)
            start_gen_time = time.time()
            outputs = model.generate(all_prompts, sampling_params)
            total_gen_time = time.time() - start_gen_time
            
            for idx, output in enumerate(outputs):
                process_and_store_result(
                    output.outputs[0].text, all_meta[idx], len(output.prompt_token_ids), 
                    len(output.outputs[0].token_ids), total_gen_time / len(data), 
                    results, simple_predictions, group_stats, benchmark
                )
        else: # HF
            for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"Eval {current_name}"):
                batch_prompts = all_prompts[i : i + batch_size]
                batch_meta = all_meta[i : i + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                
                start_gen_time = time.time()
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=tokenizer.pad_token_id, temperature=None,)
                batch_elapsed = time.time() - start_gen_time

                responses = tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                for idx, raw_pred in enumerate(responses):
                    process_and_store_result(
                        raw_pred, batch_meta[idx], (inputs.input_ids[idx] != tokenizer.pad_token_id).sum().item(),
                        (output_ids[idx, inputs.input_ids.shape[1]:] != tokenizer.pad_token_id).sum().item(),
                        batch_elapsed / len(batch_prompts), results, simple_predictions, group_stats, benchmark
                    )

        total_time = time.time() - overall_start_time
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(simple_predictions, f, indent=4, ensure_ascii=False)
        save_results(results, total_time, group_stats, output_file, group_key)
        all_results[current_name] = results

    return all_results


def process_and_store_result(raw_pred, meta, in_t, out_t, lat, results, simple_predictions, group_stats, benchmark):
    # Determine extraction logic: OQ for ToMC/ToMi, MC for others
    if benchmark in ["ToMChallenges", "ToMi", "ToMi_reality"]:
        pred = extract_answer_OQ(raw_pred).lower().strip()
        is_correct = (meta["gold"] in pred) if pred != "" else False
    else:
        pred = extract_answer_MC(raw_pred).upper().strip()
        is_correct = (meta["gold"] == pred) if pred != "" else False
    
    is_not_answered = (pred == "")
    res_entry = {
        "predicted_raw": raw_pred, "predicted": pred, "gold": meta["gold"],
        "correct": is_correct, "not_answered": is_not_answered,
        "model_time": lat, "in_tokens": in_t, "out_tokens": out_t, "category": meta["category"]
    }
    results.append(res_entry)
    simple_predictions.append({"id": meta["id"], "predicted": pred, "gold": meta["gold"]})

    cat = meta["category"]
    group_stats[cat]["correct"] += 1 if is_correct else 0
    group_stats[cat]["total"] += 1
    group_stats[cat]["model_time"] += lat
    group_stats[cat]["in_tokens"] += in_t
    group_stats[cat]["out_tokens"] += out_t
    group_stats[cat]["not_answered"] += 1 if is_not_answered else 0