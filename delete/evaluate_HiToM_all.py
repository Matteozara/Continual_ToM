from vllm import LLM, SamplingParams
import os
import time
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_data import load_HiToM
from prompts import build_prompt_HiToM, build_prompt_HiToM_CoT
from utils import create_folders, save_results, MAX_TOKENS
from extract_answers import extract_answer_MC

def eval_HiToM_all(tokenizer, model, model_name="", data_path="./data/eval/HiToM/hitom.json", max_samples=None, cot=False, batch_size=64,  engine_type="vllm"):
    output_file, raw_io_file = create_folders("HiToM", model_name, cot)

    # 1. Load Data
    loader = load_HiToM(data_path)
    data = list(loader)
    if max_samples:
        data = data[:max_samples]

    results = []
    simple_predictions = []
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0, "model_time": 0.0, "not_answered": 0, "in_tokens": 0, "out_tokens": 0})

    # 2. Prepare all prompts (Shared Logic)
    all_prompts = []
    all_meta = []
    for count, ex in enumerate(data):
        prompt_content = build_prompt_HiToM_CoT(ex) if cot else build_prompt_HiToM(ex)

        # Gemma System Message Fix
        if "gemma" in model_name.lower():
            if prompt_content and prompt_content[0]["role"] == "system":
                if len(prompt_content) > 1 and prompt_content[1]["role"] == "user":
                    sys_msg = prompt_content.pop(0)["content"]
                    prompt_content[0]["content"] = f"{sys_msg}\n\n{prompt_content[0]['content']}"
                else:
                    prompt_content[0]["role"] = "user"
        
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_content, 
            tokenize=False, 
            add_generation_prompt=True
        )
        all_prompts.append(formatted_prompt)
        all_meta.append({
            "gold": ex.get("gold", "").upper().strip(),
            "q_order": ex.get("question_order", "unknown"),
            "id": ex.get("id", count)
        })

    overall_start_time = time.time()

    # --- ENGINE SWITCH ---
    
    if engine_type == "vllm":
        # model here is expected to be an instance of vllm.LLM
        sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=0.0, # Equivalent to do_sample=False
        )
        
        start_gen_time = time.time()
        outputs = model.generate(all_prompts, sampling_params)
        total_gen_time = time.time() - start_gen_time
        
        # Process vLLM outputs
        for idx, output in enumerate(outputs):
            raw_pred = output.outputs[0].text
            meta = all_meta[idx]
            
            # Extract token counts from vLLM output objects
            cur_in_tokens = len(output.prompt_token_ids)
            cur_out_tokens = len(output.outputs[0].token_ids)
            
            process_and_store_result(
                raw_pred, meta, cur_in_tokens, cur_out_tokens, 
                total_gen_time / len(data), results, simple_predictions, group_stats
            )

    else: # Default: Hugging Face Batching
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Evaluating HF..."):
            batch_prompts = all_prompts[i : i + batch_size]
            batch_meta = all_meta[i : i + batch_size]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            input_length = inputs.input_ids.shape[1]

            start_gen_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            batch_elapsed = time.time() - start_gen_time

            new_tokens = output_ids[:, input_length:]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for idx, raw_pred in enumerate(responses):
                cur_in_tokens = (inputs.input_ids[idx] != tokenizer.pad_token_id).sum().item()
                cur_out_tokens = (new_tokens[idx] != tokenizer.pad_token_id).sum().item()
                
                process_and_store_result(
                    raw_pred, batch_meta[idx], cur_in_tokens, cur_out_tokens, 
                    batch_elapsed / len(batch_prompts), results, simple_predictions, group_stats
                )

    total_end_to_end_time = time.time() - overall_start_time
    
    # 3. Save and Report
    with open(raw_io_file, "w", encoding="utf-8") as f:
        json.dump(simple_predictions, f, indent=4, ensure_ascii=False)

    save_results(results, total_end_to_end_time, group_stats, output_file, "Question Order")
    return results

def process_and_store_result(raw_pred, meta, in_t, out_t, lat, results, simple_predictions, group_stats):
    pred = extract_answer_MC(raw_pred).upper().strip()
    is_not_answered = (pred == "") 
    is_correct = meta["gold"] == pred if pred != "" else False
    
    res_entry = {
        "predicted_raw": raw_pred, "predicted": pred, "gold": meta["gold"],
        "correct": is_correct, "not_answered": is_not_answered,
        "model_time": lat, "in_tokens": in_t, "out_tokens": out_t, "question_order": meta["q_order"]
    }
    results.append(res_entry)
    simple_predictions.append({"id": meta["id"], "predicted": pred, "gold": meta["gold"]})

    q_order = meta["q_order"]
    group_stats[q_order]["correct"] += 1 if is_correct else 0
    group_stats[q_order]["total"] += 1
    group_stats[q_order]["model_time"] += lat
    group_stats[q_order]["in_tokens"] += in_t
    group_stats[q_order]["out_tokens"] += out_t