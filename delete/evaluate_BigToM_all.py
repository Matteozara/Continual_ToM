'''
Evaluate BigToM with correct and wrong answers
'''
import os
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from collections import defaultdict
from load_data import load_BigToM
from tqdm import tqdm
from prompts import build_prompt_BigToM, build_prompt_BigToM_CoT
import json
from utils import create_folders, save_results, MAX_TOKENS
from extract_answers import extract_answer_MC

import random

random.seed(42)

def eval_BigToM_all(tokenizer, model, model_name = "", data_path="./data/eval/BigToM/bigtom.json", max_samples=None, cot=False, batch_size=64):
    output_file, raw_io_file = create_folders("BigToM", model_name, cot)


    # 2. Load Data
    loader = load_BigToM(data_path)
    data = list(loader)
    if max_samples:
        data = data[:max_samples]

    results = []
    simple_predictions = []
    group_stats = defaultdict(lambda: {"correct": 0, "total": 0, "model_time": 0.0, "not_answered": 0, "in_tokens": 0, "out_tokens": 0})

    # 3. Batch Inference Loop
    count = 0
    overall_start_time = time.time()
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating BigToM (Batch Size: {batch_size})..."):
        batch_examples = data[i : i + batch_size]
        prompts = []
        batch_meta = []

        # Prepare Prompts
        for ex in batch_examples:
            true_ans = ex["true_answer"].strip()
            wrong_ans = ex["wrong_answer"].strip()
            answers = [true_ans, wrong_ans]
            random.shuffle(answers)
            
            correct_key = "A" if answers[0] == true_ans else "B"
            prompt = build_prompt_BigToM_CoT(ex, answers) if cot else build_prompt_BigToM(ex, answers)
            formatted_prompt = tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts.append(formatted_prompt)
            batch_meta.append({"gold": correct_key, "q_type": ex.get("q_type", "unknown"), "id": ex.get("id", count)})
            count += 1

        # Tokenize Batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        #batch_input_tokens = inputs.input_ids.shape[1] * len(batch_examples)
        input_length = inputs.input_ids.shape[1]

        # Generate
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        elapsed = time.time() - start_time

        # Decode and Process
        new_tokens = output_ids[:, input_length:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for idx, raw_pred in enumerate(responses):
            pred = extract_answer_MC(raw_pred).upper().strip()
            meta = batch_meta[idx]
            is_not_answered = (pred == "") 
            is_correct = meta["gold"] in pred if not is_not_answered else False
            #is_correct = meta["gold"] in pred
            cur_out_tokens = (new_tokens[idx] != tokenizer.pad_token_id).sum().item()
            cur_in_tokens = (inputs.input_ids[idx] != tokenizer.pad_token_id).sum().item()
            
            res_entry = {
                "predicted_raw": raw_pred,
                "predicted": pred,
                "gold": meta["gold"],
                "correct": is_correct,
                "not_answered": is_not_answered,
                "model_time": elapsed / len(batch_examples),
                "in_tokens": cur_in_tokens,
                "out_tokens": cur_out_tokens,
                "q_type": meta["q_type"]
            }
            results.append(res_entry)
            simple_predictions.append({
                "id": meta["id"],
                "predicted": pred,
                "gold": meta["gold"]
            })

            # Update stats
            q_type = meta["q_type"]
            group_stats[q_type]["correct"] += 1 if is_correct else 0
            group_stats[q_type]["total"] += 1
            group_stats[q_type]["model_time"] += res_entry["model_time"]
            group_stats[q_type]["not_answered"] += 1 if is_not_answered else 0
            group_stats[q_type]["in_tokens"] += cur_in_tokens
            group_stats[q_type]["out_tokens"] += cur_out_tokens

    
    total_end_to_end_time = time.time() - overall_start_time
    
    # 4. Save and Report
    with open(raw_io_file, "w", encoding="utf-8") as f:
        json.dump(simple_predictions, f, indent=4)

    save_results(results, total_end_to_end_time, group_stats, output_file, "Question Type")


    return results

