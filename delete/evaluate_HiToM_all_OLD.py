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

random.seed(42)

def eval_HiToM_all(tokenizer, model, model_name="", data_path="./data/eval/HiToM/hitom.json", max_samples=None, cot=False, batch_size=64):
    output_file, raw_io_file = create_folders("HiToM", model_name, cot)

    # 2. Load Data
    loader = load_HiToM(data_path)
    data = list(loader)
    if max_samples:
        data = data[:max_samples]

    results = []
    simple_predictions = []
    # Tracking stats per question_order
    group_stats = defaultdict(lambda: {
        "correct": 0, 
        "total": 0, 
        "model_time": 0.0, 
        "not_answered": 0, 
        "in_tokens": 0, 
        "out_tokens": 0
    })

    # 3. Batch Inference Loop
    count = 0
    overall_start_time = time.time()
    
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating HiToM (Batch Size: {batch_size})..."):
        batch_examples = data[i : i + batch_size]
        prompts = []
        batch_meta = []

        # Prepare Prompts using Chat Template
        for ex in batch_examples:
            # HiToM prompts usually expect the example dict directly
            prompt_content = build_prompt_HiToM_CoT(ex) if cot else build_prompt_HiToM(ex)

            if "gemma" in model_name.lower():
                if prompt_content and prompt_content[0]["role"] == "system":
                    # Merge system message into the first user message
                    if len(prompt_content) > 1 and prompt_content[1]["role"] == "user":
                        sys_msg = prompt_content.pop(0)["content"]
                        prompt_content[0]["content"] = f"{sys_msg}\n\n{prompt_content[0]['content']}"
                    else:
                        # Fallback: if only a system message exists, just call it a user message
                        prompt_content[0]["role"] = "user"
            
            formatted_prompt = tokenizer.apply_chat_template(
                prompt_content, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts.append(formatted_prompt)
            batch_meta.append({
                "gold": ex.get("gold", "").upper().strip(),
                "q_order": ex.get("question_order", "unknown"),
                "id": ex.get("id", count)
            })
            count += 1

        # Tokenize Batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        start_gen_time = time.time()
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
        batch_elapsed = time.time() - start_gen_time

        # Decode and Process
        new_tokens = output_ids[:, input_length:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for idx, raw_pred in enumerate(responses):
            pred = extract_answer_MC(raw_pred).upper().strip()
            meta = batch_meta[idx]
            is_not_answered = (pred == "") 
            is_correct = meta["gold"] == pred if pred != "" else False
            
            cur_out_tokens = (new_tokens[idx] != tokenizer.pad_token_id).sum().item()
            cur_in_tokens = (inputs.input_ids[idx] != tokenizer.pad_token_id).sum().item()
            
            res_entry = {
                "predicted_raw": raw_pred,
                "predicted": pred,
                "gold": meta["gold"],
                "correct": is_correct,
                "not_answered": is_not_answered,
                "model_time": batch_elapsed / len(batch_examples),
                "in_tokens": cur_in_tokens,
                "out_tokens": cur_out_tokens,
                "question_order": meta["q_order"]
            }
            results.append(res_entry)
            
            simple_predictions.append({
                "id": meta["id"],
                "predicted": pred,
                "gold": meta["gold"]
            })

            # Update stats
            q_order = meta["q_order"]
            group_stats[q_order]["correct"] += 1 if is_correct else 0
            group_stats[q_order]["total"] += 1
            group_stats[q_order]["model_time"] += res_entry["model_time"]
            group_stats[q_order]["not_answered"] += 1 if is_not_answered else 0
            group_stats[q_order]["in_tokens"] += cur_in_tokens
            group_stats[q_order]["out_tokens"] += cur_out_tokens

    total_end_to_end_time = time.time() - overall_start_time
    
    # 4. Save and Report
    with open(raw_io_file, "w", encoding="utf-8") as f:
        json.dump(simple_predictions, f, indent=4, ensure_ascii=False)

    # Reusing the BigToM style save_results function
    save_results(results, total_end_to_end_time, group_stats, output_file, "Question Order")

    return results