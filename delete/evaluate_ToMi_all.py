import os
import torch
import time
import json
import random
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_data import load_ToMi
from prompts import build_prompt_ToMi, build_prompt_ToMi_CoT
from utils import create_folders, save_results, MAX_TOKENS
from extract_answers import extract_answer_OQ

random.seed(42)

def eval_ToMi_all(tokenizer, model, model_name="", data_path="./data/eval/ToMi/", max_samples=None, cot=False, batch_size=64):
    """
    Evaluates ToMi dataset using efficient batch inference.
    Adapts BigToM's batching logic to ToMi's data structure and Open Question (OQ) extraction.
    """
    
    # ToMi evaluates both val and test splits
    for split in ["val", "test"]:
        split_data_path = os.path.join(data_path, f"{split}.json")
        if not os.path.exists(split_data_path):
            print(f"Skipping {split}: file not found at {split_data_path}")
            continue

        # 1. Setup Folders (Using helper from utils)
        # Note: Added split to the folder name to distinguish results
        output_file, raw_io_file = create_folders(f"ToMi_{split}", model_name, cot)

        # 2. Load Data
        loader = load_ToMi(split_data_path)
        data = list(loader)
        if max_samples:
            data = data[:max_samples]

        results = []
        simple_predictions = []
        # Group stats by 'category' as per ToMi requirements
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
        
        for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating ToMi {split} (Batch: {batch_size})..."):
            batch_examples = data[i : i + batch_size]
            prompts = []
            batch_meta = []

            # Prepare Prompts
            for ex in batch_examples:
                gold = ex["answer"].strip().lower()
                
                # Build chat-formatted prompt
                prompt = build_prompt_ToMi_CoT(ex) if cot else build_prompt_ToMi(ex)
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                prompts.append(formatted_prompt)
                batch_meta.append({
                    "gold": gold, 
                    "category": ex.get("category", "unknown"), 
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
                # Use Open-Question extractor for ToMi
                pred = extract_answer_OQ(raw_pred).lower().strip()
                meta = batch_meta[idx]
                
                is_not_answered = (pred == "") 
                # Open question match: check if gold answer is contained in extracted prediction
                is_correct = meta["gold"] in pred if not is_not_answered else False
                
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
                    "category": meta["category"]
                }
                results.append(res_entry)
                
                simple_predictions.append({
                    "id": meta["id"],
                    "predicted": pred,
                    "gold": meta["gold"]
                })

                # Update category-specific stats
                cat = meta["category"]
                group_stats[cat]["correct"] += 1 if is_correct else 0
                group_stats[cat]["total"] += 1
                group_stats[cat]["model_time"] += res_entry["model_time"]
                group_stats[cat]["not_answered"] += 1 if is_not_answered else 0
                group_stats[cat]["in_tokens"] += cur_in_tokens
                group_stats[cat]["out_tokens"] += cur_out_tokens

        total_duration = time.time() - overall_start_time
        
        # 4. Save results for the current split
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(simple_predictions, f, indent=4, ensure_ascii=False)

        # Re-using BigToM's report generator with ToMi's 'Category' label
        save_results(results, total_duration, group_stats, output_file, "Category")

    # Note: Returns results of the final split processed (test)
    return results