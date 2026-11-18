import os
import time
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from load_data import load_ToMChallenges
from prompts import build_prompt_ToMChallenges, build_prompt_ToMChallenges_CoT
from utils import create_folders, save_results, MAX_TOKENS
from extract_answers import extract_answer_OQ

random.seed(42)

def eval_ToMC_all(tokenizer, model, model_name="", data_path="./data/eval/ToMChallenges/", max_samples=None, cot=False, batch_size=64):
    """
    Evaluates ToMChallenges using batched inference based on the BigToM script structure.
    """
    results_summary = {}
    
    # Process both splits as per original ToMChallenges logic
    for split in ["Sally-Anne", "Smarties"]:
        split_data_path = os.path.join(data_path, f"{split}.json")
        output_file, raw_io_file = create_folders(f"ToMChallenges_{split}", model_name, cot)

        # 1. Load Data
        loader = load_ToMChallenges(split_data_path)
        data = list(loader)
        if max_samples:
            data = data[:max_samples]

        results = []
        simple_predictions = []
        group_stats = defaultdict(lambda: {
            "correct": 0, 
            "total": 0,
            "model_time": 0.0, 
            "not_answered": 0,
            "in_tokens": 0,
            "out_tokens": 0
        })

        # 2. Batch Inference Loop
        count = 0
        overall_start_time = time.time()
        for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating ToMC {split} (Batch Size: {batch_size})..."):
            batch_examples = data[i : i + batch_size]
            prompts = []
            batch_meta = []

            # Prepare Prompts
            for ex in batch_examples:
                # ToMC uses string comparison for gold answers
                gold = ex["answer"].strip().lower().replace(".", "")
                
                prompt = build_prompt_ToMChallenges_CoT(ex) if cot else build_prompt_ToMChallenges(ex)
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                prompts.append(formatted_prompt)
                batch_meta.append({
                    "gold": gold, 
                    "q_type": ex.get("question_type", "unknown"), 
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
                # Use Open-Question extractor for ToMC
                pred = extract_answer_OQ(raw_pred).lower().strip()
                meta = batch_meta[idx]
                
                is_not_answered = (pred == "") 
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

        total_time = time.time() - overall_start_time
        
        # 3. Save and Report per split
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(simple_predictions, f, indent=4, ensure_ascii=False)

        save_results(results, total_time, group_stats, output_file, "Question Type")
        results_summary[split] = results

    return results_summary