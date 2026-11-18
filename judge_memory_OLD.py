import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from utils import load_model, set_seed, MAX_TOKENS, create_folders_judge, MODELS_MAP, BENCHMARKS
from extract_answers import extract_answer_judge
from prompts import build_prompt_Judge
import argparse
import json
import torch
import time
import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
from dotenv import load_dotenv

from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in environment.")




#set_seed(42)

FOLDER_PATH = "./predictions/"





def judge_memory_update(model, tokenizer, data_path, output_path="judge_results.json", raw_path="judge_results_raw.json", batch_size=32, engine_type="vllm"):
    if engine_type == "vllm":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, , seed=42)

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        data_list = raw_data if isinstance(raw_data, list) else [raw_data]

    all_prompts = []
    metadata = []

    for entry in data_list:
        story_id = entry.get("id")
        story_list = entry.get("story", [])
        memory_logs = entry.get("memory_logs", [])
        
        current_context = []
        log_map = {log['phrase_number']: log for log in memory_logs}

        for i, phrase in enumerate(story_list):
            story_so_far = "\n".join(current_context)
            current_context.append(phrase)

            if i in log_map:
                log = log_map[i]
                prompt = build_prompt_Judge(log, story_so_far)
                formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                all_prompts.append(formatted_prompt)
                metadata.append({
                    "id": story_id, 
                    "phrase_idx": i,
                    "event_text": log.get('event_text'),
                    "past_memory": log.get('past_memory'),
                    "new_memory": log.get('new_memory')
                })

    results = []


    if engine_type == "vllm":
        outputs = model.generate(all_prompts, sampling_params)
        
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
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
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
    
    story_status = {}
    correct_count = 0
    incorrect_count = 0
    not_answered = 0
    memory_token_counts = []


    for res in results:
        pred = extract_answer_judge(res["raw_prediction"])
        res["output"] = pred
        s_id = res["id"]
        
        #initialize story to correct
        if s_id not in story_status:
            story_status[s_id] = "CORRECT"

        if "INCORRECT" in pred:
            incorrect_count += 1
            story_status[s_id] = "INCORRECT"
        elif "CORRECT" in pred:
            correct_count += 1
        else:
            not_answered += 1
            if story_status[s_id] != "INCORRECT":
                story_status[s_id] = "UNKNOWN"


        mem_tokens = tokenizer.encode(res["new_memory"])
        res["new_memory_token_count"] = len(mem_tokens)
        memory_token_counts.append(len(mem_tokens))

    total_stories = len(story_status)
    correct_stories = sum(1 for status in story_status.values() if status == "CORRECT")
    incorrect_stories = sum(1 for status in story_status.values() if status == "INCORRECT")
    unknown_stories = sum(1 for status in story_status.values() if status == "UNKNOWN")


    total = len(results)
    metrics = {
        "summary_single_memory": {
            "Total samples": total,
            "Judged correct": correct_count,
            "Judged incorrect": incorrect_count,
            "Not answered" : not_answered,
            "Accuracy rate": (correct_count / total) if total > 0 else 0,
            "Not Answered Rate": (not_answered / total) if total > 0 else 0,
            "Avg 'new_memory' tokens": float(np.mean(memory_token_counts)) if memory_token_counts else 0,
            "Max 'new_memory' tokens": int(np.max(memory_token_counts)) if memory_token_counts else 0,
        },
        "summary_whole_story": {
            "Stories fully correct": correct_stories,
            "Stories with at least one memory update incorrect": incorrect_stories,
            "Stories with at least one memory update unjudged": unknown_stories,
            "Whole correct story accuracy": (correct_stories / total_stories) if total_stories > 0 else 0,
            "Whole incorrect story accuracy": (incorrect_stories / total_stories) if total_stories > 0 else 0,
            "Whole unknown story accuracy": (unknown_stories / total_stories) if total_stories > 0 else 0,
        }
    }

    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return results






def run_judge(args): 
    
    print("Setting seeds ...")
    set_seed(42)
    if args.engine == "vllm":
        print("Setting vllm ...")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        #os.environ["VLLM_BATCH_INVARIANT"] = "1"

    print("Loading model ...")


    tokenizer, model = load_model(MODELS_MAP[args.model], engine_type=args.engine)
    if args.benchmark == "all":
        benchs = BENCHMARKS
        
    else:
        benchs = [args.benchmark]
    
    summary = {}

    for benchmark in benchs:
        print(f"\nEvaluating memory trace in: {benchmark}")

        if benchmark == "HiToM":
            output_path, raw_path = create_folders_judge(args.judge, benchmark, args.model)
            file_path = f"{benchmark}_EbE_{args.model}_raw.json"
            path = FOLDER_PATH + file_path
            summary[benchmark] = judge_memory_update(model, tokenizer, path, output_path, raw_path, args.batch_size, args.engine)

        elif benchmark == "BigToM":
            output_path, raw_path = create_folders_judge(args.judge, benchmark, args.model)
            file_path = f"{benchmark}_EbE_{args.model}_raw.json"
            path = FOLDER_PATH + file_path
            summary[benchmark] = judge_memory_update(model, tokenizer, path, output_path, raw_path, args.batch_size, args.engine)

        elif benchmark == "ToMChallenges":
            for split in ["Sally-Anne", "Smarties"]:
                bench_split = f"{benchmark}_{split}"
                output_path, raw_path = create_folders_judge(args.judge, bench_split, args.model)
                file_path = f"{benchmark}_{split}_EbE_{args.model}_raw.json"
                path = FOLDER_PATH + file_path
                summary[bench_split] = judge_memory_update(model, tokenizer, path, output_path, raw_path, args.batch_size, args.engine)


        elif benchmark == "ToMi":
            for split in ["val", "test"]:
                bench_split = f"{benchmark}_{split}"
                output_path, raw_path = create_folders_judge(args.judge, bench_split, args.model)
                file_path = f"{benchmark}_{split}_EbE_{args.model}_raw.json"
                path = FOLDER_PATH + file_path
                summary[bench_split] = judge_memory_update(model, tokenizer, path, output_path, raw_path, args.batch_size, args.engine)


    return summary



############
#   Main   #
############

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--judge", choices=list(MODELS_MAP.keys()), default="qwen2.5-7b", help="Model to use to judge the memory updates (default: qwen2.5-7b)")
    parser.add_argument("--model", choices=list(MODELS_MAP.keys()), default="qwen2.5-7b", help="Models to evaluate the memory (default: qwen2.5-7b)")
    parser.add_argument("--benchmark", choices=BENCHMARKS + ["all"], default="all", help="Benchamrk to check for the memory updates (default: all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to use (default: 4)")
    parser.add_argument("--engine", choices=["vllm", "hf"], default="vllm", help="Engine to use for inference (default: vllm)")
    
    args = parser.parse_args()

    run_judge(args)