# pip install tenacity

import os
import time
import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv
import json



from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)

load_dotenv()


from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

from load_data import load_HiToM, load_BigToM, load_ToMChallenges, load_ToMi
from prompts import (
    build_prompt_HiToM, build_prompt_HiToM_CoT, build_prompt_HiToM_Complete,
    build_prompt_BigToM, build_prompt_BigToM_CoT, build_prompt_BigToM_Complete,
    build_prompt_ToMChallenges, build_prompt_ToMChallenges_CoT, build_prompt_ToMChallenges_Complete,
    build_prompt_ToMi, build_prompt_ToMi_CoT, build_prompt_ToMi_Complete
)
from utils import create_folders, save_results, MAX_TOKENS, save_results_closed_source
from extract_answers import extract_answer_MC, extract_answer_OQ

benchmark_group_keys = {
    "HiToM": "Question Order",
    "BigToM": "Question Type",
    "ToMChallenges": "Question Type",
    "ToMi": "Category",
    "ToMi_reality": "Category"
}


@retry(
    retry=retry_if_exception_type(
        (
            APIError,
            APIConnectionError,
            RateLimitError,
            APITimeoutError,
        )
    ),
    wait=wait_random_exponential(
        min=2,
        max=60
    ),
    stop=stop_after_attempt(5),
    reraise=True,
)
def generate_openai(prompt, model_name):
    start = time.time()
    response = client.responses.create(
        model=model_name,
        input=prompt,
        #temperature=0, #maybe to remove
        max_output_tokens=MAX_TOKENS,
    )

    latency = time.time() - start

    #############################################
    print("\n===== Usage =====")
    print(response.usage)

    # Basic token counts
    print("Input tokens:", getattr(response.usage, "input_tokens", "N/A"))
    print("Output tokens:", getattr(response.usage, "output_tokens", "N/A"))

    # Check for prompt caching info
    details = getattr(response.usage, "input_tokens_details", None)

    if details is not None:
        print("Input token details:", details)

        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            print("Cached tokens:", cached)
    else:
        print("No input_tokens_details field available.")
    
    #############################################

    '''print("\n===== RAW RESPONSE TEXT =====")
    print(response.output_text)

    print("\n===== FULL RESPONSE OBJECT =====")
    print(response)'''


    return (
        response.output_text,
        response.usage.input_tokens,
        response.usage.output_tokens,
        latency,
    )


def eval_tom_benchmarks(tokenizer, model, model_name="", benchmark="HiToM", data_path=None, max_samples=None, suffix="", batch_size=64, engine_type="openai"):
    # 1. Configuration & Split Handling
    # Handle benchmarks that have multiple sub-files/splits
        

    

    if benchmark == "ToMChallenges":
        splits = ["Sally-Anne", "Smarties"]
    elif benchmark == "ToMi":
        splits = ["val", "test"]
    else:
        splits = [""]
    
    all_results = {}

    for split in splits:
        current_name = f"{benchmark}_{split}" if split else benchmark #f"{benchmark}_TABLE0"
        
        # Determine Data Path
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
        
        # Identify grouping key for the final report
        group_key = benchmark_group_keys.get(benchmark, "Question Type")

        # 2. Load Data
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

        # 3. Prepare Prompts
        for count, ex in enumerate(data):
            if benchmark == "HiToM":
                if suffix=="cot": prompt_content = build_prompt_HiToM_CoT(ex)
                elif suffix=="complete": prompt_content = build_prompt_HiToM_Complete(ex)
                else: prompt_content = build_prompt_HiToM(ex)
                gold = ex.get("gold", "").upper().strip()
                cat = ex.get("question_order", "unknown")
            elif benchmark == "BigToM":
                t_ans, w_ans = ex["true_answer"].strip(), ex["wrong_answer"].strip()
                ans_list = [t_ans, w_ans]
                random.seed(42 + count)
                random.shuffle(ans_list)
                gold = "A" if ans_list[0] == t_ans else "B"
                if suffix=="cot": prompt_content = build_prompt_BigToM_CoT(ex, ans_list)
                elif suffix=="complete": prompt_content = build_prompt_BigToM_Complete(ex, ans_list)
                else: prompt_content = build_prompt_BigToM(ex, ans_list)
                cat = ex.get("q_type", "unknown")
            elif benchmark == "ToMChallenges":
                gold = ex["answer"].strip().lower().replace(".", "")
                if suffix=="cot": prompt_content = build_prompt_ToMChallenges_CoT(ex)
                elif suffix=="complete": prompt_content = build_prompt_ToMChallenges_Complete(ex)
                else: prompt_content = build_prompt_ToMChallenges(ex)
                cat = ex.get("question_type", "unknown")
            else: # ToMi
                gold = ex["answer"].strip().lower()
                if suffix=="cot": prompt_content = build_prompt_ToMi_CoT(ex)
                elif suffix=="complete": prompt_content = build_prompt_ToMi_Complete(ex)
                else: prompt_content = build_prompt_ToMi(ex)
                cat = ex.get("category", "unknown")

            formatted_prompt = prompt_content

            all_prompts.append(formatted_prompt)
            all_meta.append({"gold": gold, "category": cat, "id": ex.get("id", count)})

        overall_start_time = time.time()

        # 4. Inference Engine
        if engine_type == "openai":

            for prompt, meta in tqdm(
                zip(all_prompts, all_meta),
                total=len(all_prompts),
                desc=f"Eval {current_name}"
            ):
                try:
                    raw_pred, in_tokens, out_tokens, latency = generate_openai(
                        prompt,
                        model_name
                    )

                except Exception as e:
                    print(f"\nFailed after retries: {e}")

                    raw_pred = ""
                    in_tokens = 0
                    out_tokens = 0
                    latency = 0

                process_and_store_result(
                    raw_pred,
                    meta,
                    in_tokens,
                    out_tokens,
                    latency,
                    results,
                    simple_predictions,
                    group_stats,
                    benchmark,
                )

        
        # 5. Finalize Split
        total_time = time.time() - overall_start_time
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(simple_predictions, f, indent=4, ensure_ascii=False)
        save_results(results, total_time, group_stats, output_file, group_key)
        save_results_closed_source(results, model_name, current_name)
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