import os
import time
import json
import torch
import random
from tqdm import tqdm
from dotenv import load_dotenv
import json


from openai import (
    APIError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

load_dotenv()


from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_KEY"))

MAX_RETRIES = 5
CHECKPOINT_EVERY = 10   # save every 10 EbE steps


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

import asyncio


async def call_openai_async(
    prompt_content,
    model_name,
    semaphore
):
    async with semaphore:

        for attempt in range(MAX_RETRIES):

            try:
                response = await client.responses.create(
                    model=model_name,
                    input=prompt_content,
                    max_output_tokens=MAX_TOKENS
                )

                return response

            except (
                APIError,
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            ) as e:

                wait = (2 ** attempt) + random.random()

                print(
                    f"API error ({type(e).__name__}) "
                    f"{type(e).__name__}: {e}"
                    f"(attempt {attempt+1}/{MAX_RETRIES})"
                )

                if attempt == MAX_RETRIES - 1:
                    return None

                await asyncio.sleep(wait)


async def run_batch_requests(
    prompts,
    model_name,
    batch_size
):
    responses = []

    for i in range(0, len(prompts), batch_size):

        chunk = prompts[i:i + batch_size]

        semaphore = asyncio.Semaphore(batch_size)

        tasks = [
            call_openai_async(
                prompt,
                model_name,
                semaphore
            )
            for prompt in chunk
        ]

        chunk_responses = await asyncio.gather(
            *tasks,
            return_exceptions=True
        )

        responses.extend(chunk_responses)

        await asyncio.sleep(2)

    return responses



async def eval_tom_benchmarks_EbE(tokenizer, model, model_name="", benchmark="HiToM", data_path=None, max_samples=None, batch_size=16, engine_type="vllm", ebe_type="standard"):
    if benchmark == "ToMChallenges": splits = ["Sally-Anne", "Smarties"]
    elif benchmark == "ToMi": splits = ["val", "test"]
    else: splits = [""]
    
    all_results = {}
    prompt_fn = PROMPT_BUILDERS.get(benchmark)

    for split in splits:
        current_name = f"{benchmark}_{split}" if split else benchmark #f"{benchmark}_TABLE0"
        current_name += f"_{ebe_type}_CLOSEDSOURCE" #f"_{ebe_type}" # f"_{ebe_type}_NEW"
        current_path = os.path.join(data_path, f"{split}.json") if split else data_path
        
        if not os.path.exists(current_path): continue

        output_file, raw_io_file = create_folders_EbE(current_name, model_name)
        checkpoint_file = raw_io_file.replace(".json", "_checkpoint.json")
        
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
                "out_tokens": 0,
                "cached_tokens": 0
            })

        max_story_len = max(len(s["story"]) for s in states)
        start_time = time.time()
        

        for step_idx in tqdm(range(max_story_len + 1), desc=f"Evaluating EbE {current_name}... "):
            active_indices = [idx for idx, s in enumerate(states) if not s["finished"]]
            if not active_indices: break
            
            prompts = []

            for idx in active_indices:

                s = states[idx]

                if step_idx < len(s["story"]):

                    prompt_content = prompt_fn(
                        s["story"][step_idx],
                        s["memory"],
                        ebe_type=ebe_type
                    )

                else:

                    if benchmark in [
                        "ToMChallenges",
                        "ToMi",
                        "ToMi_reality"
                    ]:

                        prompt_content = prompt_fn(
                            "",
                            s["memory"],
                            s["question"],
                            ebe_type=ebe_type
                        )

                    else:

                        prompt_content = prompt_fn(
                            "",
                            s["memory"],
                            s["question"],
                            s["options"],
                            ebe_type=ebe_type
                        )

                prompts.append(prompt_content)

            inference_start = time.time()
            '''print("########################")
            print(f"Prompt: {prompts[0]}")
            print("########################")'''

            responses = await run_batch_requests(
                prompts,
                model_name,
                batch_size
            )

            batch_duration = time.time() - inference_start


            for idx, response in zip(active_indices, responses):

                s = states[idx]

                if isinstance(response, Exception):
                    print(f"Request failed: {response}")
                    response = None


                if response is None:

                    raw_text = ""

                    input_tokens = 0
                    output_tokens = 0
                    cached_tokens = 0


                else:

                    raw_text = response.output_text or ""

                    if response.usage:

                        input_tokens = response.usage.input_tokens
                        output_tokens = response.usage.output_tokens

                        details = getattr(
                            response.usage,
                            "input_tokens_details",
                            None
                        )

                        cached_tokens = (
                            getattr(details, "cached_tokens", 0)
                            if details
                            else 0
                        )

                    else:

                        input_tokens = 0
                        output_tokens = 0
                        cached_tokens = 0


                s["model_time"] += batch_duration / len(active_indices)

                s["in_tokens"] += input_tokens
                s["out_tokens"] += output_tokens
                s["cached_tokens"] += cached_tokens

                

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
                    s["final_res"] = (
                        extract_answer_OQ(raw_text).lower().strip()
                        if benchmark in ["ToMChallenges", "ToMi", "ToMi_reality"]
                        else extract_answer_MC(raw_text).upper().strip()
                    )

                    s["finished"] = True
            
            if (step_idx + 1) % CHECKPOINT_EVERY == 0:

                checkpoint = []

                for s in states:
                    checkpoint.append({
                        "id": s["id"],
                        "memory": s["memory"],
                        "finished": s["finished"],
                        "final_res": s["final_res"],
                        "memory_logs": s["memory_logs"],
                        "model_time": s["model_time"],
                        "in_tokens": s["in_tokens"],
                        "out_tokens": s["out_tokens"],
                        "cached_tokens": s["cached_tokens"],
                    })

                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, indent=2)

                print(f"Checkpoint saved after step {step_idx + 1} (in: {checkpoint_file})")
            

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
                "out_tokens": s["out_tokens"],
                "cached_tokens": s["cached_tokens"],
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

            cat = s["category"]
            if cat not in group_stats:
                group_stats[cat] = {"total": 0, "correct": 0, "not_answered": 0, "model_time": 0.0}
            
            group_stats[cat]["total"] += 1
            if is_correct: group_stats[cat]["correct"] += 1
            if not_answered: group_stats[cat]["not_answered"] += 1
            group_stats[cat]["model_time"] += s["model_time"]
        
        save_results_EbE(results, time.time() - start_time, group_stats, output_file, benchmark_group_keys.get(benchmark, "Question Type"))
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(all_raw_data, f, indent=4)
        
        all_results[current_name] = results

    return all_results