import os
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from collections import defaultdict
from load_data import load_ToMChallenges
from tqdm import tqdm
from prompts import build_prompt_ToMChallenges, build_prompt_ToMChallenges_CoT
from utils import generate_answer, MAX_TOKENS, MAX_TOKENS_CoT
from extract_answers import extract_answer_OQ
import json



###############################
#   Evaluate single example   #
###############################

def evaluate_example(example, tokenizer, model, cot):
    gold = example["answer"].strip().lower().replace(".", "")

    #########   start time
    start_total = time.time()

    if cot:
        prompt = build_prompt_ToMChallenges_CoT(example)
        max_length = MAX_TOKENS_CoT
    else:
        prompt = build_prompt_ToMChallenges(example)
        max_length = MAX_TOKENS
    raw_prediction, model_time, input_tokens, output_tokens = generate_answer(tokenizer, model, prompt, max_length)
    pred = extract_answer_OQ(raw_prediction)

    #########   end time
    total_time = time.time() - start_total

    if pred:
        correct = gold in pred
    else:
        correct = False


    total_tokens = input_tokens + output_tokens

    return {
        "predicted_raw": raw_prediction,
        "predicted": pred,
        "gold": gold,
        "correct": correct,
        "model_time": model_time,
        "total_time": total_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "q_type": example.get("question_type", "unknown")
    }


#######################
#   Main evaluation   #
#######################

def eval_ToMC_all(tokenizer, model, model_name = "", data_path = "./data/eval/ToMChallenges/", max_samples=None, cot=False):
    cot_suffix = "CoT" if cot else ""
    #create path for output file
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    res_dir = "./predictions"
    os.makedirs(res_dir, exist_ok=True)

    for split in ["Sally-Anne", "Smarties"]:
        split_data_path = f"{data_path}{split}.json"
        output_file = f"{output_dir}/ToMChallenges_{split}_{cot_suffix}_all_{model_name}.txt"
        raw_io_file = f"{res_dir}/ToMChallenges_{split}_{cot_suffix}_all_{model_name}_raw.txt"


        results = []
        simple_predictions = []
        total_not_answered = 0
        total_model_time = 0.0
        total_total_time = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        #tracking per-q_type
        group_stats = defaultdict(lambda: {
            "results": [],
            "model_time": 0.0,
            "total_time": 0.0,
            "correct_count": 0,
            "not_answered": 0,
        })

        loader = load_ToMChallenges(split_data_path)
        data = list(loader)

        for idx, example in enumerate(tqdm(data, desc=f"Evaluating ToMChallenges {split}...")):
            res = evaluate_example(example, tokenizer, model, cot)
            results.append(res)


            simple_entry = {
                "id": example.get("id", idx),
                "predicted_raw": res["predicted_raw"],
                "predicted": res["predicted"],
                "gold": res["gold"]
            }
            simple_predictions.append(simple_entry)

            total_model_time += res["model_time"]
            total_total_time += res["total_time"]

            total_input_tokens += res["input_tokens"]
            total_output_tokens += res["output_tokens"]

            q_type = example.get("question_type", "unknown")
            group_stats[q_type]["results"].append(res)
            group_stats[q_type]["model_time"] += res["model_time"]
            group_stats[q_type]["total_time"] += res["total_time"]
            if res["correct"]:
                group_stats[q_type]["correct_count"] += 1
            
            if res["predicted"] == "":
                total_not_answered += 1
                group_stats[q_type]["not_answered"] += 1

            if max_samples and idx + 1 >= max_samples:
                break

        
        with open(raw_io_file, "w", encoding="utf-8") as f:
            json.dump(simple_predictions, f, indent=4, ensure_ascii=False)

        print(f"All predictions saved at: {raw_io_file}")

        #SUMMARY of results
        total_samples = len(results)
        total_correct = sum(r["correct"] for r in results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        lines = []
        lines.append("------ Overall Evaluation Summary ------")
        lines.append(f"Total samples: {total_samples}")
        lines.append(f"Total corrects: {total_correct}")
        lines.append(f"Overall Accuracy: {overall_accuracy:.2f}")
        lines.append(f"Not Answered: {total_not_answered}")
        lines.append(f"Not Answered Rate: {total_not_answered / total_samples:.2f}")
        lines.append("")
        lines.append("TIME")
        lines.append(f"Total Model Time: {total_model_time:.2f}s")
        lines.append(f"Model Time per sample: {total_model_time / total_samples:.2f}s")
        lines.append(f"Total End-to-End Time: {total_total_time:.2f}s")
        lines.append(f"Total Time per sample: {total_total_time / total_samples:.2f}s")
        lines.append("")
        lines.append("TOKENS")
        lines.append(f"Total Input Tokens: {total_input_tokens}")
        lines.append(f"Total Output Tokens: {total_output_tokens}")
        lines.append(f"Total Tokens: {total_input_tokens + total_output_tokens}")
        lines.append(f"Avg Input Tokens per sample: {total_input_tokens / total_samples:.2f}")
        lines.append(f"Avg Output Tokens per sample: {total_output_tokens / total_samples:.2f}")
        lines.append(f"Avg Tokens per sample: {(total_input_tokens + total_output_tokens) / total_samples:.2f}")
        lines.append("---------------------------------------\n")

        lines.append("------ Per Question Type Summary ------")
        for q_type, stats in group_stats.items():
            n_samples = len(stats["results"])
            accuracy = stats["correct_count"] / n_samples if n_samples > 0 else 0.0
            not_answered = stats["not_answered"]
            not_answered_rate = not_answered / n_samples if n_samples > 0 else 0.0
            model_time_avg = stats["model_time"] / n_samples if n_samples > 0 else 0.0
            total_time_avg = stats["total_time"] / n_samples if n_samples > 0 else 0.0

            lines.append(f"Question Type: {q_type}")
            lines.append(f"  Samples: {n_samples}")
            lines.append(f"  Corrects: {stats['correct_count']}")
            lines.append(f"  Accuracy: {accuracy:.2f}")
            lines.append(f"  Not Answered: {not_answered}")
            lines.append(f"  Not Answered Rate: {not_answered_rate:.2f}")
            lines.append("")
            lines.append("  TIME")
            lines.append(f"  Total Model Time: {stats['model_time']:.2f}s")
            lines.append(f"  Model Time per sample: {model_time_avg:.2f}s")
            lines.append(f"  Total Time: {stats['total_time']:.2f}s")
            lines.append(f"  Total Time per sample: {total_time_avg:.2f}s\n")
            lines.append("---------------------------------------")

        #print
        print("\n".join(lines))

        #save
        with open(output_file, "w") as f:
            f.write("\n".join(lines))

    
    summary = {
        "total_summary": {
            "total_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "total_model_time": total_model_time,
            "total_total_time": total_total_time,
            "model_time_per_sample": total_model_time / total_samples if total_samples > 0 else 0.0,
            "total_time_per_sample": total_total_time / total_samples if total_samples > 0 else 0.0,
            "results": results
        },
        "per_question_type": group_stats
    }

    return summary