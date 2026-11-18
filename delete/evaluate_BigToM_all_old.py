import os
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from collections import defaultdict
from load_data import load_BigToM
from tqdm import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"


######################
#   Prompt builder   #
######################

def build_prompt(example):
    story = " ".join(example["story"])
    question = example["question"]

    instruction = (
        "Read the story and answer the question.\n"
        "Answer concisely.\n"
    )

    prompt = (
        f"{instruction}\n"
        f"Story:\n{story}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n"
    )

    return prompt


##################
#   Generation   #
##################

def generate_answer(tokenizer, model, prompt, max_length=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #count tokens input
    input_tokens = inputs["input_ids"].shape[-1]
    
    start_model = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    model_time = time.time() - start_model

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    #count output tokens
    output_tokens = generated_tokens.shape[-1]

    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip(), model_time, input_tokens, output_tokens


#########################
#   Answer extraction   #
#########################

def clean_answer(text):
    return text.strip().lower()


###############################
#   Evaluate single example   #
###############################

def evaluate_example(example, tokenizer, model):
    gold = example["answer"].strip().lower()

    #########   start time
    start_total = time.time()

    prompt = build_prompt(example)
    raw_prediction, model_time, input_tokens, output_tokens = generate_answer(tokenizer, model, prompt)
    pred = raw_prediction.strip().lower()
    correct = gold in pred or pred in gold

    #########   end time
    total_time = time.time() - start_total

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
        "question_order": example.get("question_order", "unknown")
    }


#######################
#   Main evaluation   #
#######################

def eval_BigToM_all(tokenizer, model, model_name = "", data_path = "./data/eval/BigToM/bigtom.json", max_samples=None,):
    #create path for output file
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/BigToM_{model_name}.txt"

    results = []
    total_not_answered = 0
    total_model_time = 0.0
    total_total_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    #tracking per-question_order
    group_stats = defaultdict(lambda: {
        "results": [],
        "model_time": 0.0,
        "total_time": 0.0,
        "correct_count": 0,
        "not_answered": 0,
    })

    loader = load_BigToM(data_path)

    flattened_data = []

    #repeat the story for each question/answer pair
    for item in loader:
        story = item["story"]
        concept = item.get("concept", "")

        for qa in item["qa_pairs"]:
            flattened_data.append({
                "story": story,
                "concept": concept,
                "question": qa["question"],
                "answer": qa["answer"]
            })

    for idx, example in enumerate(tqdm(flattened_data, desc="Evaluating BigToM...")):
        res = evaluate_example(example, tokenizer, model)
        results.append(res)

        total_model_time += res["model_time"]
        total_total_time += res["total_time"]

        total_input_tokens += res["input_tokens"]
        total_output_tokens += res["output_tokens"]

        q_order = example.get("question_order", "unknown")
        group_stats[q_order]["results"].append(res)
        group_stats[q_order]["model_time"] += res["model_time"]
        group_stats[q_order]["total_time"] += res["total_time"]
        if res["correct"]:
            group_stats[q_order]["correct_count"] += 1
        
        if res["predicted"] == "":
            total_not_answered += 1
            group_stats[q_order]["not_answered"] += 1

        if max_samples and idx + 1 >= max_samples:
            break

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

    lines.append("------ Per Question Order Summary ------")
    for q_order, stats in group_stats.items():
        n_samples = len(stats["results"])
        accuracy = stats["correct_count"] / n_samples if n_samples > 0 else 0.0
        not_answered = stats["not_answered"]
        not_answered_rate = not_answered / n_samples if n_samples > 0 else 0.0
        model_time_avg = stats["model_time"] / n_samples if n_samples > 0 else 0.0
        total_time_avg = stats["total_time"] / n_samples if n_samples > 0 else 0.0

        lines.append(f"Question Order: {q_order}")
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
        "per_question_order": group_stats
    }

    return summary