import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
import numpy as np
import random
import os
from difflib import SequenceMatcher
import math
from collections import Counter
from vllm import LLM
import json

MAX_TOKENS = 4096
MAX_TOKENS_CoT = 512
MAX_TOKENS_EbE = 512

MODELS_MAP = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507", #run
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct", #run
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "gpt-20b": "openai/gpt-oss-20b", #needs H100 for vllm
    "gemma2-27b": "google/gemma-2-27b-it",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "granite4-32b": "ibm-granite/granite-4.0-h-small-base",
    "gemma4-31b": "google/gemma-4-31B-it",
    "qwen3.6-35b": "Qwen/Qwen3.6-35B-A3B",
    "llama3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",  #"Qwen/Qwen2.5-72B-Instruct",
    #"gemma2-9b": "google/gemma-2-9b-it", #needs H100 for vllm
    #"qwen3.5-9b": "Qwen/Qwen3.5-9B",
    #"qwen3-80b": "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF"
    #"deepseek-dist8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    #"llama4-17b": "meta-llama/Llama-4-Scout-17B-16E-Instruct-Original",

    "lfm2.5-1b": "LiquidAI/LFM2.5-1.2B-Instruct",
    "glm4.7-31b": "zai-org/GLM-4.7-Flash",
    "deepseek-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",

    "gemma4-26b": "google/gemma-4-26B-A4B-it", #VL
    "gemma4-31b-un": "unsloth/gemma-4-31B-it-GGUF", #VL
    "gemma4-26b-ata": "google/gemma-4-26B-A4B-it-assistant", #any to any
    "gemma4-8b-ata": "google/gemma-4-E4B", #any to any
    "mistral3.2-24b": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", #VL
    "mistral3-14b": "mistralai/Ministral-3-14B-Instruct-2512", #VL
    "phi4-6b": "microsoft/Phi-4-multimodal-instruct", #VL
}

BENCHMARKS = ["HiToM", "BigToM", "ToMChallenges", "ToMi"]


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_folders(method, model_name, suffix):
    output_dir = "./results"
    res_dir = "./predictions"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{method}_{suffix}_all_{model_name}.txt"
    raw_io_file = f"{res_dir}/{method}_{suffix}_all_{model_name}_raw.json"
    return output_file, raw_io_file

def create_folders_judge(judge, bench, model_name):
    parent_dir = "./judge_outputs/"
    output_dir = parent_dir+ "results"
    res_dir = parent_dir + "predictions"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    output_file = f"{output_dir}/judge{judge}_{bench}_{model_name}"#.txt"
    raw_io_file = f"{res_dir}/judge{judge}_{bench}_{model_name}_raw"#.json"
    return output_file, raw_io_file

def create_folders_EbE(method, model_name):
    output_dir = "./results"
    res_dir = "./predictions"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{method}_EbE_{model_name}.txt"
    raw_io_file = f"{res_dir}/{method}_EbE_{model_name}_raw.json"
    return output_file, raw_io_file

def create_folders_Custom(method, model_name):
    output_dir = "./custom_results"
    res_dir = "./custom_predictions"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{method}_EbE_{model_name}.txt"
    raw_io_file = f"{res_dir}/{method}_EbE_{model_name}_raw.json"
    return output_file, raw_io_file


def create_folders_wH(method, model_name):
    output_dir = "./results"
    res_dir = "./dataset_with_history"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{method}_wH_{model_name}.txt"
    raw_io_file = f"{res_dir}/{method}_wH_{model_name}_raw.json"
    return output_file, raw_io_file




def extract_letter(s, options=None):
        s = s.upper()
        letters = [chr(65+i) for i in range(len(options))] if options else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        match = re.findall(r'\b([A-Z])\b', s)
        for m in match:
            if m in letters:
                return m
        return None



def device_available():
    return "cuda" if torch.cuda.is_available() else "cpu"



'''def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" # Required for batching in decoder models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model.eval()

    return tokenizer, model
'''

def supports_bf16():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 8


def load_model(model_name, engine_type="vllm"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    is_large_llama = "70b" in model_name.lower() and "llama" in model_name.lower()
    has_bf16 = supports_bf16()

    if engine_type == "vllm":
        if has_bf16:
            best_dtype = "bfloat16"
        elif torch.cuda.is_available():
            best_dtype = "float16"
        else:
            best_dtype = "float32"
        quant_type = "fp8" if is_large_llama else None
        #best_dtype = "float16" if torch.cuda.is_available() else "float32" #"bfloat16" if has_bf16 else "float16" if torch.cuda.is_available() else "float32"
        model = LLM(
            model=model_name,
            #dtype="float16",
            dtype=best_dtype, #"float16" if torch.cuda.is_available() else "float32",
            max_model_len=8192,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            quantization=quant_type
        )
    else:
        if has_bf16:
            best_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            best_dtype = torch.float16
        else:
            best_dtype = torch.float32
        
        load_args = {
            "device_map": "auto",
            "torch_dtype": best_dtype,
        }

        if is_large_llama:
            load_args["load_in_8bit"] = True
        #best_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 #torch.bfloat16 if has_bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_args
        )
        model.eval()

    return tokenizer, model


def generate_answer(tokenizer, model, prompt, max_length=64):
    device = device_available()

    #convert the prompt style (instruction + user)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    
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
            temperature=None,
            top_p=None,
            top_k=None,
        )
    model_time = time.time() - start_model

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    #count output tokens
    output_tokens = generated_tokens.shape[-1]

    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return answer.strip(), model_time, input_tokens, output_tokens



def check_similarity(event, memory):
    return event in memory




def save_results(results, total_end_to_end_time, group_stats, output_file, group_category="Question Type"):
    total_samples = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    total_not_answered = sum(1 for r in results if r["not_answered"])
    total_model_time = sum(r["model_time"] for r in results)    
    total_input_tokens = sum(r["in_tokens"] for r in results)
    total_output_tokens = sum(r["out_tokens"] for r in results)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\n--- Evaluation Finished ---")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    lines = []
    lines.append("------ Overall Evaluation Summary ------")
    lines.append(f"Total samples: {total_samples}")
    lines.append(f"Total corrects: {total_correct}")
    lines.append(f"Overall Accuracy: {overall_accuracy:.4f}")
    lines.append(f"Not Answered: {total_not_answered}")
    lines.append(f"Not Answered Rate: {total_not_answered / total_samples:.2f}")
    lines.append("")
    lines.append("TIME")
    lines.append(f"Total Model Time: {total_model_time:.2f}s")
    lines.append(f"Model Time per sample: {total_model_time / total_samples:.2f}s")
    lines.append(f"Total End-to-End Time: {total_end_to_end_time:.2f}s")
    lines.append(f"Total Time per sample: {total_end_to_end_time / total_samples:.2f}s")
    lines.append("")
    lines.append("TOKENS")
    lines.append(f"Total Input Tokens: {total_input_tokens}")
    lines.append(f"Total Output Tokens: {total_output_tokens}")
    lines.append(f"Total Tokens: {total_input_tokens + total_output_tokens}")
    lines.append(f"Avg Input Tokens per sample: {total_input_tokens / total_samples:.2f}")
    lines.append(f"Avg Output Tokens per sample: {total_output_tokens / total_samples:.2f}")
    lines.append(f"Avg Tokens per sample: {(total_input_tokens + total_output_tokens) / total_samples:.2f}")
    lines.append("---------------------------------------\n")


    lines.append(f"------ Per {group_category} Summary ------")
    for q_type, stats in group_stats.items():
        n_samples = stats["total"]
        accuracy = stats["correct"] / n_samples if n_samples > 0 else 0.0
        not_answered = stats["not_answered"]
        not_answered_rate = not_answered / n_samples if n_samples > 0 else 0.0
        model_time_avg = stats["model_time"] / n_samples if n_samples > 0 else 0.0

        lines.append(f"{group_category}: {q_type}")
        lines.append(f"  Samples: {n_samples}")
        lines.append(f"  Corrects: {stats['correct']}")
        lines.append(f"  Accuracy: {accuracy:.4f}")
        lines.append(f"  Not Answered: {not_answered}")
        lines.append(f"  Not Answered Rate: {not_answered_rate:.2f}")
        lines.append("")
        lines.append("  TIME")
        lines.append(f"  Total Model Time: {stats['model_time']:.2f}s")
        lines.append(f"  Model Time per sample: {model_time_avg:.2f}s")
        lines.append("---------------------------------------")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    



def save_results_EbE(results, total_end_to_end_time, group_stats, output_file, group_category="Question Type"):
    total_samples = len(results)
    if total_samples == 0:
        return

    # Aggregate Overall Totals
    total_correct = sum(1 for r in results if r["correct"])
    total_not_answered = sum(1 for r in results if r["not_answered"])
    total_model_time = sum(r["model_time_total"] for r in results)    
    total_input_tokens = sum(r["input_tokens_total"] for r in results)
    total_output_tokens = sum(r["output_tokens_total"] for r in results)
    
    # Aggregate Event-Level Averages (EbE specific)
    avg_event_time = sum(r["model_time_event"] for r in results) / total_samples
    avg_event_in_tokens = sum(r["input_tokens_event"] for r in results) / total_samples
    avg_event_out_tokens = sum(r["output_tokens_event"] for r in results) / total_samples
    
    overall_accuracy = total_correct / total_samples
    
    print(f"\n--- Evaluation Finished ---")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")

    lines = []
    lines.append("------ Overall Evaluation Summary ------")
    lines.append(f"Total samples: {total_samples}")
    lines.append(f"Total corrects: {total_correct}")
    lines.append(f"Overall Accuracy: {overall_accuracy:.4f}")
    lines.append(f"Not Answered: {total_not_answered}")
    lines.append(f"Not Answered Rate: {total_not_answered / total_samples:.2f}")
    lines.append("")
    
    lines.append("TIME")
    lines.append(f"Total Model Time: {total_model_time:.2f}s")
    lines.append(f"Model Time per sample: {total_model_time / total_samples:.2f}s")
    lines.append(f"Avg Time per EVENT: {avg_event_time:.4f}s") # New EbE Metric
    lines.append(f"Total End-to-End Time: {total_end_to_end_time:.2f}s")
    lines.append(f"Total Time per sample: {total_end_to_end_time / total_samples:.2f}s")
    lines.append("")
    
    lines.append("TOKENS")
    lines.append(f"Total Input Tokens: {total_input_tokens}")
    lines.append(f"Total Output Tokens: {total_output_tokens}")
    lines.append(f"Total Tokens: {total_input_tokens + total_output_tokens}")
    lines.append(f"Avg Input Tokens per sample: {total_input_tokens / total_samples:.2f}")
    lines.append(f"Avg Output Tokens per sample: {total_output_tokens / total_samples:.2f}")
    lines.append(f"Avg Input Tokens per EVENT: {avg_event_in_tokens:.2f}") # New EbE Metric
    lines.append(f"Avg Output Tokens per EVENT: {avg_event_out_tokens:.2f}") # New EbE Metric
    lines.append("---------------------------------------\n")

    lines.append(f"------ Per {group_category} Summary ------")
    for q_type, stats in group_stats.items():
        n_samples = stats["total"]
        accuracy = stats["correct"] / n_samples if n_samples > 0 else 0.0
        not_answered = stats["not_answered"]
        not_answered_rate = not_answered / n_samples if n_samples > 0 else 0.0
        
        lines.append(f"{group_category}: {q_type}")
        lines.append(f"  Samples: {n_samples}")
        lines.append(f"  Corrects: {stats['correct']}")
        lines.append(f"  Accuracy: {accuracy:.4f}")
        lines.append(f"  Not Answered: {not_answered}")
        lines.append(f"  Not Answered Rate: {not_answered_rate:.2f}")
        lines.append("")
        lines.append("  TIME & TOKENS")
        lines.append(f"  Total Model Time: {stats['model_time']:.2f}s")
        lines.append(f"  Model Time per sample: {stats['model_time'] / n_samples:.2f}s")
        
        # Adding token info to per-category if you want it (derived from results)
        cat_in_tokens = sum(r["input_tokens_total"] for r in results if r["category"] == q_type)
        cat_out_tokens = sum(r["output_tokens_total"] for r in results if r["category"] == q_type)
        lines.append(f"  Avg Tokens (In/Out): {cat_in_tokens/n_samples:.1f} / {cat_out_tokens/n_samples:.1f}")
        lines.append("---------------------------------------")

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))





def save_results_closed_source(results, model_name, benchmark):
    folder_name = f"{model_name}_{benchmark}"
    output_dir = os.path.join("CLOSED_SOURCE_responses", folder_name)

    os.makedirs(output_dir, exist_ok=True)

    json_file = os.path.join(output_dir, "responses.json")

    response_dump = []

    for r in results:
        response_dump.append({
            "raw_output": r["predicted_raw"],
            "prediction": r["predicted"],
            "gold": r["gold"],
            "correct": r["correct"],
            "category": r["category"]
        })

    # Append if file exists
    # Append if file exists and contains valid JSON
    if os.path.exists(json_file) and os.path.getsize(json_file) > 0:
        with open(json_file, "r", encoding="utf-8") as f:
            old_results = json.load(f)

        old_results.extend(response_dump)
        response_dump = old_results

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(response_dump, f, indent=4, ensure_ascii=False)

    print(f"Saved raw responses to {json_file}")