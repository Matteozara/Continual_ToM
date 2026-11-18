import re
import json

def extract_overall_accuracy(file_path):
    """Extracts Overall Accuracy, converts to %, and rounds to 2 decimals."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
            if "Overall Evaluation Summary" in content:
                match = re.search(r"Overall Accuracy:\s*([\d.]+)", content)
                if match:
                    # Multiply by 100 and round to 2 decimal places
                    return round(float(match.group(1)) * 100, 2)
        return 0.0
    except FileNotFoundError:
        return 0.0

# --- Configuration ---
benchmarks = ["Custom"] #HiToM_TABLE0 #["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
models_gpu1 = ["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
models_gpu2 = ["qwen3-30b", "qwen2.5-72b"] #["gpt-20b", "gemma2-27b", "llama3-70b", "qwen3.6-35b", "qwen3-30b", "qwen2.5-72b"] #"granite4-32b",
all_models =  models_gpu2 #models_gpu2 #models_gpu1 +


ebe_type = ["standard_ebe", "task_specific_ebe", "standard_gold_memory"] #"obj_location_NEW"

# This will store our data as: final_results[model_config] = [acc1, acc2, acc3...]
final_results = {}

# --- Processing ---
for model in all_models:
    # 1. Process Standard/CoT/Complete Runs
    for run in ebe_type:
        code = f"{model}_{run}" if run else f"{model}_base"
        accuracies = []
        
        for bench in benchmarks:
            # Match the file naming convention from your loop
            path = f'./custom_results/{bench}_{run}_EbE_{model}.txt'
            accuracies.append(extract_overall_accuracy(path))
            
        final_results[code] = accuracies

# --- Final Output ---
print("\nOVERALL ACCURACIES PER MODEL CONFIGURATION")

for config, acc_list in final_results.items():
    formatted_scores = [f"{a:.2f}" for a in acc_list]
    
    # 2. Join them with a comma and a space
    scores_str = " & ".join(formatted_scores)
    
    # 3. Use a single f-string with a space between the two variables
    # The quotes around the f-string are CRITICAL
    print(f"{config} & {scores_str}")



