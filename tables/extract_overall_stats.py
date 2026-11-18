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
benchmarks = ["HiToM_TABLE0", "ToMi_reality"] #HiToM_TABLE0 #["HiToM", "BigToM", "ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"]
models_gpu1 = ["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
models_gpu2 = ["qwen3-30b", "qwen2.5-72b"] #["gpt-20b", "gemma2-27b", "llama3-70b", "qwen3.6-35b", "qwen3-30b", "qwen2.5-72b"] #"granite4-32b",
all_models =  models_gpu2 #models_gpu1 +

run_type = ["", "CoT", "Complete"]
#run_type = ["", "CoT", "Complete"]
ebe_type = ["standard", "obj_location", "obj_location_NEW"] #"obj_location_NEW"
#ebe_type = ["standard", "standard_OLD", "standard_NEW"]
#ebe_type = ["standard", "obj_location", "task_specific"]

# This will store our data as: final_results[model_config] = [acc1, acc2, acc3...]
final_results = {}

# --- Processing ---
for model in all_models:
    # 1. Process Standard/CoT/Complete Runs
    for run in run_type:
        code = f"{model}_{run}" if run else f"{model}_base"
        accuracies = []
        
        for bench in benchmarks:
            # Match the file naming convention from your loop
            file_mid = f"{run}_all"
            path = f'./../results/{bench}_{file_mid}_{model}.txt'
            accuracies.append(extract_overall_accuracy(path))
            
        final_results[code] = accuracies

    # 2. Process EbE Runs
    for ebe in ebe_type:
        code = f"{model}_{ebe}"
        accuracies = []
        
        for bench in benchmarks:
            # Match the EbE file naming convention from your loop
            path = f'./../results/{bench}_{ebe}_EbE_{model}.txt'
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



