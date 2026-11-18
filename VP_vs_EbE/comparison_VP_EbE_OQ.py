import json
import re
import os

def clean_prediction(text):
    """
    Extracts the core answer from strings like '\\answer{{cabinet' or '\\answer{cabinet}'
    """
    if not isinstance(text, str):
        return str(text)
    
    # Matches \answer{{content or \answer{content} or \answer{{content}}
    # This regex looks for the last sequence of word characters/hyphens 
    # inside the brackets/braces.
    match = re.search(r'\\answer\{+([^}\s]+)', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: if regex fails, just strip common LaTeX characters
    return text.replace('\\answer', '').replace('{', '').replace('}', '').strip()

def find_differing_ids(file1_path, file2_path):
    # Load the JSON data
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Create dictionaries for easy lookup: {id: is_correct}
    # We clean the 'predicted' value before comparing to 'gold'
    results1 = {
        item['id']: (clean_prediction(item['predicted']) in item['gold']) 
        for item in data1
    }
    results2 = {
        item['id']: (clean_prediction(item['predicted']) in item['gold']) 
        for item in data2
    }
    
    # Store gold labels for the sanity check
    gold_map1 = {item['id']: item['gold'] for item in data1}
    gold_map2 = {item['id']: item['gold'] for item in data2}

    f1_right_f2_wrong = []
    f1_wrong_f2_right = []

    # Get all common IDs
    common_ids = sorted(set(results1.keys()) & set(results2.keys()))
    
    if not common_ids:
        print("Warning: No common IDs found between the two files.")
    else:
        # --- Sanity Check: Ensure gold labels match ---
        for idx in common_ids:
            if gold_map1[idx] != gold_map2[idx]:
                raise ValueError(
                    f"FATAL: Gold label mismatch for ID {idx}!\n"
                    f"File 1: {gold_map1[idx]} | File 2: {gold_map2[idx]}"
                )
        print(f"Sanity Check Passed: Gold labels match for all {len(common_ids)} common IDs.")

    for idx in common_ids:
        correct1 = results1[idx]
        correct2 = results2[idx]

        if correct1 and not correct2:
            f1_right_f2_wrong.append(idx)
        elif not correct1 and correct2:
            f1_wrong_f2_right.append(idx)

    all_differing_ids = sorted(f1_right_f2_wrong + f1_wrong_f2_right)

    return {
        "f1_right_f2_wrong": f1_right_f2_wrong,
        "f1_wrong_f2_right": f1_wrong_f2_right,
        "all_differing_ids": all_differing_ids
    }

# --- Configuration and Execution ---
PATH_DIR = "diff_VP_EbE_gpu1" #"diff_VP_EbE"
os.makedirs(PATH_DIR, exist_ok=True)

benchs = ["ToMChallenges_Sally-Anne", "ToMChallenges_Smarties", "ToMi_val", "ToMi_test"] #"HiToM", "BigToM", 
models_gpu1 = ["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
models_gpu2 = ["gpt-20b", "gemma2-27b", "granite4-32b", "llama3-70b", "qwen3.6-35b"]
model_list = models_gpu1


for model in model_list:
    for benchmark in benchs:
        print(f"\n--- Processing Model: {model} in {benchmark}---")
        output_filename = f"{PATH_DIR}/diff_{benchmark}_{model}.json"
        file1 = f'../predictions_gpu1/{benchmark}__all_{model}_raw.json' #f'../predictions/{benchmark}__all_{model}_raw.json'
        file2 = f'../predictions_gpu1/{benchmark}_standard_EbE_{model}_raw.json' #f'../predictions/{benchmark}_standard_EbE_{model}_raw.json'


        try:
            result = find_differing_ids(file1, file2)

            final_report = {
                "vp_right_ebe_wrong": {
                    "count": len(result["f1_right_f2_wrong"]),
                    "ids": result["f1_right_f2_wrong"]
                },
                "vp_wrong_ebe_right": {
                    "count": len(result["f1_wrong_f2_right"]),
                    "ids": result["f1_wrong_f2_right"]
                },
                "all_differing_ids": {
                    "total_count": len(result["all_differing_ids"]),
                    "ids": result["all_differing_ids"]
                }
            }

            with open(output_filename, 'w') as f:
                json.dump(final_report, f, indent=4)

            print(f"IDs where VP is Right and EbE is Wrong: {len(result['f1_right_f2_wrong'])}")
            print(f"IDs where VP is Wrong and EbE is Right: {len(result['f1_wrong_f2_right'])}")
            print(f"Successfully saved report to {output_filename}")

        except FileNotFoundError as e:
            print(f"Error: Could not find one of the input files. {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

print("\nAll models processed.")