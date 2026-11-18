import re
import json
from collections import defaultdict

def extract_summary_data(file_path):
    results = []
    
    # Regular expression patterns for each field
    patterns = {
        "category": r"Category:\s*(.*)",
        "samples": r"Samples:\s*(\d+)",
        "corrects": r"Corrects:\s*(\d+)",
        "accuracy": r"Accuracy:\s*([\d.]+)",
        "not_answered": r"Not Answered:\s*(\d+)"
    }

    try:
        with open(file_path, 'r') as file:
            content = file.read()
            
            # Split the file by the dashed separator to process each block individually
            blocks = content.split('---------------------------------------')
            
            for block in blocks:
                if "Category:" in block:
                    entry = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, block)
                        if match:
                            # Convert numbers to int/float, keep category as string
                            val = match.group(1).strip()
                            if key in ["samples", "corrects", "not_answered"]:
                                entry[key] = int(val)
                            elif key == "accuracy":
                                entry[key] = float(val)
                            else:
                                entry[key] = val
                    
                    if entry:
                        results.append(entry)
                        
        return results

    except FileNotFoundError:
        print("Error: The file was not found.")
        return []

# Usage
benchmark = "ToMi"
data = {}
models_gpu1 = ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]  # ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]
models_gpu2 = ["gpt-20b","gemma2-27b","granite4-32b","llama3-70b"]
ebe_type = ["standard"] #"obj_location", "obj_location_NEW" #["standard", "obj_location", "task_specific"]
run_type = ["CoT"]
output_path = f"{benchmark}_summary_gpu1.json"  #{models_gpu1[0]}_
stats_output_path = f"{benchmark}_top3_summary.txt" #{models_gpu1[0]}_


for model in models_gpu1:
    for split in ["val", "test"]:
        for run in run_type:
            code = f"{split}_{model}_{run}"
            print(f"DATA: ./../../results/{benchmark}_{run}_all_{model}.txt")
            data[code] = extract_summary_data(f'./../../results/{benchmark}_{split}_{run}_all_{model}.txt')
        for ebe in ebe_type:
            code = f"{split}_{model}_{ebe}"
            data[code] = extract_summary_data(f'./../../results/{benchmark}_{split}_{ebe}_EbE_{model}.txt')


print(json.dumps(data, indent=4))

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)







#####   PRINT THE TOP 3 METHODS SCORING FOR EACH SUB-CATEGORY
#####   PRINT THE TOP 3 METHODS SCORING FOR EACH SUB-question_type
all_results = []
for source_key, questions in data.items():
    parts = source_key.split('_')
    split_name = parts[0]
    model_name = parts[1]
    method_type = parts[2] if parts[2] != "" else "Base" # Handle empty run_type
    
    for q in questions:
        all_results.append({
            "split": split_name,
            "model": model_name,
            "method": method_type,
            "category": q["category"],
            "accuracy": q["accuracy"]
        })

# 2. Group by Split, then by Question Type
# Structure: grouped[split][question_type] = [list of results]
grouped = defaultdict(lambda: defaultdict(list))
for record in all_results:
    grouped[record["split"]][record["category"]].append(record)

with open(stats_output_path, "w", encoding="utf-8") as txt_file:
    title = f"TOP 3 ACCURACIES PER CATEGORY PER SPLIT ({benchmark})"
    border = "=" * 70
    
    print(f"\n{border}\n{title}\n{border}")
    txt_file.write(f"{border}\n{title}\n{border}\n")

    # Iterate through each Split (Sally-Anne, Smarties)
    for split_name in sorted(grouped.keys()):
        split_header = f"\n>>> SPLIT: {split_name.upper()}\n{'-' * 30}\n"
        print(split_header)
        txt_file.write(split_header)

        # Iterate through Question Types within that split
        for q_type, results in grouped[split_name].items():
            # Sort by accuracy descending and take top 3
            top_3 = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:3]
            
            category_line = f"Category: {q_type}"
            print(f"{category_line}")
            txt_file.write(f"{category_line}\n")
            
            for i, res in enumerate(top_3, 1):
                line = f"  {i}. {res['accuracy']:.4f} | {res['model']} ({res['method']})\n"
                print(line, end="")
                txt_file.write(line)
            print() # Spacer
            txt_file.write("\n")

    footer = "=" * 70 + "\n"
    print(footer)
    txt_file.write(footer)

print(f"Summary saved to: {stats_output_path}")