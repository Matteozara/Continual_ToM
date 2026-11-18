import re
import json
from collections import defaultdict

def extract_summary_data(file_path):
    results = []
    
    # Regular expression patterns for each field
    patterns = {
        "question_type": r"Question Type:\s*(.*)",
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
                if "Question Type:" in block:
                    entry = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, block)
                        if match:
                            # Convert numbers to int/float, keep question_type as string
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
benchmark = "BigToM"
data = {}
models_gpu1 = ["qwen3-4b","qwen2.5-7b","llama3-8b","nemo-12b"]
models_gpu2 = ["gpt-20b","gemma2-27b","granite4-32b","llama3-70b", "qwen3.6-35b"]
ebe_type = ["standard"] #["standard", "obj_location", "task_specific"]
run_type = ["CoT", "Complete"] #["", "CoT", "Complete"]
output_path = f"{benchmark}_summary_gpu1.json"
stats_output_path = f"{benchmark}_top3_summary.txt"


for model in models_gpu2:
    for run in run_type:
        code = f"{model}_{run}"
        data[code] = extract_summary_data(f'./../../results/{benchmark}_{run}_all_{model}.txt')
    for ebe in ebe_type:
        code = f"{model}_{ebe}"
        data[code] = extract_summary_data(f'./../../results/{benchmark}_{ebe}_EbE_{model}.txt')


print(json.dumps(data, indent=4))

with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)





all_results = []
for source_key, questions in data.items():
    # Splitting the key back into model and ebe (assumes 'model_ebe' format)
    model_name, ebe_val = source_key.rsplit('_', 1) 
    for q in questions:
        all_results.append({
            "model": model_name,
            "ebe": ebe_val,
            "question_type": q["question_type"],
            "accuracy": q["accuracy"]
        })

# 2. Group by Question Type
grouped_by_type = defaultdict(list)
for record in all_results:
    grouped_by_type[record["question_type"]].append(record)

with open(stats_output_path, "w", encoding="utf-8") as txt_file:
    header = "="*60 + f"\nTOP 3 ACCURACIES PER CATEGORY ({benchmark})\n" + "="*60 + "\n"
    print(header)
    txt_file.write(header)

    for q_type, results in grouped_by_type.items():
        # Sort by accuracy descending and take top 3
        top_3 = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:3]
        
        category_header = f"\nCategory: {q_type}\n"
        print(category_header, end="")
        txt_file.write(category_header)
        
        for i, res in enumerate(top_3, 1):
            line = f"  {i}. {res['accuracy']:.4f} | {res['model']} ({res['ebe']})\n"
            print(line, end="")
            txt_file.write(line)

    footer = "\n" + "="*60 + "\n"
    print(footer)
    txt_file.write(footer)

print(f"Summary saved to: {stats_output_path}")