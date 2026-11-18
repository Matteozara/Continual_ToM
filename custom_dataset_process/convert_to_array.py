# Your input text
text_data = """gpt-20b_standard_ebe & 61.33
gpt-20b_task_specific_ebe & 58.67
gpt-20b_standard_gold_memory & 96.00
gemma2-27b_standard_ebe & 42.00
gemma2-27b_task_specific_ebe & 43.33
gemma2-27b_standard_gold_memory & 100.00
llama3-70b_standard_ebe & 41.33
llama3-70b_task_specific_ebe & 36.00
llama3-70b_standard_gold_memory & 93.33
qwen3.6-35b_standard_ebe & 60.00
qwen3.6-35b_task_specific_ebe & 57.33
qwen3.6-35b_standard_gold_memory & 100.00"""

# Dictionary to hold the grouped data (now only holding 2 items: standard and gold)
merged_data = {}

for line in text_data.strip().split("\n"):
    if not line:
        continue

    identifier, value = [item.strip() for item in line.split("&")]

    if "_standard_ebe" in identifier:
        model = identifier.replace("_standard_ebe", "")
        index = 0
    elif "_standard_gold_memory" in identifier:
        model = identifier.replace("_standard_gold_memory", "")
        index = 1
    else:
        # This automatically skips "_task_specific_ebe" and any other patterns
        continue

    if model not in merged_data:
        merged_data[model] = [None, None]

    merged_data[model][index] = value

# Convert the filtered dictionary into the LaTeX string format
for model, values in merged_data.items():
    values_str = " & ".join(values)
    latex_row = f"{model} & {values_str} \\\\"
    print(latex_row)