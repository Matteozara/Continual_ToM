import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Prepare the Data
# Rows = Benchmarks, Columns = Models
data = {
    "Gpt-oss-20b": [ 99.17,100.00, 73.33, 72.25],
    "Gpt-oss-20b CoT": [ 100.00,100.00, 74.11, 73.69],
    "Gpt-oss-20b Complete": [ 98.33,100.00, 66.61, 67.03],
    "Gpt-oss-20b EbE": [ 87.50,96.67, 79.04, 75.44],
    "Gpt-oss-20b EbE TS": [ 80.00,86.67, 77.66, 75.56],
    "Gpt-oss-20b EbE OBJ": [ 83.33,96.67, 90.81, 88.11]
}
output_path = "Memory_test/bar_gpt_oss.png"
title = "Accuracy on Memory test"
benchmarks = ['HiToM 0', 'ToMC SA reality', 'ToMi val tb', 'ToMi test tb']
df = pd.DataFrame(data, index=benchmarks)

# 2. Setup Plotting Parameters
models = df.columns
x = np.arange(len(benchmarks))  # Label locations
width = 0.12  # Width of each individual bar

# 3. Create the Visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Iterate through models and plot their respective bars with an offset
for i, model in enumerate(models):
    # Calculate the offset for each bar in the group
    offset = (i - len(models)/2 + 0.5) * width
    ax.bar(x + offset, df[model], width, label=model)

# 4. Add Styling and Labels
ax.set_xlabel('Benchmarks', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
ax.set_title(title, fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)

# Place legend outside to avoid overlapping bars
ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add gridlines for easier reading
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# 5. Save and Display
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print("Graph saved as 'model_comparison_chart.png'")




'''
memory_test = {
    "Qwen3-4b": [85.83 , 100.00 , 80.84 , 79.22],
    "Qwen3-4b CoT": [100.00 , 100.00 , 60.12 , 60.90],
    "Qwen3-4b Complete": [ 99.58,100.00, 56.04, 58.44],
    "Qwen3-4b EbE": [ 71.67,86.67, 56.58, 54.65],
    "Qwen3-4b EbE TS": [ 74.17,86.67, 60.48, 56.94],
    "Qwen3-4b EbE OBJ": [ 66.67,26.67, 90.15, 87.03],

    "Qwen2.5-7b": [ 53.33,100.00, 79.16, 69.61],
    "Qwen2.5-7b CoT": [ 87.50,100.00, 53.75, 56.94],
    "Qwen2.5-7b Complete": [ 78.75,100.00, 52.55, 55.14],
    "Qwen2.5-7b EbE": [ 55.42,93.33, 57.12, 55.08],
    "Qwen2.5-7b EbE TS": [ 67.08,86.67, 62.04, 62.70],
    "Qwen2.5-7b EbE OBJ": [ 65.42,100.00, 78.20, 78.44],

    "Llama3-8b": [ 50.00,100.00, 73.81, 73.51],
    "Llama3-8b CoT": [ 87.50,90.00, 58.68, 61.50],
    "Llama3-8b Complete": [ 68.75,93.33, 61.02, 60.72],
    "Llama3-8b EbE": [ 51.25,90.00, 63.48, 62.46],
    "Llama3-8b EbE TS": [ 49.58,86.67, 64.26, 61.32],
    "Llama3-8b EbE OBJ": [ 33.33,96.67, 76.52, 72.55],

    "Nemo-12b": [ 54.17,100.00, 74.29, 72.49],
    "Nemo-12b CoT": [ 86.67,100.00, 56.40, 56.16],
    "Nemo-12b Complete": [ 62.08,100.00, 54.17, 58.56],
    "Nemo-12b EbE": [ 21.25,83.33, 41.44, 39.04],
    "Nemo-12b EbE TS": [ 19.17,93.33, 41.38, 36.28],
    "Nemo-12b EbE OBJ": [ 38.33,100.00, 90.51, 86.61],

    "Gpt-oss-20b": [ 99.17,100.00, 73.33, 72.25],
    "Gpt-oss-20b CoT": [ 100.00,100.00, 74.11, 73.69],
    "Gpt-oss-20b Complete": [ 98.33,100.00, 66.61, 67.03],
    "Gpt-oss-20b EbE": [ 87.50,96.67, 79.04, 75.44],
    "Gpt-oss-20b EbE TS": [ 80.00,86.67, 77.66, 75.56],
    "Gpt-oss-20b EbE OBJ": [ 83.33,96.67, 90.81, 88.11]
}


labels_memory_test = ['HiToM 0', 'ToMC SA', 'ToMi val', 'ToMi test']
output_path = "Memory_test/Llama3_3.png"
'''