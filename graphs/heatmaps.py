import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# LOAD JSON
# -------------------------


'''#ToMC Sally-Anne
data = {
    #"Sally-Anne_qwen3-4b_": [100.0, 0.0, 80.0, 6.67, 100.0],
    "Sally-Anne_qwen3-4b_CoT": [100.0, 100.0, 100.0, 96.67, 100.0],
    #"Sally-Anne_qwen3-4b_Complete": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_qwen3-4b_EbE": [83.33, 0.0, 3.33, 3.33, 86.67],
    #"Sally-Anne_qwen3-4b_obj_location": [46.67, 3.33, 3.33, 3.33, 26.67],
    #"Sally-Anne_qwen3-4b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_qwen2.5-7b_": [100.0, 10.0, 30.0, 60.0, 100.0],
    "Sally-Anne_qwen2.5-7b_CoT": [66.67, 96.67, 96.67, 96.67, 100.0],
    #"Sally-Anne_qwen2.5-7b_Complete": [63.33, 100.0, 96.67, 100.0, 100.0],
    "Sally-Anne_qwen2.5-7b_EbE": [86.67, 23.33, 0.0, 0.0, 93.33],
    #"Sally-Anne_qwen2.5-7b_obj_location": [96.67, 3.33, 3.33, 3.33, 100.0],
    #"Sally-Anne_qwen2.5-7b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_llama3-8b_": [100.0, 6.67, 20.0, 0.0, 100.0],
    "Sally-Anne_llama3-8b_CoT": [73.33, 93.33, 90.0, 100.0, 86.67],
    #"Sally-Anne_llama3-8b_Complete": [50.0, 66.67, 70.0, 46.67, 93.33],
    "Sally-Anne_llama3-8b_EbE": [90.0, 10.0, 10.0, 13.33, 96.67],
    #"Sally-Anne_llama3-8b_obj_location": [83.33, 0.0, 0.0, 0.0, 96.67],
    #"Sally-Anne_llama3-8b_obj_location_NEW": [73.33, 6.67, 3.33, 0.0, 76.67],
    #"Sally-Anne_nemo-12b_": [100.0, 93.33, 83.33, 83.33, 100.0],
    "Sally-Anne_nemo-12b_CoT": [100.0, 100.0, 96.67, 96.67, 100.0],
    #"Sally-Anne_nemo-12b_Complete": [90.0, 93.33, 76.67, 90.0, 100.0],
    "Sally-Anne_nemo-12b_EbE": [83.33, 3.33, 3.33, 3.33, 83.33],
    #"Sally-Anne_nemo-12b_obj_location": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_nemo-12b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_gpt-20b_": [100.0, 100.0, 93.33, 80.0, 100.0],
    "Sally-Anne_gpt-20b_CoT": [100.0, 96.67, 90.0, 86.67, 100.0],
    #"Sally-Anne_gpt-20b_Complete": [93.33, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_gpt-20b_EbE": [96.67, 90.0, 86.67, 10.0, 96.67],
    #"Sally-Anne_gpt-20b_obj_location": [96.67, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_gpt-20b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_gemma2-27b_": [100.0, 100.0, 96.67, 100.0, 100.0],
    "Sally-Anne_gemma2-27b_CoT": [73.33, 100.0, 100.0, 100.0, 100.0],
    #"Sally-Anne_gemma2-27b_Complete": [60.0, 90.0, 76.67, 100.0, 100.0],
    "Sally-Anne_gemma2-27b_EbE": [93.33, 6.67, 6.67, 6.67, 93.33],
    #"Sally-Anne_gemma2-27b_obj_location": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_gemma2-27b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_llama3-70b_": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_llama3-70b_CoT": [90.0, 100.0, 100.0, 100.0, 100.0],
    #"Sally-Anne_llama3-70b_Complete": [100.0, 100.0, 100.0, 93.33, 100.0],
    "Sally-Anne_llama3-70b_EbE": [100.0, 10.0, 10.0, 0.0, 76.67],
    #"Sally-Anne_llama3-70b_obj_location": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_llama3-70b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0],
    #"Sally-Anne_qwen3.6-35b_": [100.0, 100.0, 100.0, 76.67, 100.0],
    "Sally-Anne_qwen3.6-35b_CoT": [100.0, 100.0, 100.0, 86.67, 100.0],
    #"Sally-Anne_qwen3.6-35b_Complete": [100.0, 100.0, 100.0, 90.0, 100.0],
    "Sally-Anne_qwen3.6-35b_EbE": [100.0, 73.33, 70.0, 10.0, 96.67],
    #"Sally-Anne_qwen3.6-35b_obj_location": [100.0, 0.0, 0.0, 6.67, 100.0],
    #"Sally-Anne_qwen3.6-35b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0]
}'''

#ToMC Sally-Anne selected
data = {
    
    "Sally-Anne_qwen2.5-7b_CoT": [66.67, 96.67, 96.67, 96.67, 100.0],
    #"Sally-Anne_qwen2.5-7b_Complete": [63.33, 100.0, 96.67, 100.0, 100.0],
    "Sally-Anne_qwen2.5-7b_EbE": [86.67, 23.33, 0.0, 0.0, 93.33],
    "Sally-Anne_gemma2-27b_CoT": [73.33, 100.0, 100.0, 100.0, 100.0],
    #"Sally-Anne_gemma2-27b_Complete": [60.0, 90.0, 76.67, 100.0, 100.0],
    "Sally-Anne_gemma2-27b_EbE": [93.33, 6.67, 6.67, 6.67, 93.33],
    "Sally-Anne_llama3-70b_CoT": [90.0, 100.0, 100.0, 100.0, 100.0],
    #"Sally-Anne_llama3-70b_Complete": [100.0, 100.0, 100.0, 93.33, 100.0],
    "Sally-Anne_llama3-70b_EbE": [100.0, 10.0, 10.0, 0.0, 76.67],
    
}


#ToMi
'''data = {
    "val_qwen3-4b_CoT": [72.97, 71.83, 60.0],
    "val_qwen3-4b_EbE": [40.42, 45.23, 56.58],
    "test_qwen3-4b_CoT": [75.98, 71.29, 61.56],
    "test_qwen3-4b_EbE": [41.02, 43.3, 54.95],
    "val_qwen2.5-7b_CoT": [74.77, 66.37, 53.51],
    "val_qwen2.5-7b_EbE": [43.48, 47.27, 57.24],
    "test_qwen2.5-7b_CoT": [77.66, 67.39, 56.82],
    "test_qwen2.5-7b_EbE": [45.53, 48.11, 55.14],
    "val_llama3-8b_CoT": [62.52, 58.62, 59.04],
    "val_llama3-8b_EbE": [44.32, 44.38, 63.6],
    "test_llama3-8b_CoT": [64.14, 60.42, 61.2],
    "test_llama3-8b_EbE": [44.56, 45.77, 61.98],
    "val_nemo-12b_CoT": [68.59, 60.6, 56.4],
    "val_nemo-12b_EbE": [34.71, 13.93, 41.62],
    "test_nemo-12b_CoT": [68.05, 59.58, 56.1],
    "test_nemo-12b_EbE": [32.07, 12.25, 38.98],
    "val_gpt-20b_CoT": [69.25, 77.06, 72.85],
    "val_gpt-20b_EbE": [50.93, 65.47, 78.08],
    "test_gpt-20b_CoT": [69.79, 77.9, 74.47],
    "test_gpt-20b_EbE": [52.31, 66.55, 76.34],
    "val_gemma2-27b_CoT": [68.83, 66.43, 58.86],
    "val_gemma2-27b_EbE": [45.29, 56.88, 63.42],
    "test_gemma2-27b_CoT": [70.87, 65.35, 59.28],
    "test_gemma2-27b_EbE": [45.17, 58.26, 62.1],
    "val_llama3-70b_CoT": [70.69, 59.58, 51.47],
    "val_llama3-70b_EbE": [41.56, 45.29, 53.03],
    "test_llama3-70b_CoT": [72.01, 59.22, 53.03],
    "test_llama3-70b_EbE": [40.84, 47.57, 50.75]
}'''

# -------------------------
# CREATE DATAFRAME
# -------------------------

#ToMC SA
classes = ["1stA", "1stB", "2ndA", "2ndB", "reality"]

#ToMi
#classes = ["false_belief", "2nd_order_false_belief", "true_belief"]


parsed_labels = []
parsed_values = []

for key, values in data.items():

    parts = key.split("_")

    benchmark = parts[0]
    prompt = parts[-1]

    model = "_".join(parts[1:-1])

    # Cleaner label
    label = f"{model}\n{prompt}"

    parsed_labels.append(label)
    parsed_values.append(values)

# ==========================================================
# CREATE DATAFRAME
# ==========================================================

df = pd.DataFrame(
    parsed_values,
    index=parsed_labels,
    columns=classes
)

# Sort columns alphabetically
df = df.sort_index()

# TRANSPOSE:
# rows -> classes
# columns -> models/prompts
heatmap_data = df.T

# ==========================================================
# FIGURE SIZE
# ==========================================================

n_cols = len(heatmap_data.columns)

fig_width = max(3.5, n_cols * 0.25)

plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(
    figsize=(fig_width, 2.8)
)

# ==========================================================
# HEATMAP
# ==========================================================

im = ax.imshow(
    heatmap_data.values,
    cmap="viridis",
    aspect="auto",
    vmin=0,
    vmax=100
)

# ==========================================================
# AXES
# ==========================================================

# X axis -> models
ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(
    heatmap_data.columns,
    fontsize=4,
    rotation=45,
    ha="right"
)

# Y axis -> classes
ax.set_yticks(np.arange(len(classes)))
ax.set_yticklabels(
    classes,
    fontsize=4
)

# ==========================================================
# CELL VALUES
# ==========================================================

for i in range(len(classes)):
    for j in range(n_cols):

        value = heatmap_data.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white" if value < 70 else "black"
        )

# ==========================================================
# COLORBAR
# ==========================================================

cbar = fig.colorbar(im, ax=ax)

cbar.ax.tick_params(labelsize=9)

cbar.set_label(
    "Accuracy (%)",
    fontsize=3
)

# ==========================================================
# TITLE
# ==========================================================

ax.set_title(
    "Accuracy on ToM Challenges SA per class",
    fontsize=8,
    pad=20
)



# ==========================================================
# LAYOUT
# ==========================================================

plt.tight_layout()

# ==========================================================
# SAVE
# ==========================================================

plt.savefig(
    "benchmark_heatmap_flipped.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "benchmark_heatmap_flipped.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "benchmark_heatmap_flipped.svg",
    bbox_inches="tight"
)

plt.close()