

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.colors as mcolors

# custom red-only colormap (light -> dark red)
cmap = sns.color_palette("Reds_r", as_cmap=True)
# ==========================================
# 1. PASTE YOUR LATEX DATA HERE
# ==========================================
# Copy and paste your LaTeX table rows directly into this string.
# Do not worry about trailing backslashes (\\) or spaces; the script cleans them.
raw_data = """
qwen2.5-7b VP& 56.33 & 83.17 & 60.00 & 76.67 & 72.21 & 73.57
qwen2.5-7b CoT&  54.00 & 86.17 & 91.33 & 86.11 & 64.88 & 67.29
qwen2.5-7b Complete& 56.33 & 90.02 & 92.00 & 92.22 & 65.91 & 67.03
qwen2.5-7b EbE & 37.67 & 68.96 & 40.67 & 18.33 & 49.33 & 49.59

"""

copy = """qwen3-4b VP&   57.00 & 81.67 & 57.33 & 80.56 & 72.73 & 73.19
qwen3-4b CoT&    65.67 & 85.41 & 99.33 & 81.67 & 68.27 & 69.61
qwen3-4b Complete &  68.33 & 91.39 & 100.00 & 89.44 & 66.63 & 68.55
qwen3-4b EbE & 44.33 & 77.00 & 35.33 & 62.78 & 47.41 & 46.43
qwen2.5-7b VP& 56.33 & 83.17 & 60.00 & 76.67 & 72.21 & 73.57
qwen2.5-7b CoT&  54.00 & 86.17 & 91.33 & 86.11 & 64.88 & 67.29
qwen2.5-7b Complete& 56.33 & 90.02 & 92.00 & 92.22 & 65.91 & 67.03
qwen2.5-7b-7b EbE& 37.67 & 68.96 & 40.67 & 18.33 & 49.33 & 49.59
Gemma2-27b VP & 63.67 & 88.22 & 99.33 & 88.33 & 63.08 & 64.78
Gemma2-27b CoT & 62.33 & 91.87 & 94.67 & 96.11 & 64.70 & 65.17
Gemma2-27b Complete & 54.67 & 91.67 & 85.33 & 94.44 & 62.74 & 63.54
Gemma2-27b EbE & 47.00 & 75.07 & 41.33 & 23.89 & 55.20 & 55.18
Llama3.3-70b VP & 72.67 & 90.11 & 100.00 & 90.00 & 57.88 & 58.44
Llama3.3-70b CoT & 79.00 & 92.07 & 98.00 & 95.56 & 60.58 & 61.42
Llama3.3-70b Complete & 76.67 & 92.89 & 98.67 & 96.67 & 57.66 & 58.20
Llama3.3-70b EbE & 45.33 & 78.17 & 39.33 & 33.89 & 46.63 & 46.39"""


benchmarks = [
    "HiToM",
    "BigToM",
    "ToMC Sally-Anne",
    "ToMC Smarties",
    "ToMi val",
    "ToMi test",
]

rows = []

for line in raw_data.strip().split("\n"):
    parts = [p.strip() for p in line.split("&")]

    model_prompt = parts[0].split()
    model = model_prompt[0]
    prompt = model_prompt[1]

    scores = list(map(float, parts[1:]))

    row = {
        "Model": model,
        "Prompt": prompt,
    }

    for b, s in zip(benchmarks, scores):
        row[b] = s

    rows.append(row)

df = pd.DataFrame(rows)

print(df)


colors = {
    "VP": "#2E8B57",   # dark green
    "CoT": "#66C266",   # medium green
    "Complete": "#A5D6A7",   # light green
    "EbE": "#D62728",       # red
}



'''heatmap_data = df.groupby("Prompt")[benchmarks].mean()

# Optional: reorder rows
order = ["VP", "CoT", "Complete", "EbE"]
heatmap_data = heatmap_data.loc[order]

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(8, 3.8))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",   # red -> yellow -> green
    linewidths=1,
    linecolor='white',
    cbar_kws={"label": "Accuracy (%)"}
)

# =========================================================
# STYLE
# =========================================================

plt.title(
    "Prompt Performance Across Benchmarks",
    fontsize=14,
    pad=12
)

plt.xlabel("")
plt.ylabel("")

plt.yticks(rotation=0)

# =========================================================
# STAGGER X LABELS
# =========================================================

ax = plt.gca()

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0
)

for i, label in enumerate(ax.get_xticklabels()):

    if i % 2 == 0:
        label.set_y(-0.02)
    else:
        label.set_y(-0.08)

plt.tight_layout()

# =========================================================
# SAVE
# =========================================================

plt.savefig(
    "prompt_heatmap.png",
    dpi=400,
    bbox_inches='tight'
)

plt.close()'''


# AVERAGE ACROSS MODELS (Llama3 + GPT4)
# =========================================================

"""grouped = df.groupby("Prompt")[benchmarks].mean()

# keep consistent order
order = ["VP", "CoT", "Complete", "EbE"]
grouped = grouped.loc[order]

# =========================================================
# GAP vs BEST PROMPT (PER BENCHMARK)
# =========================================================

gap = grouped - grouped.max(axis=0)

# =========================================================
# PLOT (heatmap)
# =========================================================

plt.figure(figsize=(8, 3.8))

ax = sns.heatmap(
    gap,
    annot=True,
    fmt=".1f",

    cmap=cmap,

    vmin=-20,
    vmax=0,

    linewidths=1,
    linecolor="white",
    cbar_kws={"label": "Accuracy Gap vs Best"}
)

# =========================================================
# STYLE
# =========================================================


plt.title("Performance Gap Fully vs Partial observability", fontweight="bold", fontsize=13, pad=10)

plt.xlabel("Benchmarks", fontweight="bold")
plt.ylabel("Prompts", fontweight="bold")

plt.xticks(rotation=0)
plt.yticks(rotation=0)

ax = plt.gca()

# IMPORTANT: ensure labels are explicitly fetched from axis
labels = ax.get_xticklabels()

for i, label in enumerate(labels):
    if i % 2 == 0:
        label.set_y(-0.02)   # slightly higher
    else:
        label.set_y(-0.08)   # slightly lower

plt.tight_layout()

# =========================================================
# SAVE
# =========================================================

plt.savefig(
    "prompt_gap_heatmap.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close()"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================

order = ["VP", "CoT", "Complete", "EbE"]

# your colormap (keep yours)
# cmap = sns.light_palette("red", as_cmap=True, reverse=True)

# =========================================================
# LOOP OVER MODELS
# =========================================================

for model_name, df_model in df.groupby("Model"):

    # =====================================================
    # AVERAGE ACROSS PROMPT INSTANCES
    # =====================================================

    grouped = df_model.groupby("Prompt")[benchmarks].mean()

    grouped = grouped.loc[order]

    # =====================================================
    # GAP vs BEST
    # =====================================================

    gap = grouped - grouped.max(axis=0)

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(8, 3.8))

    ax = sns.heatmap(
        gap,
        annot=True,
        fmt=".1f",

        cmap=cmap,

        vmin=-20,
        vmax=0,

        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Accuracy Gap vs Best"}
    )

    # =====================================================
    # STYLE
    # =====================================================

    plt.title(
        f"{model_name}: Prompt Performance Gap",
        fontweight="bold",
        fontsize=13,
        pad=10
    )

    plt.xlabel("Benchmarks", fontweight="bold")
    plt.ylabel("Prompts", fontweight="bold")

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    ax = plt.gca()

    labels = ax.get_xticklabels()

    for i, label in enumerate(labels):
        if i % 2 == 0:
            label.set_y(-0.02)
        else:
            label.set_y(-0.08)

    plt.tight_layout()

    # =====================================================
    # SAVE PER MODEL
    # =====================================================

    filename = f"prompt_gap_heatmap_{model_name}.png"

    plt.savefig(
        filename,
        dpi=400,
        bbox_inches="tight"
    )

    plt.close()