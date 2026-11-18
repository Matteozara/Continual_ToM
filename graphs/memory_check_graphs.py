import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def save_percentage_graph(data_dict, labels, filename="graph.png", title="TITLE"):    
    plt.figure(figsize=(10, 6))

    # Define our color sets
    # Using hex codes for better control over "light" vs "dark"
    light_colors = ['#ADD8E6', '#90EE90', '#FFB6C1'] # Light Blue, Light Green, Light Pink
    dark_colors = ['#00008B', '#006400', '#8B0000']  # Dark Blue, Dark Green, Dark Red

    # Loop through the dictionary with an index
    for i, (label, values) in enumerate(data_dict.items()):
        # Determine color based on index
        if i < 3:
            current_color = light_colors[i % len(light_colors)]
        else:
            # This handles the "last three" logic assuming roughly 6 entries
            # or just applies dark colors to everything after the first three
            current_color = dark_colors[(i - 3) % len(dark_colors)]

        plt.plot(labels, values, marker='o', label=label, linewidth=2, color=current_color)

    # Formatting the Y-axis to show '%' symbol
    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(xticks)

    # Adding aesthetic touches
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved as {filename}")
    
    plt.close()

# --- Example Usage ---
data = {
    "Gpt-oss-20b": [ 99.17,100.00, 73.33, 72.25],
    "Gpt-oss-20b CoT": [ 100.00,100.00, 74.11, 73.69],
    "Gpt-oss-20b Complete": [ 98.33,100.00, 66.61, 67.03],
    "Gpt-oss-20b EbE": [ 87.50,96.67, 79.04, 75.44],
    "Gpt-oss-20b EbE TS": [ 80.00,86.67, 77.66, 75.56],
    "Gpt-oss-20b EbE OBJ": [ 83.33,96.67, 90.81, 88.11]
    
}

labels = ['HiToM 0', 'ToMC SA', 'ToMi val', 'ToMi test']
output_path = "Memory_test/gpt_oss.png"
title = "Accuracy on Memory test"

save_percentage_graph(data, labels, output_path, title)




'''
######  Memory test (data HiToM 0, ToMC SA, ToMi val, ToMi test)
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
output_path = "Memory_test/Llama3_3.png"'''