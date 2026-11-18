import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

def save_percentage_bar_graph(data_dict, labels, filename="bar_graph.png", title="TITLE"):    
    # Set up the positions for the groups
    x = np.arange(len(labels))  # The label locations [0, 1, 2, 3, 4]
    width = 0.2  # The width of the individual bars
    
    plt.figure(figsize=(10, 6))

    # Plot each set of data
    # We multiply the width by the index to shift each group's bar to the right
    for i, (label, values) in enumerate(data_dict.items()):
        offset = i * width
        plt.bar(x + offset, values, width, label=label)

    # Formatting the Y-axis for '%'
    fmt = '%.0f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(yticks)

    # Centering the X-axis labels between the groups of bars
    # Calculation: x + (half the total width of all bars in a group)
    plt.xticks(x + (width * (len(data_dict) - 1) / 2), labels)

    # Aesthetic touches
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Grid only on Y-axis for bars
    plt.ylim(0, 100)
    plt.legend()
    
    plt.tight_layout()

    # Save the file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bar graph successfully saved as {filename}")

# --- Example Usage ---

data_HiToM = {
    "qwen3.6-35b_CoT": [73.33, 80.0, 80.0, 71.67],
    "qwen3.6-35b_Complete": [76.67, 91.67, 75.0, 65.0],
    "qwen3.6-35b_standard": [48.33, 40.0, 41.67, 36.67]
      
}

labels_HiToM = ['Order 1', 'Order 2', 'Order 3', 'Order 4']

save_percentage_bar_graph(data_HiToM, labels_HiToM, "HiToM_qwenVL_CCE.png", "Accuracy on HiToM per higher ToM orders for Qwen3.6-35b VL")








'''
#data CoT - Complete - EbE (FB, 2nd FB) [No TB] ToMi


data_ToMi = {
    "Qwen3-4b CoT (val)": [71.83, 60.0],
    "Qwen3-4b Complete (val)": [68.59, 56.94],
    "Qwen3-4b EbE (val)": [45.23, 56.58],
    "Qwen3-4b CoT (test)": [71.29, 61.56],
    "Qwen3-4b Complete (test)": [70.45, 58.92],
    "Qwen3-4b EbE (test)": [43.3, 54.95],
    "Qwen2.5-7b CoT (val)": [66.37, 53.51],
    "Qwen2.5-7b Complete (val)": [66.37, 52.61],
    "Qwen2.5-7b EbE (val)": [47.27, 57.24],
    "Qwen2.5-7b CoT (test)": [67.39, 56.82],
    "Qwen2.5-7b Complete (test)": [66.31, 55.26],
    "Qwen2.5-7b EbE (test)": [48.11, 55.14],
    "Llama3-8b CoT (val)": [58.62, 59.04],
    "Llama3-8b Complete (val)": [60.06, 60.9],
    "Llama3-8b EbE (val)": [44.38, 63.6],
    "Llama3-8b CoT (test)": [60.42, 61.2],
    "Llama3-8b Complete (test)": [61.74, 60.84],
    "Llama3-8b EbE (test)": [45.77, 61.98],
    "Nemo-12b CoT (val)": [60.6, 56.4],
    "Nemo-12b Complete (val)": [60.36, 54.11],
    "Nemo-12b EbE (val)": [13.93, 41.62],
    "Nemo-12b CoT (test)": [59.58, 56.1],
    "Nemo-12b Complete (test)": [57.66, 58.14],
    "Nemo-12b EbE (test)": [12.25, 38.98]
}


labels_ToMi = ["FB", "2nd FB"]

'''






'''
#data CoT - Complete - EbE (1,2,3,4 order) [no 0] HiToM
data_HiToM = {
    "Qwen3-4b CoT": [73.33, 75.0, 53.33, 26.67],
    "Qwen3-4b Complete": [71.67, 68.33, 58.33, 43.33],
    "Qwen3-4b EbE": [40.0, 31.67, 35.0, 40.0],
    "Qwen2.5-7b CoT": [61.67, 50.0, 38.33, 36.67],
    "Qwen2.5-7b Complete": [55.0, 55.0, 43.33, 56.67],
    "Qwen2.5-7b EbE": [41.67, 20.0, 30.0, 35.0],
    "Llama3-8b CoT": [70.0, 55.0, 33.33, 43.33],
    "Llama3-8b Complete": [58.33, 41.67, 36.67, 46.67],
    "Llama3-8b EbE": [31.67, 20.0, 23.33, 25.0],
    "Nemo-12b CoT": [60.0, 55.0, 43.33, 36.67],
    "Nemo-12b Complete": [46.67, 48.33, 41.67, 38.33],
    "Nemo-12b EbE": [11.67, 13.33, 18.33, 18.33],

    "gpt-20b_CoT": [78.33, 50.0, 25.0, 18.33],
    "gpt-20b_Complete": [71.67, 55.0, 23.33, 13.33],
    "gpt-20b_standard": [40.0, 26.67, 25.0, 30.0],
    "gemma2-27b_CoT": [70.0, 65.0, 36.67, 45.0],
    "gemma2-27b_Complete": [68.33, 48.33, 46.67, 30.0],
    "gemma2-27b_standard": [46.67, 30.0, 36.67, 41.67],
    "granite4-32b_CoT": [58.33, 48.33, 48.33, 45.0],
    "granite4-32b_Complete": [43.33, 46.67, 30.0, 43.33],
    "granite4-32b_standard": [13.33, 11.67, 15.0, 20.0],
    "llama3-70b_CoT": [73.33, 81.67, 73.33, 66.67],
    "llama3-70b_Complete": [68.33, 76.67, 68.33, 71.67],
    "llama3-70b_standard": [48.33, 25.0, 31.67, 33.33],
    "qwen3.6-35b_CoT": [73.33, 80.0, 80.0, 71.67],
    "qwen3.6-35b_Complete": [76.67, 91.67, 75.0, 65.0],
    "qwen3.6-35b_standard": [48.33, 40.0, 41.67, 36.67]
}

labels_HiToM = ['Order 1', 'Order 2', 'Order 3', 'Order 4']
'''






'''
#old data
data_HiToM = {
    "Nemo": [54.17, 54.17, 44.17, 37.08, 31.67],
    #"Qwen3": [85.83, 54.17, 40.83, 40, 36.67],
    #"Qwen2.5": [53.33, 52.92, 45.42, 45, 48.75],

    "Nemo CoT": [86.67, 57.08, 44.17, 37.08, 35],
    #"Qwen3 CoT": [100, 75, 57.92, 49.17, 28.33],
    #"Qwen2.5 CoT": [87.50, 60.83, 43.75, 31.67, 31.67],

    "Nemo Complete": [62.08, 45, 39.17, 38.33, 33.75],
    #"Qwen3 Complete": [99.58, 73.33, 60, 45.52, 28.33],
    #"Qwen2.5 Complete": [78.75, 54.17, 45.42, 36.25, 44.17]
}
'''