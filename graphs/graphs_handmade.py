import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def save_percentage_graph(data_dict, labels, filename="graph.png", title="TITLE"):    
    plt.figure(figsize=(10, 6))

    # Loop through the dictionary to plot each list
    for label, values in data_dict.items():
        plt.plot(labels, values, marker='o', label=label, linewidth=2)

    # Formatting the Y-axis to show '%' symbol
    fmt = '%.0f%%' # Change to '%.1f%%' if you want decimals
    xticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(xticks)

    # Adding aesthetic touches
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100) # Assuming 0-100 range
    plt.legend()
    
    plt.tight_layout()
    #plt.show()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved as {filename}")
    
    # Close the plot to free up memory
    plt.close()

# --- Example Usage ---
data = {
    "Qwen3-4b": [85.83 , 100.00 , 80.84 , 79.22],
    "Qwen3-4b CoT": [100.00 , 100.00 , 60.12 , 60.90],
    "Qwen3-4b Complete": [ 99.58,100.00, 56.04, 58.44],
    "Qwen3-4b EbE": [ 71.67,86.67, 56.58, 54.65],
    "Qwen3-4b EbE TS": [ 74.17,86.67, 60.48, 56.94],
    "Qwen3-4b EbE OBJ": [ 66.67,26.67, 90.15, 87.03],
    '''"Qwen2.5-7b": [ 53.33,100.00, 79.16, 69.61],
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
    "Gpt-oss-20b EbE OBJ": [ 83.33,96.67, 90.81, 88.11]'''
}

labels = ['HiToM 0', 'ToMC SA', 'ToMi val', 'ToMi test']
output_path = "Memory_test/qwen3.png"
title = "Accuracy on Memory test"

save_percentage_graph(data, labels, output_path, title)


'''
######  HiToM
data_HiToM = {
    #"Nemo": [54.17, 54.17, 44.17, 37.08, 31.67],
    #"Qwen3": [85.83, 54.17, 40.83, 40, 36.67],
    #"Qwen2.5": [53.33, 52.92, 45.42, 45, 48.75],
    #"llama3.1": [50.00, 45.83, 35.42, 26.25, 24.58],
    #"Gpt-oss": [99.17, 79.17, 48.33, 32.50, 18.33],
    #"Gemma2": [75.00, 60.83, 50.42, 48.33, 50.83],
    "llama3.3": [100.00, 70.83, 59.58, 47.50, 44.17],

    #"Nemo CoT": [86.67, 57.08, 44.17, 37.08, 35],
    #"Qwen3 CoT": [100, 75, 57.92, 49.17, 28.33],
    #"Qwen2.5 CoT": [87.50, 60.83, 43.75, 31.67, 31.67],
    #"llama3.1 CoT": [87.50, 67.92, 42.50, 31.25, 40.00],
    #"Gpt-oss CoT": [100.00, 79.58, 50.42, 27.08, 15.42],
    #"Gemma2 CoT": [93.75, 67.08, 48.33, 33.33, 35.42],
    "llama3.3 CoT": [100.00, 72.08, 67.50, 52.08, 51.67],

    #"Nemo Complete": [62.08, 45, 39.17, 38.33, 33.75],
    #"Qwen3 Complete": [99.58, 73.33, 60, 45.52, 28.33],
    #"Qwen2.5 Complete": [78.75, 54.17, 45.42, 36.25, 44.17],
    #"llama3.1 Complete": [68.75, 52.92, 33.33, 31.25, 35.00],
    #"Gpt-oss Complete": [98.33, 79.17, 49.58, 23.33, 10.00],
    #"Gemma2 Complete": [82.08, 60.83, 45.83, 41.67, 31.25],
    "llama3.3 Complete": [99.17, 63.33, 60.83, 56.67, 56.67],

    #"Nemo EbE": [15.00, 13.75, 12.08, 16.25, 12.50],
    #"Qwen3 EbE": [69.17, 39.58, 25.42, 31.25, 30.42],
    #"Qwen2.5 EbE": [59.17, 30.83, 22.08, 27.92, 25.17],
    #"Gpt-oss EbE": [7.08, 6.67, 8.33, 8.75, 10.00],
    #"Gemma2 EbE": [74.17, 41.67, 25.83, 32.08, 31.67],
    "llama3.3 EbE": [77.50, 53.33, 29.58, 28.75, 31.25],
}
labels_HiToM = ['Order 0', 'Order 1', 'Order 2', 'Order 3', 'Order 4']
output_path = "Single_confronts_HiToM/Llama3_3.png"
'''




'''
######  ToMi val
data_tomi = {
    #"Nemo": [74.29, 59.94, 63.42],
    #"Qwen3": [80.84, 61.62, 75.68],
    #"Qwen2.5": [79.16, 68.23, 69.25],
    #"llama3.1": [73.81, 64.80, 70.81],
    #"Gpt-oss": [73.33, 66.85, 75.44],
    #"Gemma2": [58.62, 65.47, 65.53],
    "llama3.3": [47.21, 69.25, 57.18],

    #"Nemo CoT": [56.40, 68.47, 60.66],
    #"Qwen3 CoT": [60.12, 73.63, 71.05],
    #"Qwen2.5 CoT": [53.75, 75.14, 66.67],
    #"llama3.1 CoT": [58.68, 62.58, 58.80],
    #"Gpt-oss CoT": [74.11, 69.19, 76.46],
    #"Gemma2 CoT": [58.56, 68.05, 66.19],
    "llama3.3 CoT": [51.47, 70.69, 59.52],

    #"Nemo Complete": [54.17, 69.25, 60.60],
    #"Qwen3 Complete": [56.04, 74.47, 68.95],
    #"Qwen2.5 Complete": [52.55, 79.22, 66.73],
    #"llama3.1 Complete": [61.02, 65.11, 59.58],
    #"Gpt-oss Complete": [66.61, 74.41, 68.59],
    #"Gemma2 Complete": [54.05, 72.37, 61.44],
    "llama3.3 Complete": [49.49, 66.37, 57.36],

    #"Nemo EbE": [39.22, 33.99, 6.67],
    #"Qwen3 EbE": [65.29, 46.43, 48.35],
    #"Qwen2.5 EbE": [52.61, 39.94, 45.35],
    #"lama3.1 EbE": [35.86, 27.33, 19.58],
    #"Gpt-oss EbE": [1.14, 2.22, 1.08],
    #"Gemma2 EbE": [66.91, 45.17, 57.36],
    "llama3.3 EbE": [53.99, 41.68, 45.05],
}

labels = ['True belief', 'False belief', 'Second order fb']
output_path = "Single_confronts_ToMi/llama3_3.png"
title = "Accuracy on ToMi(val) per ToM category"'''




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