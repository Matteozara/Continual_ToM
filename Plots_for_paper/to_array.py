import numpy as np
import pandas as pd

# 1. Your raw data
raw_input_data = """
qwen3-4b VP&   57.00 & 81.67 & 57.33 & 80.56 & 72.73 & 73.19
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
Llama3.3-70b EbE & 45.33 & 78.17 & 39.33 & 33.89 & 46.63 & 46.39
"""
# 2. Parse into a list of rows
parsed_matrix = []
for line in raw_input_data.strip().split("\n"):
    if line.strip():
        row = [item.strip() for item in line.split("&")]
        parsed_matrix.append(row)



print(parsed_matrix)
