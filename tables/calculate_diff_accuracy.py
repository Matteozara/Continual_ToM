import re

def calculate_accuracy_difference(data_string):
    # Split by lines and filter out empty strings
    lines = [line.strip() for line in data_string.strip().split('\n')]
    
    parsed_data = []
    for line in lines:
        # Split by '&' and clean whitespace
        parts = [p.strip() for p in line.split('&')]
        label = parts[0]
        # Convert the remaining parts to floats
        values = [float(x) for x in parts[1:]]
        parsed_data.append((label, values))

    # Process pairs (CoT followed by Standard)
    for i in range(0, len(parsed_data), 2):
        cot_label, cot_vals = parsed_data[i]
        std_label, std_vals = parsed_data[i+1]
        
        # Calculate difference (Standard - CoT)
        diffs = [std - cot for cot, std in zip(cot_vals, std_vals)]
        
        # Clean the label (remove _CoT or _standard)
        base_label = cot_label.replace('_CoT', '').replace('_standard', '')
        
        # Format each difference
        formatted_diffs = []
        for d in diffs:
            # Data is already 0-100, so we don't multiply by 100 here
            color = "red" if d < 0 else "blue"
            formatted_diffs.append(f"\\textcolor{{{color}}}{{{d:+.2f}%}}")
        
        # Join with ampersands for LaTeX table rows
        latex_row = f"Difference {base_label} & " + " & ".join(formatted_diffs) + " \\\\"
        print(latex_row)

# Example input
raw_data = """
qwen3-30b_CoT & 72.33 & 91.00 & 100.00 & 86.67 & 66.03 & 67.43
qwen3-30b_standard & 48.00 & 82.48 & 28.67 & 40.00 & 64.30 & 63.98
qwen2.5-72b_CoT & 72.33 & 92.07 & 100.00 & 90.56 & 70.83 & 70.89
qwen2.5-72b_standard & 48.67 & 83.30 & 46.00 & 49.44 & 61.88 & 61.64
"""

if __name__ == "__main__":
    calculate_accuracy_difference(raw_data)


