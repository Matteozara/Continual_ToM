import re

# Your input data (LaTeX table rows)
data = """
qwen3-4b ToMi reality standard & 10.51 & 24.47 & 0.00 \\
qwen3-4b ToMi reality obj location & 2.15 & 4.35 & 34.44 \\
qwen3-4b ToMi reality obj location NEW & 0.67 & 1.20 & 36.29 \\
qwen2.5-7b ToMi reality standard & 14.36 & 33.18 & 0.00 \\
qwen2.5-7b ToMi reality obj location & 37.99 & 52.95 & 22.79 \\
qwen2.5-7b ToMi reality obj location NEW & 1.17 & 2.75 & 36.32 \\
llama3-8b ToMi reality standard & 18.22 & 37.39 & 0.80 \\
llama3-8b ToMi reality obj location & 10.11 & 39.19 & 45.00 \\
llama3-8b ToMi reality obj location NEW & 26.30 & 76.13 & 62.03 \\
nemo-12b ToMi reality standard & 46.27 & 84.33 & 0.31 \\
nemo-12b ToMi reality obj location & 1.00 & 2.10 & 37.03 \\
nemo-12b ToMi reality obj location NEW & 0.09 & 0.25 & 38.55 \\
gpt-20b ToMi reality standard & 20.86 & 52.35 & 0.14 \\
gpt-20b ToMi reality obj location & 0.59 & 0.95 & 34.95 \\
gpt-20b ToMi reality obj location NEW & 0.57 & 1.00 & 35.39 \\
gemma2-27b ToMi reality standard & 9.97 & 27.13 & 0.00 \\
gemma2-27b ToMi reality obj location & 0.11 & 0.20 & 35.80 \\
gemma2-27b ToMi reality obj location NEW & 0.06 & 0.15 & 35.83 \\
llama3-70b ToMi reality standard & 4.20 & 10.76 & 0.00 \\
llama3-70b ToMi reality obj location & 3.46 & 6.06 & 29.99 \\
llama3-70b ToMi reality obj location NEW & 0.56 & 0.85 & 33.65 \\
qwen3-30b ToMi reality standard & 5.45 & 14.06 & 0.00 \\
qwen3-30b ToMi reality obj location & 9.36 & 25.83 & 34.70 \\
qwen3-30b ToMi reality obj location NEW & 0.04 & 0.15 & 36.28 \\
qwen2.5-72b ToMi reality standard & 2.56 & 7.61 & 0.00 \\
qwen2.5-72b ToMi reality obj location & 0.04 & 0.10 & 35.70 \\
qwen2.5-72b ToMi reality obj location NEW & 0.02 & 0.05 & 35.84 \\
"""

def invert_percentages(text_data):
    # Regex pattern to capture the text before the first '&', and the first two numbers
    pattern = r"^([^&]+)&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&(.*)$"
    
    def replace_func(match):
        prefix = match.group(1)
        num1 = float(match.group(2))
        num2 = float(match.group(3))
        rest = match.group(4)
        
        # Invert the percentages
        inv1 = 100.0 - num1
        inv2 = 100.0 - num2
        
        # Format back to 2 decimal places
        return f"{prefix}& {inv1:.2f} & {inv2:.2f} &{rest}"

    processed_lines = []
    for line in text_data.strip().split('\n'):
        # Apply the substitution line by line
        new_line = re.sub(pattern, replace_func, line)
        processed_lines.append(new_line)
        
    return '\n'.join(processed_lines)

# Run the function and print results
output = invert_percentages(data)
print(output)