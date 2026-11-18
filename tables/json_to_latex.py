'''
script to convert json t latex
'''

import json

# Your input data
data = {
    "qwen3-30b_standard": [80.0, 38.33, 40.0, 41.67, 40.0],
    "qwen3-30b_task_specific": [80.0, 43.33, 50.0, 46.67, 43.33],
    "qwen2.5-72b_standard": [95.0, 45.0, 31.67, 33.33, 38.33],
    "qwen2.5-72b_task_specific": [90.0, 53.33, 36.67, 35.0, 38.33]
}

def format_json_data(json_input):
    formatted_lines = []
    
    for key, values in json_input.items():
        # 1. Capitalize the first letter
        model_name = key[0].upper() + key[1:]
        
        # 2. Remove the first number (index 0)
        remaining_numbers = values[1:]
        
        # 3. Convert numbers to strings and join with " & "
        numbers_str = " & ".join(map(str, remaining_numbers))
        
        # 4. Combine into final format
        line = f"{model_name} & {numbers_str}"
        formatted_lines.append(line)
        
    return "\n".join(formatted_lines)

# Generate and print the output
result = format_json_data(data)
print(result)

