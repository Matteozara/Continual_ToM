'''
script to merge raw text and json
'''

import json
import re

# Input Data
text1 = """qwen3-30b_base & 78.33 & 99.85
qwen3-30b_CoT & 100.00 & 99.85
qwen3-30b_Complete & 99.17 & 100.00
qwen3-30b_standard & 80.83 & 98.95
qwen3-30b_obj_location & 72.50 & 88.54
qwen3-30b_obj_location_NEW & 81.67 & 99.80
qwen2.5-72b_base & 96.67 & 100.00
qwen2.5-72b_CoT & 99.17 & 100.00
qwen2.5-72b_Complete & 99.17 & 100.00
qwen2.5-72b_standard & 95.00 & 100.00
qwen2.5-72b_obj_location & 82.50 & 99.05
qwen2.5-72b_obj_location_NEW & 82.50 & 99.05"""

# Assuming the JSON snippet provided is part of a valid dictionary
text2_json = {
    "Sally-Anne_qwen3-30b_": [93.33, 93.33, 96.67, 83.33, 100.0],
    "Sally-Anne_qwen3-30b_CoT": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_qwen3-30b_Complete": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_qwen3-30b_standard": [53.33, 6.67, 13.33, 3.33, 66.67],
    "Sally-Anne_qwen3-30b_obj_location": [100.0, 0.0, 0.0, 0.0, 90.0],
    "Sally-Anne_qwen3-30b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 93.33],
    "Sally-Anne_qwen2.5-72b_": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_qwen2.5-72b_CoT": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_qwen2.5-72b_Complete": [100.0, 100.0, 100.0, 100.0, 100.0],
    "Sally-Anne_qwen2.5-72b_standard": [100.0, 26.67, 20.0, 3.33, 80.0],
    "Sally-Anne_qwen2.5-72b_obj_location": [100.0, 0.0, 0.0, 0.0, 100.0],
    "Sally-Anne_qwen2.5-72b_obj_location_NEW": [100.0, 0.0, 0.0, 0.0, 100.0]
}

def merge_models(t1_str, t2_dict):
    results = []
    
    # Process each line in the first text
    lines = t1_str.strip().split('\n')
    
    for line in lines:
        # Split by '&' and strip whitespace
        parts = [p.strip() for p in line.split('&')]
        model_name_t1 = parts[0]
        remaining_parts = " & ".join(parts[1:])

        # Format the model name: capitalize first letter and replace '_' with a space
        formatted_model = model_name_t1[0].upper() + model_name_t1[1:]
        formatted_model = formatted_model.replace('_', ' ')
        
        # Logic to find the matching key in text2_dict
        # We look for a key that contains the core model name
        # Mapping 'qwen3-4b_base' -> 'Sally-Anne_qwen3-4b_'
        # Mapping 'qwen3-4b_CoT'  -> 'Sally-Anne_qwen3-4b_CoT'
        
        match_found = False
        for key, values in t2_dict.items():
            # Clean keys for matching: remove 'Sally-Anne_' prefix
            clean_key = key.replace("Sally-Anne_", "")
            
            # Check for direct match or base model match
            if clean_key == model_name_t1 or \
               (model_name_t1.endswith("_base") and clean_key == model_name_t1.replace("_base", "_")):
                
                # Get the last number from the list
                last_val = f"{values[-1]:.2f}"
                
                # Construct the new line
                new_line = f"{formatted_model} & {remaining_parts} & {last_val} \\\\"
                results.append(new_line)
                match_found = True
                break
        
        if not match_found:
            results.append(f"{formatted_model} & {remaining_parts} & N/A \\\\")
            
    return "\n".join(results)

# Execute and print result
output = merge_models(text1, text2_json)
print(output)
