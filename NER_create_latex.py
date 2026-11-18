import json

def convert_json_to_table(json_file_path, output_file_path=None):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the header columns from 'classes_order'
    classes = data.get("classes_order", [])
    
    # Find the index of the column we want to exclude
    target_column = "% stories perfect"
    exclude_index = classes.index(target_column) if target_column in classes else None
    
    # Filter the header classes to exclude the target column
    filtered_classes = [c for c in classes if c != target_column]
    header_row = "Model & " + " & ".join(filtered_classes)
    
    output_lines = [header_row]
    
    # Iterate through all other keys (the models)
    for model_name, values in data.items():
        if model_name == "classes_order":
            continue
        
        # Round numeric results to 2 decimal places, leave others as strings
        string_values = []
        for idx, val in enumerate(values):
            # Skip the value if it belongs to the excluded column index
            if idx == exclude_index:
                continue
                
            if isinstance(val, (int, float)):
                string_values.append(f"{val:.2f}")
            else:
                string_values.append(str(val))
        
        # Combine model name and values separated by ' & '
        row = f"{model_name} & " + " & ".join(string_values) + " \\\\"
        output_lines.append(row)
    
    # Print the result to the console
    for line in output_lines:
        print(line)
        
    # Optional: Save the output to a text file if a path is provided
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines) + "\n")



# Example Usage:
# Replace 'data.json' with your actual input file path
convert_json_to_table("./NER_Beliefs_obj/REORDERED_GPT20.json")
#convert_json_to_table("./NER_Beliefs/REORDERED_ADDITIONAL_MODELS.json")
#convert_json_to_table("./NER_Beliefs_gpu1/REORDERED.json")





'''
#GPU4

Model & %wrong memories & % stories_with errors & %empty memories
gpt-20b_HiToM_standard & 49.33 & 79.33 & 0.21 \\
gemma2-27b_HiToM_standard & 51.57 & 86.33 & 0.00 \\
gemma2-27b_HiToM_obj_location & 26.76 & 49.17 & 4.41 \\
llama3-70b_HiToM_standard & 31.22 & 62.33 & 0.00 \\
llama3-70b_HiToM_obj_location & 32.32 & 53.33 & 4.51 \\
qwen3-30b_HiToM_standard & 39.16 & 65.00 & 0.00 \\
qwen3-30b_HiToM_obj_location & 35.30 & 64.00 & 9.77 \\
qwen3-30b_HiToM_obj_location_NEW & 25.71 & 44.00 & 4.67 \\
qwen2.5-72b_HiToM_standard & 24.24 & 47.67 & 0.00 \\
qwen2.5-72b_HiToM_obj_location & 28.62 & 48.00 & 4.46 \\
qwen2.5-72b_HiToM_obj_location_NEW & 27.94 & 50.00 & 4.54 \\
gpt-20b_BigToM_standard & 20.02 & 39.67 & 0.04 \\
gemma2-27b_BigToM_standard & 9.29 & 26.37 & 0.00 \\
llama3-70b_BigToM_standard & 25.47 & 49.48 & 0.00 \\
qwen3-30b_BigToM_standard & 25.46 & 48.57 & 0.00 \\
qwen3-30b_BigToM_obj_location & 15.98 & 32.74 & 27.19 \\
qwen3-30b_BigToM_obj_location_NEW & 1.04 & 3.83 & 74.64 \\
qwen2.5-72b_BigToM_standard & 16.38 & 34.22 & 0.00 \\
qwen2.5-72b_BigToM_obj_location & 5.37 & 14.39 & 34.93 \\
qwen2.5-72b_BigToM_obj_location_NEW & 2.81 & 7.80 & 49.64 \\
gpt-20b_ToMChallenges_Sally-Anne_standard & 0.13 & 0.67 & 0.00 \\
gpt-20b_ToMChallenges_Sally-Anne_obj_location & 0.40 & 0.67 & 38.53 \\
gpt-20b_ToMChallenges_Sally-Anne_obj_location_NEW & 0.00 & 0.00 & 40.00 \\
gemma2-27b_ToMChallenges_Sally-Anne_standard & 8.00 & 20.00 & 0.00 \\
gemma2-27b_ToMChallenges_Sally-Anne_obj_location & 0.00 & 0.00 & 20.00 \\
gemma2-27b_ToMChallenges_Sally-Anne_obj_location_NEW & 4.00 & 6.67 & 35.33 \\
llama3-70b_ToMChallenges_Sally-Anne_standard & 0.00 & 0.00 & 0.00 \\
llama3-70b_ToMChallenges_Sally-Anne_obj_location & 0.00 & 0.00 & 20.00 \\
llama3-70b_ToMChallenges_Sally-Anne_obj_location_NEW & 0.00 & 0.00 & 20.00 \\
qwen3-30b_ToMChallenges_Sally-Anne_standard & 0.00 & 0.00 & 0.00 \\
qwen3-30b_ToMChallenges_Sally-Anne_obj_location & 17.33 & 33.33 & 20.00 \\
qwen3-30b_ToMChallenges_Sally-Anne_obj_location_NEW & 4.00 & 6.67 & 30.67 \\
qwen2.5-72b_ToMChallenges_Sally-Anne_standard & 3.33 & 10.00 & 0.00 \\
qwen2.5-72b_ToMChallenges_Sally-Anne_obj_location & 6.00 & 10.00 & 20.00 \\
qwen2.5-72b_ToMChallenges_Sally-Anne_obj_location_NEW & 4.00 & 6.67 & 20.00 \\
gpt-20b_ToMChallenges_Smarties_standard & 20.08 & 35.00 & 0.00 \\
gpt-20b_ToMChallenges_Smarties_obj_location & 0.00 & 0.00 & 1.43 \\
gpt-20b_ToMChallenges_Smarties_obj_location_NEW & 2.86 & 3.33 & 1.43 \\
gemma2-27b_ToMChallenges_Smarties_standard & 59.52 & 96.67 & 0.00 \\
gemma2-27b_ToMChallenges_Smarties_obj_location & 25.71 & 56.67 & 7.14 \\
gemma2-27b_ToMChallenges_Smarties_obj_location_NEW & 4.76 & 13.33 & 35.24 \\
llama3-70b_ToMChallenges_Smarties_standard & 46.67 & 86.67 & 0.00 \\
llama3-70b_ToMChallenges_Smarties_obj_location & 21.90 & 50.00 & 0.48 \\
llama3-70b_ToMChallenges_Smarties_obj_location_NEW & 0.95 & 3.33 & 2.86 \\
qwen3-30b_ToMChallenges_Smarties_standard & 28.57 & 56.67 & 0.00 \\
qwen3-30b_ToMChallenges_Smarties_obj_location & 35.71 & 53.33 & 1.43 \\
qwen3-30b_ToMChallenges_Smarties_obj_location_NEW & 3.81 & 6.67 & 14.29 \\
qwen2.5-72b_ToMChallenges_Smarties_standard & 20.95 & 40.00 & 0.00 \\
qwen2.5-72b_ToMChallenges_Smarties_obj_location & 6.67 & 16.67 & 1.90 \\
qwen2.5-72b_ToMChallenges_Smarties_obj_location_NEW & 2.86 & 6.67 & 2.38 \\
gpt-20b_ToMi_val_standard & 21.96 & 54.73 & 0.12 \\
gpt-20b_ToMi_val_obj_location & 0.21 & 0.42 & 35.14 \\
gemma2-27b_ToMi_val_standard & 9.17 & 25.33 & 0.00 \\
gemma2-27b_ToMi_val_obj_location & 0.11 & 0.30 & 35.67 \\
llama3-70b_ToMi_val_standard & 4.44 & 11.21 & 0.00 \\
llama3-70b_ToMi_val_obj_location & 2.68 & 5.01 & 30.29 \\
qwen3-30b_ToMi_val_standard & 4.81 & 12.31 & 0.00 \\
qwen3-30b_ToMi_val_obj_location & 8.72 & 24.72 & 34.31 \\
qwen3-30b_ToMi_val_obj_location_NEW & 0.07 & 0.20 & 36.33 \\
qwen2.5-72b_ToMi_val_standard & 2.06 & 6.57 & 0.00 \\
qwen2.5-72b_ToMi_val_obj_location & 0.09 & 0.20 & 35.60 \\
qwen2.5-72b_ToMi_val_obj_location_NEW & 0.04 & 0.10 & 35.76 \\
gpt-20b_ToMi_test_standard & 21.21 & 53.55 & 0.09 \\
gpt-20b_ToMi_test_obj_location & 0.57 & 0.80 & 35.10 \\
gemma2-27b_ToMi_test_standard & 11.08 & 29.63 & 0.00 \\
gemma2-27b_ToMi_test_obj_location & 0.14 & 0.20 & 35.92 \\
llama3-70b_ToMi_test_standard & 3.94 & 10.21 & 0.00 \\
llama3-70b_ToMi_test_obj_location & 4.23 & 7.11 & 29.69 \\
qwen3-30b_ToMi_test_standard & 6.13 & 15.92 & 0.00 \\
qwen3-30b_ToMi_test_obj_location & 10.01 & 27.03 & 35.09 \\
qwen3-30b_ToMi_test_obj_location_NEW & 0.01 & 0.10 & 36.23 \\
qwen2.5-72b_ToMi_test_standard & 2.97 & 8.93 & 0.00 \\
qwen2.5-72b_ToMi_test_obj_location & 0.00 & 0.00 & 35.79 \\
qwen2.5-72b_ToMi_test_obj_location_NEW & 0.00 & 0.00 & 35.92 \\
gpt-20b_ToMi_reality_standard & 20.86 & 52.35 & 0.14 \\
gpt-20b_ToMi_reality_obj_location & 0.59 & 0.95 & 34.95 \\
gemma2-27b_ToMi_reality_standard & 9.97 & 27.13 & 0.00 \\
gemma2-27b_ToMi_reality_obj_location & 0.11 & 0.20 & 35.80 \\
gemma2-27b_ToMi_reality_obj_location_NEW & 0.06 & 0.15 & 35.83 \\
llama3-70b_ToMi_reality_standard & 4.20 & 10.76 & 0.00 \\
llama3-70b_ToMi_reality_obj_location & 3.46 & 6.06 & 29.99 \\
llama3-70b_ToMi_reality_obj_location_NEW & 0.56 & 0.85 & 33.65 \\
qwen3-30b_ToMi_reality_standard & 5.45 & 14.06 & 0.00 \\
qwen3-30b_ToMi_reality_obj_location & 9.36 & 25.83 & 34.70 \\
qwen3-30b_ToMi_reality_obj_location_NEW & 0.04 & 0.15 & 36.28 \\
qwen2.5-72b_ToMi_reality_standard & 2.56 & 7.61 & 0.00 \\
qwen2.5-72b_ToMi_reality_obj_location & 0.04 & 0.10 & 35.70 \\
qwen2.5-72b_ToMi_reality_obj_location_NEW & 0.02 & 0.05 & 35.84 \\
gpt-20b_HiToM_TABLE0_standard & 44.51 & 72.50 & 0.29 \\
gpt-20b_HiToM_TABLE0_obj_location & 30.16 & 52.50 & 4.35 \\
gpt-20b_HiToM_TABLE0_obj_location_NEW & 26.76 & 47.50 & 4.54 \\
gemma2-27b_HiToM_TABLE0_standard & 51.37 & 86.67 & 0.00 \\
gemma2-27b_HiToM_TABLE0_obj_location & 26.76 & 49.17 & 4.41 \\
gemma2-27b_HiToM_TABLE0_obj_location_NEW & 27.94 & 50.83 & 4.54 \\
llama3-70b_HiToM_TABLE0_standard & 32.76 & 62.50 & 0.00 \\
llama3-70b_HiToM_TABLE0_obj_location & 32.32 & 53.33 & 4.51 \\
llama3-70b_HiToM_TABLE0_obj_location_NEW & 27.94 & 50.00 & 4.54 \\
qwen3-30b_HiToM_TABLE0_standard & 40.10 & 71.67 & 0.00 \\
qwen3-30b_HiToM_TABLE0_obj_location & 38.03 & 65.00 & 9.97 \\
qwen3-30b_HiToM_TABLE0_obj_location_NEW & 26.79 & 46.67 & 4.54 \\
qwen2.5-72b_HiToM_TABLE0_standard & 27.14 & 50.83 & 0.00 \\
qwen2.5-72b_HiToM_TABLE0_obj_location & 29.49 & 50.83 & 4.32 \\
qwen2.5-72b_HiToM_TABLE0_obj_location_NEW & 27.94 & 50.00 & 4.54 \\





#GPU1

Model & %wrong memories & % stories_with errors & %empty memories
qwen3-4b_HiToM_standard & 61.64 & 93.67 & 0.00 \\
qwen3-4b_HiToM_obj_location & 33.78 & 58.33 & 5.56 \\
qwen3-4b_HiToM_obj_location_NEW & 26.16 & 46.00 & 4.57 \\
qwen2.5-7b_HiToM_standard & 56.84 & 91.00 & 0.00 \\
qwen2.5-7b_HiToM_obj_location & 43.07 & 73.00 & 5.71 \\
qwen2.5-7b_HiToM_obj_location_NEW & 27.75 & 54.00 & 5.17 \\
llama3-8b_HiToM_standard & 57.17 & 93.00 & 0.27 \\
llama3-8b_HiToM_obj_location & 52.03 & 91.33 & 16.19 \\
llama3-8b_HiToM_obj_location_NEW & 68.46 & 98.33 & 43.12 \\
nemo-12b_HiToM_standard & 83.16 & 100.00 & 0.23 \\
nemo-12b_HiToM_obj_location & 44.35 & 68.00 & 4.67 \\
nemo-12b_HiToM_obj_location_NEW & 43.05 & 66.33 & 4.67 \\
qwen3-4b_BigToM_standard & 27.01 & 47.93 & 0.00 \\
qwen3-4b_BigToM_obj_location & 8.42 & 17.85 & 50.98 \\
qwen3-4b_BigToM_obj_location_NEW & 0.36 & 1.17 & 87.25 \\
qwen2.5-7b_BigToM_standard & 49.28 & 85.00 & 0.00 \\
qwen2.5-7b_BigToM_obj_location & 10.81 & 27.87 & 45.64 \\
qwen2.5-7b_BigToM_obj_location_NEW & 1.71 & 5.35 & 80.21 \\
llama3-8b_BigToM_standard & 23.76 & 52.74 & 0.08 \\
llama3-8b_BigToM_obj_location & 11.64 & 27.59 & 55.56 \\
llama3-8b_BigToM_obj_location_NEW & 6.37 & 18.17 & 85.55 \\
nemo-12b_BigToM_standard & 26.24 & 56.37 & 0.00 \\
nemo-12b_BigToM_obj_location & 3.55 & 9.00 & 65.20 \\
nemo-12b_BigToM_obj_location_NEW & 0.07 & 0.39 & 88.19 \\
qwen3-4b_ToMChallenges_Sally-Anne_standard & 10.00 & 23.33 & 0.00 \\
qwen3-4b_ToMChallenges_Sally-Anne_obj_location & 16.00 & 33.33 & 17.33 \\
qwen3-4b_ToMChallenges_Sally-Anne_obj_location_NEW & 0.00 & 0.00 & 40.67 \\
qwen2.5-7b_ToMChallenges_Sally-Anne_standard & 23.33 & 43.33 & 0.00 \\
qwen2.5-7b_ToMChallenges_Sally-Anne_obj_location & 50.00 & 86.67 & 20.00 \\
qwen2.5-7b_ToMChallenges_Sally-Anne_obj_location_NEW & 32.00 & 53.33 & 26.67 \\
llama3-8b_ToMChallenges_Sally-Anne_standard & 34.67 & 63.33 & 0.00 \\
llama3-8b_ToMChallenges_Sally-Anne_obj_location & 16.67 & 43.33 & 20.00 \\
llama3-8b_ToMChallenges_Sally-Anne_obj_location_NEW & 12.67 & 33.33 & 35.33 \\
nemo-12b_ToMChallenges_Sally-Anne_standard & 23.33 & 70.00 & 0.00 \\
nemo-12b_ToMChallenges_Sally-Anne_obj_location & 0.00 & 0.00 & 40.00 \\
nemo-12b_ToMChallenges_Sally-Anne_obj_location_NEW & 0.00 & 0.00 & 40.00 \\
qwen3-4b_ToMChallenges_Smarties_standard & 25.71 & 56.67 & 0.00 \\
qwen3-4b_ToMChallenges_Smarties_obj_location & 27.14 & 63.33 & 1.43 \\
qwen3-4b_ToMChallenges_Smarties_obj_location_NEW & 1.90 & 10.00 & 58.10 \\
qwen2.5-7b_ToMChallenges_Smarties_standard & 58.10 & 90.00 & 0.00 \\
qwen2.5-7b_ToMChallenges_Smarties_obj_location & 40.00 & 80.00 & 0.95 \\
qwen2.5-7b_ToMChallenges_Smarties_obj_location_NEW & 7.14 & 20.00 & 22.38 \\
llama3-8b_ToMChallenges_Smarties_standard & 47.14 & 96.67 & 0.00 \\
llama3-8b_ToMChallenges_Smarties_obj_location & 40.00 & 76.67 & 19.52 \\
llama3-8b_ToMChallenges_Smarties_obj_location_NEW & 13.33 & 23.33 & 69.52 \\
nemo-12b_ToMChallenges_Smarties_standard & 43.81 & 83.33 & 0.00 \\
nemo-12b_ToMChallenges_Smarties_obj_location & 42.38 & 83.33 & 6.19 \\
nemo-12b_ToMChallenges_Smarties_obj_location_NEW & 25.71 & 56.67 & 18.10 \\
qwen3-4b_ToMi_val_standard & 9.67 & 21.68 & 0.00 \\
qwen3-4b_ToMi_val_obj_location & 2.42 & 4.80 & 34.30 \\
qwen3-4b_ToMi_val_obj_location_NEW & 0.56 & 1.00 & 36.53 \\
qwen2.5-7b_ToMi_val_standard & 13.89 & 33.03 & 0.00 \\
qwen2.5-7b_ToMi_val_obj_location & 38.37 & 53.03 & 23.01 \\
qwen2.5-7b_ToMi_val_obj_location_NEW & 0.96 & 2.50 & 36.46 \\
llama3-8b_ToMi_val_standard & 17.78 & 36.52 & 0.86 \\
llama3-8b_ToMi_val_obj_location & 10.41 & 39.46 & 45.29 \\
llama3-8b_ToMi_val_obj_location_NEW & 26.19 & 75.90 & 61.99 \\
nemo-12b_ToMi_val_standard & 46.79 & 84.58 & 0.27 \\
nemo-12b_ToMi_val_obj_location & 0.83 & 1.90 & 37.12 \\
nemo-12b_ToMi_val_obj_location_NEW & 0.14 & 0.30 & 38.82 \\
qwen3-4b_ToMi_test_standard & 11.41 & 27.23 & 0.00 \\
qwen3-4b_ToMi_test_obj_location & 1.88 & 4.00 & 34.56 \\
qwen3-4b_ToMi_test_obj_location_NEW & 0.79 & 1.40 & 36.05 \\
qwen2.5-7b_ToMi_test_standard & 14.45 & 32.85 & 0.00 \\
qwen2.5-7b_ToMi_test_obj_location & 37.45 & 52.67 & 22.65 \\
qwen2.5-7b_ToMi_test_obj_location_NEW & 1.39 & 3.10 & 36.18 \\
llama3-8b_ToMi_test_standard & 18.95 & 38.40 & 0.69 \\
llama3-8b_ToMi_test_obj_location & 9.52 & 37.28 & 44.46 \\
llama3-8b_ToMi_test_obj_location_NEW & 25.91 & 75.38 & 61.61 \\
nemo-12b_ToMi_test_standard & 45.84 & 83.80 & 0.27 \\
nemo-12b_ToMi_test_obj_location & 1.15 & 2.30 & 36.94 \\
nemo-12b_ToMi_test_obj_location_NEW & 0.06 & 0.30 & 38.55 \\
qwen3-4b_ToMi_reality_standard & 10.51 & 24.47 & 0.00 \\
qwen3-4b_ToMi_reality_obj_location & 2.15 & 4.35 & 34.44 \\
qwen3-4b_ToMi_reality_obj_location_NEW & 0.67 & 1.20 & 36.29 \\
qwen2.5-7b_ToMi_reality_standard & 14.36 & 33.18 & 0.00 \\
qwen2.5-7b_ToMi_reality_obj_location & 37.99 & 52.95 & 22.79 \\
qwen2.5-7b_ToMi_reality_obj_location_NEW & 1.17 & 2.75 & 36.32 \\
llama3-8b_ToMi_reality_standard & 18.22 & 37.39 & 0.80 \\
llama3-8b_ToMi_reality_obj_location & 10.11 & 39.19 & 45.00 \\
llama3-8b_ToMi_reality_obj_location_NEW & 26.30 & 76.13 & 62.03 \\
nemo-12b_ToMi_reality_standard & 46.27 & 84.33 & 0.31 \\
nemo-12b_ToMi_reality_obj_location & 1.00 & 2.10 & 37.03 \\
nemo-12b_ToMi_reality_obj_location_NEW & 0.09 & 0.25 & 38.55 \\
qwen3-4b_HiToM_TABLE0_standard & 63.78 & 94.17 & 0.00 \\
qwen3-4b_HiToM_TABLE0_obj_location & 35.52 & 61.67 & 5.40 \\
qwen3-4b_HiToM_TABLE0_obj_location_NEW & 27.17 & 47.50 & 4.41 \\
qwen2.5-7b_HiToM_TABLE0_standard & 58.83 & 92.50 & 0.00 \\
qwen2.5-7b_HiToM_TABLE0_obj_location & 43.37 & 70.83 & 5.65 \\
qwen2.5-7b_HiToM_TABLE0_obj_location_NEW & 29.08 & 53.33 & 5.08 \\
llama3-8b_HiToM_TABLE0_standard & 56.51 & 93.33 & 0.32 \\
llama3-8b_HiToM_TABLE0_obj_location & 52.32 & 91.67 & 15.75 \\
llama3-8b_HiToM_TABLE0_obj_location_NEW & 69.02 & 97.50 & 43.08 \\
nemo-12b_HiToM_TABLE0_standard & 84.19 & 100.00 & 0.16 \\
nemo-12b_HiToM_TABLE0_obj_location & 47.43 & 73.33 & 4.54 \\
nemo-12b_HiToM_TABLE0_obj_location_NEW & 46.00 & 71.67 & 4.54 \\
'''


