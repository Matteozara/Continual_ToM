import json

# Define the file paths
PATH_DIR = "./NER_Beliefs"  #"./NER_Beliefs_gpu1" 
input_filename = f"{PATH_DIR}/FINAL_SUMMARY_ADDITIONAL_MODELS.json" #f"{PATH_DIR}/FINAL_SUMMARY.json"
output_filename = f"{PATH_DIR}/REORDERED_ADDITIONAL_MODELS.json" #f"{PATH_DIR}/REORDERED.json"

try:
    # 1. Load the JSON data from your input file
    with open(input_filename, "r", encoding="utf-8") as f:
        input_data = json.load(f)
        
    # 2. Define the order of the metric classes
    classes_order = [
        "%wrong memories", 
        "% stories perfect", 
        "% stories_with errors", 
        "%empty memories"
    ]

    # 3. Initialize the final output dictionary structure
    output_json = {
        "classes_order": classes_order
    }

    # 4. Extract the percentages for each model dynamically
    for model_name, metrics in input_data.items():
        # Safeguard against potential missing keys using .get()
        val_wrong = metrics.get("percentage_wrong_memories")
        val_perfect = metrics.get("percentage_stories_perfect")
        val_errors = metrics.get("percentage_stories_with_errors")
        val_empty = metrics.get("percentage_empty_memories")
        
        # Append the extracted values in the strict order requested
        output_json[model_name] = [val_wrong, val_perfect, val_errors, val_empty]

    # 5. Save the structured result to the new JSON file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4)

    print(f"Success! Data loaded from '{input_filename}' and saved to '{output_filename}'.")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found. Please ensure it is in the same directory.")
except json.JSONDecodeError:
    print(f"Error: '{input_filename}' does not contain valid JSON. Check for trailing commas or syntax errors.")