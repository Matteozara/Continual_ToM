import json

# Define the file paths
PATH_DIR = "./NER_Beliefs/beliefs"  #"./NER_Beliefs_gpu1/beliefs" 
input_filename = f"{PATH_DIR}/FINAL_SUMMARY.json"
output_filename = f"{PATH_DIR}/A_REORDERED.json"

try:
    # 1. Load the JSON data from your input file
    with open(input_filename, "r", encoding="utf-8") as f:
        input_data = json.load(f)
        
    # 2. Define the order of the new metric classes
    classes_order = [
        "% stories with beliefs", 
        "% stories with drops"
    ]

    # 3. Initialize the final output dictionary structure
    output_json = {
        "classes_order": classes_order
    }

    # 4. Extract the percentages for each model dynamically
    for model_name, metrics in input_data.items():
        # Map the precise keys used in your source JSON
        val_beliefs = metrics.get("percentage_stories_with_beliefs")
        val_drops = metrics.get("percentage_total_stories_with_drops")
        
        # Append the extracted values in the strict order requested
        output_json[model_name] = [val_beliefs, val_drops]

    # 5. Save the structured result to the new JSON file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4)

    print(f"Success! Data loaded from '{input_filename}' and saved to '{output_filename}'.")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found. Please ensure it is in the same directory.")
except json.JSONDecodeError:
    print(f"Error: '{input_filename}' does not contain valid JSON. Check for closing brackets.")