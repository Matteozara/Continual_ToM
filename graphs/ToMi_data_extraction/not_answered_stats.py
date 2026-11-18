import json

def process_grouped_stats(input_file, output_file):
    try:
        # 1. Load the original JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        grouped_results = {}

        # 2. Iterate through each model
        for model_name, categories in data.items():
            # Create a dictionary for this model if it doesn't exist
            grouped_results[model_name] = {}
            
            for item in categories:
                category = item.get("category", "unknown_type")
                samples = item.get("samples", 0)
                not_answered = item.get("not_answered", 0)
                
                # 3. Calculate percentage
                percentage = (not_answered / samples * 100) if samples > 0 else 0
                
                # 4. Nest the stats under the question type for this model
                grouped_results[model_name][category] = {
                    "samples": samples,
                    "not_answered": not_answered,
                    "percentage_not_answered": round(percentage, 4)
                }

        # 5. Print the grouped JSON to the screen
        print(json.dumps(grouped_results, indent=4))

        # 6. Save the grouped results to a new file
        with open(output_file, 'w') as f:
            json.dump(grouped_results, f, indent=4)
        
        print(f"\nSuccessfully saved grouped results to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure 'input.json' exists in your directory
    process_grouped_stats('ToMi_summary_gpu1.json', 'extracted_stats.json')