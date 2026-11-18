import json
import os

def analyze_standard_models(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON. Check the file format.")
            return

    # Filter for models that represent the 'standard' version
    # This looks for keys ending in '_standard' (e.g., 'qwen3-4b_standard')
    standard_models = {k: v for k, v in data.items() if k.endswith('_standard')}

    if not standard_models:
        print("No models found ending with '_standard'.")
        print("Available keys in file:", list(data.keys()))
        return

    for model_name, entries in standard_models.items():
        # Sort by accuracy: Highest to lowest
        sorted_entries = sorted(entries, key=lambda x: x['accuracy'], reverse=True)
        
        # Identify Top 3 and Bottom 3
        top_3 = sorted_entries[:3]
        bottom_3 = sorted_entries[-3:][::-1] # Reverse to show worst at the top of the list

        print(f"==========================================")
        print(f"MODEL: {model_name.upper()}")
        print(f"==========================================")
        
        print(f"\n[ TOP 3 PERFORMING TYPES ]")
        for i, entry in enumerate(top_3, 1):
            print(f"  {i}. {entry['question_type']:<35} | Accuracy: {entry['accuracy']:.2%}")

        print(f"\n[ BOTTOM 3 PERFORMING TYPES ]")
        for i, entry in enumerate(bottom_3, 1):
            print(f"  {i}. {entry['question_type']:<35} | Accuracy: {entry['accuracy']:.2%}")
        print("\n")

if __name__ == "__main__":
    # Change 'results.json' to the actual name of your file
    analyze_standard_models('BigToM_summary_gpu1.json')
    