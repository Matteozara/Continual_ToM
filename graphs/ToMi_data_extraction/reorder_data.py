import json

def process_accuracy_data(input_filename, output_filename):
    try:
        with open(input_filename, 'r') as f:
            data = json.load(f)
        
        results_dict = {}
        class_names = []

        for model_name, entries in data.items():
            if not isinstance(entries, list):
                continue
                
            # Sort entries by question_type to ensure consistent order
            sorted_entries = sorted(entries, key=lambda x: str(x['category']))
            
            # Capture the class names only once (from the first model processed)
            if not class_names:
                class_names = [str(item['category']) for item in sorted_entries]
            
            accuracies = [round(item['accuracy'] * 100, 2) for item in sorted_entries]
            results_dict[model_name] = accuracies

        # Save to file with custom formatting
        with open(output_filename, 'w') as f:
            # Write the class names as a comment or header at the top
            f.write(f"// Class Order: {', '.join(class_names)}\n")
            f.write("{\n")
            
            items = list(results_dict.items())
            for i, (model, scores) in enumerate(items):
                # json.dumps(scores) keeps the list on one line
                line = f'    "{model}": {json.dumps(scores)}'
                
                if i < len(items) - 1:
                    line += ","
                
                f.write(line + "\n")
            f.write("}\n")
        
        print(f"Successfully saved inline dictionary to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    process_accuracy_data('ToMi_summary_gpu1.json', 'output_results.json')