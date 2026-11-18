import json

def process_accuracy_data(input_filename, output_filename):
    try:
        with open(input_filename, 'r') as f:
            data = json.load(f)
        
        results_dict = {}

        for model_name, entries in data.items():
            if not isinstance(entries, list):
                continue
            
            # Initialize counters for our 3 target categories
            aggregated = {
                "reality": {"samples": 0, "corrects": 0},
                "1st_order": {"samples": 0, "corrects": 0},
                "2nd_order": {"samples": 0, "corrects": 0}
            }
            
            for item in entries:
                q_type = item.get('question_type')
                samples = item.get('samples', 0)
                corrects = item.get('corrects', 0)
                
                if q_type == "reality":
                    aggregated["reality"]["samples"] += samples
                    aggregated["reality"]["corrects"] += corrects
                elif q_type in ["1stA", "1stB"]:
                    aggregated["1st_order"]["samples"] += samples
                    aggregated["1st_order"]["corrects"] += corrects
                elif q_type in ["2ndA", "2ndB"]:
                    aggregated["2nd_order"]["samples"] += samples
                    aggregated["2nd_order"]["corrects"] += corrects

            # Calculate accuracies and format the output for this model
            model_results = {}
            for category, counts in aggregated.items():
                samples = counts["samples"]
                corrects = counts["corrects"]
                
                # Prevent DivisionbyZero if samples are 0
                accuracy = round((corrects / samples) * 100, 2) if samples > 0 else 0.0
                
                model_results[category] = {
                    "total_samples": samples,
                    "total_corrects": corrects,
                    "accuracy_percentage": accuracy
                }
                
            results_dict[model_name] = model_results

        # Save to file as standard, valid JSON
        with open(output_filename, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Successfully saved aggregated data to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Replace with your actual file paths
    process_accuracy_data('ToMChallenges_summary_gpu1.json', 'output_results_paper.json')