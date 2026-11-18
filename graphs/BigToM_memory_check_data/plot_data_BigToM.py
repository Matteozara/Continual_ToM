import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_JSON = "BigToM_summary_gpu1.json"
CORE_MODELS = ["gpt-20b","gemma2-27b","granite4-32b","llama3-70b", "qwen3.6-35b"] #["qwen3-4b", "qwen2.5-7b", "llama3-8b", "nemo-12b"]
#mine (good)
TARGET_QUESTIONS = [
    "1_backward_belief_true_belief", 
    "1_percept_to_belief_true_belief", 
    "1_forward_belief_true_belief"
]
#best
'''TARGET_QUESTIONS = [
    "1_percept_to_belief_true_belief", 
    "1_forward_belief_false_control", 
    "1_backward_belief_false_control"
]'''

#worst
'''TARGET_QUESTIONS = [
    "1_forward_belief_false_belief", 
    "1_forward_action_false_belief", 
    "0_forward_belief_false_belief", #0_forward_action_false_belief"
]'''




def generate_transposed_plots():
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON} not found.")
        return

    output_dir = "plots_by_question_type"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for core in CORE_MODELS:
        # Dictionary to store accuracy for each variation (legend) across question types (x-axis)
        model_data = {}
        
        for source_key, questions in data.items():
            if source_key.startswith(core):
                # Extract variation name (e.g., 'obj_location')
                variation = source_key.replace(core, "").strip("_")
                if variation == "": variation = "vp"
                if variation == "standard": variation = "EbE"
                
                # Create a sub-dictionary for this variation
                model_data[variation] = {}
                for q in questions:
                    if q["question_type"] in TARGET_QUESTIONS:
                        model_data[variation][q["question_type"]] = q["accuracy"]

        if model_data:
            # Create DataFrame and Transpose it
            # We want: Index = Question Types (X-axis), Columns = Variations (Legend)
            df = pd.DataFrame(model_data)
            
            # Formatting the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')

            plt.title(f'Model Analysis: {core}', fontsize=20, fontweight='bold', pad=15)
            plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
            plt.xlabel('Question Type', fontsize=14, fontweight='bold')
            
            # Clean up X-axis labels (remove underscores for readability)
            clean_labels = [label.replace('_', ' ') for label in df.index]
            ax.set_xticklabels(clean_labels, rotation=15, ha='right', fontsize=11)
            
            plt.ylim(0, 1.1)
            plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
            
            # Legend settings
            plt.legend(fontsize=12, title_fontsize=13, 
                       loc='upper left', bbox_to_anchor=(1, 1))

            # Save
            file_name = f"{output_dir}/{core}_by_qtype.png"
            plt.tight_layout()
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated plot for {core} at {file_name}")

if __name__ == "__main__":
    generate_transposed_plots()