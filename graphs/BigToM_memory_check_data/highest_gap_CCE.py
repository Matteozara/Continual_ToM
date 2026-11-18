import json
from collections import defaultdict

def analyze_gaps(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Group data by base model name
    # groups['qwen3-4b']['standard'] = [...]
    groups = defaultdict(dict)
    for key, values in data.items():
        if "_" in key:
            base_name, suffix = key.rsplit('_', 1)
            groups[base_name][suffix] = {item['question_type']: item['accuracy'] for item in values}

    for model_base, variants in groups.items():
        standard_scores = variants.get('standard')
        # Check for CoT first, then Complete as the comparison target
        comparison_suffix = 'CoT' if 'CoT' in variants else 'Complete'
        comparison_scores = variants.get(comparison_suffix)

        if not standard_scores or not comparison_scores:
            print(f"Skipping {model_base}: Missing standard or comparison data.\n")
            continue

        # Calculate gaps
        gaps = []
        for q_type, std_acc in standard_scores.items():
            if q_type in comparison_scores:
                comp_acc = comparison_scores[q_type]
                gap = comp_acc - std_acc
                gaps.append({
                    'type': q_type,
                    'std': std_acc,
                    'comp': comp_acc,
                    'gap': gap
                })

        # Sort by gap value
        sorted_gaps = sorted(gaps, key=lambda x: x['gap'], reverse=True)
        
        print(f"## Model: {model_base} ({comparison_suffix} vs Standard)")
        print(f"{'Question Type':<35} | {'Std':>6} | {comparison_suffix:>6} | {'Gap':>7}")
        print("-" * 65)

        print("Top 3 Biggest Gaps (CoT/Complete Improved Most):")
        for item in sorted_gaps[:3]:
            print(f"{item['type']:<35} | {item['std']:>6.1%} | {item['comp']:>6.1%} | +{item['gap']:>6.1%}")

        print("\nTop 3 Smallest/Negative Gaps (Standard was better or similar):")
        # Reverse the end of the list to show the most "negative" or smallest gap first
        for item in sorted_gaps[-3:][::-1]:
            indicator = "+" if item['gap'] > 0 else ""
            print(f"{item['type']:<35} | {item['std']:>6.1%} | {item['comp']:>6.1%} | {indicator}{item['gap']:>6.1%}")
        
        print("\n" + "="*65 + "\n")

if __name__ == "__main__":
    # Ensure your filename matches here
    analyze_gaps('BigToM_summary_gpu1.json')