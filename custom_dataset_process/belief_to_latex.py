import json
import re

# 1. Define your JSON data (can also be loaded from a file using json.load())
json_data = {
    "qwen3-4b_standard_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1905/2445 (77.91%)\n1ST ORDER BEL:  420/1395 (30.11%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 235/885 (26.55%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "qwen3-4b_task_specific_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1845/2445 (75.46%)\n1ST ORDER BEL:  80/1395 (5.73%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 195/885 (22.03%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "qwen2.5-7b_standard_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1935/2445 (79.14%)\n1ST ORDER BEL:  470/1395 (33.69%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 230/885 (25.99%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "qwen2.5-7b_task_specific_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1905/2445 (77.91%)\n1ST ORDER BEL:  585/1395 (41.94%)\nHIGHER ORDER:   55/1275 (4.31%)\n\nTOTAL STEP SUCCESS RATE: 245/885 (27.68%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "llama3-8b_standard_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1755/2445 (71.78%)\n1ST ORDER BEL:  100/1395 (7.17%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 220/885 (24.86%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "llama3-8b_task_specific_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1760/2445 (71.98%)\n1ST ORDER BEL:  170/1395 (12.19%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 225/885 (25.42%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "nemo-12b_standard_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1540/2445 (62.99%)\n1ST ORDER BEL:  60/1395 (4.30%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 210/885 (23.73%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n",
    "nemo-12b_task_specific_ebe": "\n==================================================\nOVERALL LOGIC STATISTICS (BY CATEGORY)\n==================================================\nLOCATIONS:      1665/2445 (68.10%)\n1ST ORDER BEL:  190/1395 (13.62%)\nHIGHER ORDER:   0/1275 (0.00%)\n\nTOTAL STEP SUCCESS RATE: 215/885 (24.29%)\nTOTAL PERFECT STORIES:   0/150 (0.00%)\n==================================================\n"
}

def extract_metrics(data_dict):
    output_lines = []
    
    # 2. Compile regular expressions to extract percentages inside parentheses
    loc_pattern = re.compile(r'LOCATIONS:\s+\d+/\d+\s+\(([\d.]+)%\)')
    bel1_pattern = re.compile(r'1ST ORDER BEL:\s+\d+/\d+\s+\(([\d.]+)%\)')
    higher_pattern = re.compile(r'HIGHER ORDER:\s+\d+/\d+\s+\(([\d.]+)%\)')
    
    for model_name, report in data_dict.items():
        # Clean up model name formatting if desired (e.g., replacing underscores)
        # For exact match with your request, we can keep the key or clean it up:
        clean_name = model_name.replace('_ebe', '').replace('_', ' ')
        
        # 3. Search for matches in the text block
        loc_match = loc_pattern.search(report)
        bel1_match = bel1_pattern.search(report)
        higher_match = higher_pattern.search(report)
        
        # 4. Extract strings or fallback to '0.00' if not found
        loc_pct = loc_match.group(1) if loc_match else "0.00"
        bel1_pct = bel1_match.group(1) if bel1_match else "0.00"
        higher_pct = higher_match.group(1) if higher_match else "0.00"
        
        # 5. Format the string using the '&' separator
        formatted_line = f"{clean_name} & {loc_pct}\\% & {bel1_pct}\\% & {higher_pct}\\% \\\\"
        output_lines.append(formatted_line)
        
    return output_lines

# Execute and print results
results = extract_metrics(json_data)
for line in results:
    print(line)