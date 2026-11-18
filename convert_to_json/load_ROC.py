from datasets import load_dataset
import json

# Load dataset from HuggingFace
dataset = load_dataset("mintujupally/ROCStories")

# Convert to list of dicts
train_data = dataset["train"].to_list()
test_data = dataset["test"].to_list()

# Save to JSON files
with open("data/train/ROC/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open("data/train/ROC/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print("Saved train and test datasets to JSON.")