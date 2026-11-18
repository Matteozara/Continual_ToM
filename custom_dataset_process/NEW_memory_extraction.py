import json
import os


def extract_model_results(gold_path, prediction_path, output_file):
    """
    Extract story information, model answers,
    predicted memories, and ground truth memories.
    """

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    with open(prediction_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)


    # Index gold stories by id
    gold_lookup = {
        item["id"]: item
        for item in gold_data
    }


    extracted_results = []


    for pred_item in pred_data:

        story_id = pred_item.get("id")

        if story_id not in gold_lookup:
            continue


        gold_item = gold_lookup[story_id]


        result = {
            "id": story_id,

            "story": (
                gold_item.get("story")
                or gold_item.get("context")
                or []
            ),

            "question": (
                gold_item.get("question")
                or ""
            ),

            "gold": (
                gold_item.get("answer")
                or gold_item.get("gold")
                or ""
            ),

            "predicted": (
                pred_item.get("answer")
                or pred_item.get("prediction")
                or ""
            )
        }


        # Model predicted memories
        predicted_memories = pred_item.get(
            "memory_logs",
            []
        )

        # Dataset ground-truth memories
        gold_memories = gold_item.get(
            "memory_tracking",
            []
        )


        # Store memories as pairs
        max_steps = max(
            len(predicted_memories),
            len(gold_memories)
        )


        for step in range(max_steps):

            memory_id = step + 1


            if step < len(predicted_memories):

                result[
                    f"memory_predicted_{memory_id}"
                ] = (
                    predicted_memories[step]
                    .get("new_memory", "")
                )

            else:

                result[
                    f"memory_predicted_{memory_id}"
                ] = None



            if step < len(gold_memories):

                result[
                    f"gold_memory_{memory_id}"
                ] = gold_memories[step]

            else:

                result[
                    f"gold_memory_{memory_id}"
                ] = None



        extracted_results.append(result)



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            extracted_results,
            f,
            indent=4,
            ensure_ascii=False
        )



    print(
        f"Saved {len(extracted_results)} stories to {output_file}"
    )



# ============================================================
# Configuration
# ============================================================

pred_folder = "./custom_predictions" #"./custom_predictions_gpu1" #"./custom_predictions"

GOLD_MEMORY = "./../data/custom/memory_dataset_OQ.json"

OUT_DIR = "./new_memory_extracted"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


benchmarks = [
    "Custom"
]


models_gpu1 = [
    "qwen3-4b",
    "qwen2.5-7b",
    "llama3-8b",
    "nemo-12b"
]


models_gpu2 = [
    "gpt-20b",
    "gemma2-27b",
    "llama3-70b",
    "qwen3.6-35b",
    "qwen3-30b",
    "qwen2.5-72b"
]

models_official_1 = [
    "qwen3-4b",
    "qwen2.5-7b"
]

models_official_2 = [
    "gemma2-27b",
    "llama3-70b"
]

closed_source= ["gpt-5.6-luna"]


# Select models to process
all_models = closed_source


ebe_type = [
    "standard_ebe"
    #"task_specific_ebe"
]



# ============================================================
# Run extraction
# ============================================================

for model in all_models:

    for run in ebe_type:


        model_name = f"{model}_{run}"


        for bench in benchmarks:


            prediction_file = (
                f"{pred_folder}/"
                f"{bench}_{run}_EbE_{model}_raw.json"
            )


            output_file = (
                f"{OUT_DIR}/"
                f"{model_name}.json"
            )


            if not os.path.exists(prediction_file):

                print(
                    f"Skipping missing file: {prediction_file}"
                )

                continue



            extract_model_results(
                GOLD_MEMORY,
                prediction_file,
                output_file
            )


print("Finished exporting all models.")