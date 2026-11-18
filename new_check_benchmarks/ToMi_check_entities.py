import json
import statistics
import os
import spacy

# 1. Initialize spaCy
MODEL_NAME = "en_core_web_lg"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"Downloading {MODEL_NAME}...")
    os.system(f"python -m spacy download {MODEL_NAME}")
    nlp = spacy.load(MODEL_NAME)


def extract_entities_from_story(story_sentences):
    """
    Combines the sentences into one text block and finds all unique 
    proper nouns and key subject/object nouns.
    """
    full_text = " ".join(story_sentences)
    doc = nlp(full_text)
    
    entities = set()
    for token in doc:
        # Match Proper Nouns (Jackson, Chloe)
        if token.pos_ == "PROPN":
            entities.add(token.text.lower())
        
        # Match Nouns acting as subjects or objects (boots)
        # Ignores locations like bathtub/dining_room/pantry in prepositional phrases
        elif token.pos_ == "NOUN" and token.dep_ in ["nsubj", "nsubjpass", "dobj"]:
            entities.add(token.text.lower())
            
    return len(entities)


def calculate_story_metrics(json_file_path):
    # Load your JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentence_counts = []
    entity_counts = []
    
    print(f"Processing {len(data)} stories...")
    
    for entry in data:
        # Target 'story' key for this dataset structure
        story_sentences = entry.get("story", [])
        
        # Guard clause: skip empty list or truncated fragments (e.g., the last item)
        if not story_sentences:
            continue
            
        # 1. Count sentences
        num_sentences = len(story_sentences)
        sentence_counts.append(num_sentences)
        
        # 2. Count unique entities using the exact logic from your original script
        num_entities = extract_entities_from_story(story_sentences)
        entity_counts.append(num_entities)

    # Calculate and display metrics
    if sentence_counts:
        mean_sentences = statistics.mean(sentence_counts)
        mean_entities = statistics.mean(entity_counts)
        
        print("\n--- Final Results ---")
        print(f"Mean sentences per story: {mean_sentences:.2f}")
        print(f"Mean unique entities per story: {mean_entities:.2f}")
    else:
        print("No valid story data found to analyze.")


# --- Execution ---
if __name__ == "__main__":
    json_path = "./../data/eval/ToMi/test.json" #"./../data/eval/ToMi/val.json" #"./../data/eval/ToMi/test.json"
    
    calculate_story_metrics(json_path)