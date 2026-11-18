import json
import statistics
import os
import spacy

# 1. Initialize spaCy (Using CPU is perfectly fine and fast for this)
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
    # Join sentences into a single block of text for processing
    full_text = " ".join(story_sentences)
    doc = nlp(full_text)
    
    entities = set()
    for token in doc:
        # Match Proper Nouns (Avery, Owen, Watermelon)
        if token.pos_ == "PROPN":
            entities.add(token.text.lower())
        
        # Match Nouns acting as subjects or objects (ignoring locations in prepositions)
        elif token.pos_ == "NOUN" and token.dep_ in ["nsubj", "nsubjpass", "dobj"]:
            entities.add(token.text.lower())
            
    return len(entities)


def calculate_story_metrics(json_file_path):
    # Load your JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # Note: If your file is truncated/malformed at the end due to copy-paste, 
        # ensure it ends with valid closing brackets ] before running.
        data = json.load(f)
    
    sentence_counts = []
    entity_counts = []
    
    print(f"Processing {len(data)} stories...")
    
    for entry in data:
        story_sentences = entry.get("story", [])
        
        # 1. Count sentences (each item in the "story" array is a sentence)
        num_sentences = len(story_sentences)
        sentence_counts.append(num_sentences)
        
        # 2. Count unique entities in this specific story
        num_entities = extract_entities_from_story(story_sentences)
        entity_counts.append(num_entities)

    # Calculate means
    if sentence_counts:
        mean_sentences = statistics.mean(sentence_counts)
        mean_entities = statistics.mean(entity_counts)
        
        print("\n--- Final Results ---")
        print(f"Mean sentences per story: {mean_sentences:.2f}")
        print(f"Mean unique entities per story: {mean_entities:.2f}")
    else:
        print("No story data found.")

# --- Execution ---
# Replace 'stories.json' with your actual file path
if __name__ == "__main__":
    # Dummy file generation for demonstration if needed, or point straight to your path
    json_path = "./../data/eval/HiToM/hitom.json" 
    
    calculate_story_metrics(json_path)
    