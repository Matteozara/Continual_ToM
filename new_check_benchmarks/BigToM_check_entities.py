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


def extract_entities_from_narrative(narrative_sentences):
    """
    Combines the sentences into one text block and finds all unique 
    proper nouns and key subject/object nouns.
    """
    full_text = " ".join(narrative_sentences)
    doc = nlp(full_text)
    
    entities = set()
    for token in doc:
        # Match Proper Nouns (Noor, Cappuccino)
        if token.pos_ == "PROPN":
            entities.add(token.text.lower())
        
        # Match Nouns acting as subjects or objects (ignoring locations in prepositions)
        elif token.pos_ == "NOUN" and token.dep_ in ["nsubj", "nsubjpass", "dobj"]:
            entities.add(token.text.lower())
            
    return len(entities)


def calculate_narrative_metrics(json_file_path):
    # Load your JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentence_counts = []
    entity_counts = []
    
    print(f"Processing {len(data)} narratives...")
    
    for entry in data:
        # Adapted key: changed from 'story' to 'narrative'
        narrative_sentences = entry.get("narrative", [])
        
        # Guard clause: skip empty or incomplete fragments
        if not narrative_sentences:
            continue
            
        # 1. Count sentences
        num_sentences = len(narrative_sentences)
        sentence_counts.append(num_sentences)
        
        # 2. Count unique entities
        num_entities = extract_entities_from_narrative(narrative_sentences)
        entity_counts.append(num_entities)

    # Calculate means
    if sentence_counts:
        mean_sentences = statistics.mean(sentence_counts)
        mean_entities = statistics.mean(entity_counts)
        
        print("\n--- Final Results ---")
        print(f"Mean sentences per story: {mean_sentences:.2f}")
        print(f"Mean unique entities per story: {mean_entities:.2f}")
    else:
        print("No valid narrative data found to analyze.")


# --- Execution ---
if __name__ == "__main__":
    json_path = "./../data/eval/BigToM/bigtom.json"
    
    
    calculate_narrative_metrics(json_path)