import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time
import numpy as np
import random
import os
from difflib import SequenceMatcher
import math
from collections import Counter



def levenshtein_distance(s1, s2):
    #Levenshtein distance counts the minimum number of single-character changed
    #A higher number means the strings are more different.
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def get_cosine_similarity(text1, text2):
    #Calculates semantic similarity based on word frequency.
    #1.0 means identical word usage; 0.0 means no shared words. (so doesn't spot synonyms)
    
    # 1. Tokenize words (simple split)
    words1 = Counter(text1.lower().split())
    words2 = Counter(text2.lower().split())
    
    # 2. Get unique words from both
    all_words = set(words1.keys()) | set(words2.keys())
    
    # 3. Calculate Dot Product and Magnitudes
    dot_product = sum(words1.get(w, 0) * words2.get(w, 0) for w in all_words)
    mag1 = math.sqrt(sum(words1.get(w, 0)**2 for w in all_words))
    mag2 = math.sqrt(sum(words2.get(w, 0)**2 for w in all_words))
    
    if not mag1 or not mag2:
        return 0.0
        
    return dot_product / (mag1 * mag2)


def calculate_similarity(text1, text2):
    #sequence matching
    seq_matching = SequenceMatcher(None, text1, text2).ratio() #higher means more similar
    #cosine similarity
    vocab_sim = get_cosine_similarity(text1, text2) #higher means more similar

    l_dist = levenshtein_distance(text1, text2) #higher means more different
    max_len = max(len(text1), len(text2))
    levenshtein_score = 1 - (l_dist / max_len)

    print(f"values: Sequence matching {seq_matching} - Cosin similarity {vocab_sim} - Levenshtein {levenshtein_score}")


    if seq_matching < 0.5 and vocab_sim < 0.45 and levenshtein_score < 0.45:
        return True #accettable 
    else:
        return False #too similar
    



print(calculate_similarity("Jackson entered the hall.", "Jackson is in the hall"))