# import numpy as np
# import torch
# import matplotlib.pyplot as plt
import os
import string
from collections import Counter


STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

def prepare_data_from_file(file_path, context_window_size=2, word_limit=200):
    # book folders outside of lab1 folder
    with open(file_path,  'r', encoding='utf-8') as f:
        CORPUS_TEXT = f.read()

    clean_text = CORPUS_TEXT.lower()
    
    # from these prints it's noticeable ' the ' and ' "the ' are being kept in vocab
    punctuation_list = string.punctuation + '“' + '”' + '’'
    translator = str.maketrans('', '', punctuation_list)
    clean_text = clean_text.translate(translator)
    
    # Tokenise and filter for alphabetic-only words
    tokens = [word for word in clean_text.split() if word.isalpha()]

    # optional to remove common stop words, (can use libraries but.... im lazy to sort out nltk as of now)
    # could change later
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # Build vocabulary from the most frequent words up to the word_limit
    word_counts = Counter(tokens)
    vocabulary = [word for word, count in word_counts.most_common(word_limit)]
    
    # forward and backward mapping 
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for i, word in enumerate(vocabulary)}
    
    # Filter the original tokens to only include words that are in our new limited vocabulary
    tokens_in_vocab = [word for word in tokens if word in word_to_idx]
    
    training_pairs = []
    for i, focal_word in enumerate(tokens_in_vocab):
        focal_idx = word_to_idx[focal_word]
        # Iterate through the context window
        for j in range(max(0, i - context_window_size),  min(len(tokens_in_vocab), i + context_window_size + 1)):
            if i == j:
                continue
            
            context_word = tokens_in_vocab[j]
            context_idx = word_to_idx[context_word]
            training_pairs.append((focal_idx, context_idx))
            
    return vocabulary, word_to_idx, idx_to_word, training_pairs

# --- Example of how to use the function ---
FILE_PATH = "harry_potter/HP4.txt"
vocabulary, word_to_idx, idx_to_word, training_pairs = prepare_data_from_file(FILE_PATH)
n = 10
print(f"Vocabulary size: {len(vocabulary)}")
print(f"First {n} words in vocab: {vocabulary[:n]}")
print(f"Number of training pairs: {len(training_pairs)}")
