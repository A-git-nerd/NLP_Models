import numpy as np
import os
import gensim.downloader as api

def download_glove_embeddings(glove_dir='glove_data', embedding_dim=100):
    """Download GloVe embeddings using gensim"""
    print(f"Loading GloVe embeddings with dimension {embedding_dim}...")
    
    # Map embedding dimensions to gensim model names
    model_map = {
        50: 'glove-wiki-gigaword-50',
        100: 'glove-wiki-gigaword-100',
        200: 'glove-wiki-gigaword-200',
        300: 'glove-wiki-gigaword-300'
    }
    
    if embedding_dim not in model_map:
        print(f"Warning: {embedding_dim}d not available, using 100d instead")
        embedding_dim = 100
    
    model_name = model_map[embedding_dim]
    
    try:
        # This will download and cache the model
        model = api.load(model_name)
        print(f"GloVe {embedding_dim}d embeddings loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading GloVe: {e}")
        return None

def load_glove_embeddings(glove_model, vocab, embedding_dim=100):
    """Load GloVe embeddings from gensim model into vocabulary"""
    if glove_model is None:
        print("No GloVe model provided, using random embeddings")
        vocab_size = len(vocab)
        return np.random.randn(vocab_size, embedding_dim) * 0.01
    
    print(f"Creating embedding matrix for vocabulary...")
    
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    found_words = 0
    for word, idx in vocab.items():
        if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
            if word == '<PAD>':
                embedding_matrix[idx] = np.zeros(embedding_dim)
            else:
                embedding_matrix[idx] = np.random.randn(embedding_dim) * 0.01
        elif word in glove_model:
            embedding_matrix[idx] = glove_model[word]
            found_words += 1
        else:
            embedding_matrix[idx] = np.random.randn(embedding_dim) * 0.01
    
    coverage = (found_words / (vocab_size - 4)) * 100 if vocab_size > 4 else 0
    print(f"Found embeddings for {found_words}/{vocab_size-4} words ({coverage:.2f}% coverage)")
    
    return embedding_matrix
