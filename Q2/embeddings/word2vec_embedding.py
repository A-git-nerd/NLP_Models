from gensim.models import Word2Vec
import numpy as np
import pickle
import os

class Word2VecEmbedding:
    def __init__(self, vector_size=128, window=5, min_count=1, workers=4, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
    
    def train(self, texts):
        sentences = [text.split() for text in texts]
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        
        return self.model
    
    def get_embedding_matrix(self):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        vocab_size = len(self.model.wv)
        embedding_matrix = np.zeros((vocab_size + 2, self.vector_size))
        
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        
        for idx, word in enumerate(self.model.wv.index_to_key):
            word_to_idx[word] = idx + 2
            embedding_matrix[idx + 2] = self.model.wv[word]
        
        embedding_matrix[1] = np.random.randn(self.vector_size)
        
        return embedding_matrix, word_to_idx
    
    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = Word2Vec.load(filepath)
        return self.model
