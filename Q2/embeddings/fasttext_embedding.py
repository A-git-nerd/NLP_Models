from gensim.models import FastText
import numpy as np

class FastTextEmbedding:
    def __init__(self, vector_size=128, window=5, min_count=1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def train(self, texts):
        # Tokenize texts into sentences of words
        sentences = [text.split() for text in texts]
        
        # Train FastText model using gensim
        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=1  # 1 for skip-gram, 0 for CBOW
        )
        
        return self.model
    
    def get_embedding_matrix(self):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get vocabulary from the model
        words = list(self.model.wv.index_to_key)
        vocab_size = len(words)
        embedding_matrix = np.zeros((vocab_size + 2, self.vector_size))
        
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        
        for idx, word in enumerate(words):
            word_to_idx[word] = idx + 2
            embedding_matrix[idx + 2] = self.model.wv[word]
        
        # Random embedding for unknown tokens
        embedding_matrix[1] = np.random.randn(self.vector_size)
        
        return embedding_matrix, word_to_idx
    
    def save(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = FastText.load(filepath)
        return self.model
