import numpy as np
from collections import defaultdict
import pickle

class GloveEmbedding:
    def __init__(self, vector_size=128, learning_rate=0.05, epochs=10, x_max=100, alpha=0.75):
        self.vector_size = vector_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.x_max = x_max
        self.alpha = alpha
        self.word_vectors = None
        self.word_to_idx = None
    
    def build_cooccurrence_matrix(self, texts, window_size=5):
        word_counts = defaultdict(int)
        cooccurrence = defaultdict(float)
        
        for text in texts:
            words = text.split()
            for word in words:
                word_counts[word] += 1
        
        vocab = {word: idx for idx, (word, _) in enumerate(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))}
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab)
        
        for text in texts:
            words = text.split()
            for i, word in enumerate(words):
                if word not in vocab:
                    continue
                for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                    if i != j and words[j] in vocab:
                        distance = abs(i - j)
                        cooccurrence[(vocab[word], vocab[words[j]])] += 1.0 / distance
        
        return cooccurrence, vocab
    
    def train(self, texts):
        cooccurrence, self.word_to_idx = self.build_cooccurrence_matrix(texts)
        
        vocab_size = len(self.word_to_idx)
        
        W = np.random.randn(vocab_size, self.vector_size) * 0.01
        W_tilde = np.random.randn(vocab_size, self.vector_size) * 0.01
        b = np.random.randn(vocab_size) * 0.01
        b_tilde = np.random.randn(vocab_size) * 0.01
        
        for epoch in range(self.epochs):
            total_loss = 0
            for (i, j), x_ij in cooccurrence.items():
                weight = min(1.0, (x_ij / self.x_max) ** self.alpha) if x_ij < self.x_max else 1.0
                
                diff = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j] - np.log(x_ij + 1e-10)
                loss = weight * diff * diff
                total_loss += loss
                
                grad = 2 * weight * diff
                
                # Update with proper gradient
                W_grad = grad * W_tilde[j]
                W_tilde_grad = grad * W[i]
                
                W[i] -= self.learning_rate * W_grad
                W_tilde[j] -= self.learning_rate * W_tilde_grad
                b[i] -= self.learning_rate * grad
                b_tilde[j] -= self.learning_rate * grad
            
            if (epoch + 1) % 5 == 0:
                print(f"GloVe Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")
        
        # Average the two context representations (don't normalize - causes issues)
        self.word_vectors = (W + W_tilde) / 2.0
        
        print(f"Final embedding stats - Mean: {self.word_vectors.mean():.4f}, Std: {self.word_vectors.std():.4f}")
        
        return self.word_vectors
    
    def get_embedding_matrix(self):
        if self.word_vectors is None:
            raise ValueError("Model not trained yet")
        
        return self.word_vectors, self.word_to_idx
    
    def save(self, filepath):
        if self.word_vectors is None:
            raise ValueError("Model not trained yet")
        with open(filepath, 'wb') as f:
            pickle.dump((self.word_vectors, self.word_to_idx), f)
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.word_vectors, self.word_to_idx = pickle.load(f)
        return self.word_vectors
