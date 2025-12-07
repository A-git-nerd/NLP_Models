import numpy as np
import torch
import torch.nn as nn
from collections import Counter

class CharLevelBiLSTM(nn.Module):
    """Character-level BiLSTM for contextual embeddings (ELMo-like)"""
    def __init__(self, char_vocab_size, char_embedding_dim=64, hidden_dim=512):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(char_embedding_dim, hidden_dim, num_layers=2, 
                             bidirectional=True, batch_first=True)
        self.output_dim = hidden_dim * 2
    
    def forward(self, char_ids):
        # char_ids: (batch_size, max_chars)
        char_embeds = self.char_embedding(char_ids)
        lstm_out, _ = self.bilstm(char_embeds)
        # Take mean pooling over sequence
        return lstm_out.mean(dim=1)

class ELMoEmbedding:
    """Simplified ELMo-like embedding using character-level BiLSTM"""
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim
        self.char_to_idx = None
        self.model = None
        self.max_chars = 200
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_char_vocab(self, texts):
        """Build character vocabulary from texts"""
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, char in enumerate(sorted(all_chars), start=2):
            self.char_to_idx[char] = idx
        
        return len(self.char_to_idx)
    
    def text_to_char_ids(self, text):
        """Convert text to character IDs"""
        char_ids = [self.char_to_idx.get(c, 1) for c in text[:self.max_chars]]
        if len(char_ids) < self.max_chars:
            char_ids += [0] * (self.max_chars - len(char_ids))
        return char_ids
    
    def initialize(self, texts):
        """Initialize the character-level model"""
        char_vocab_size = self.build_char_vocab(texts)
        self.model = CharLevelBiLSTM(char_vocab_size, char_embedding_dim=64, 
                                     hidden_dim=self.embedding_dim // 2)
        self.model.to(self.device)
        self.model.eval()
        return self.model
    
    def get_sentence_embedding(self, sentences, batch_size=32):
        """Get contextual embeddings for sentences"""
        if self.model is None:
            self.initialize(sentences)
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                
                # Convert to character IDs
                char_ids_batch = [self.text_to_char_ids(sent) for sent in batch_sentences]
                char_ids_tensor = torch.tensor(char_ids_batch, dtype=torch.long).to(self.device)
                
                # Get embeddings
                embeddings = self.model(char_ids_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self):
        return self.embedding_dim
