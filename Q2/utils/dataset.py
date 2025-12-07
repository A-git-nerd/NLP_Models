import torch
from torch.utils.data import Dataset
import numpy as np

class EmbeddingDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def text_to_indices(self, text):
        words = text.split()
        indices = []
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx.get('<UNK>', 1))
        
        if len(indices) < self.max_len:
            indices += [self.word_to_idx.get('<PAD>', 0)] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return indices
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = self.text_to_indices(text)
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class ELMoDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)
