import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re

class SimpleTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        # basic tokenization
        all_words = []
        for text in texts:
            words = text.split() # Simple split by whitespace
            all_words.extend(words)
        
        counter = Counter(all_words)
        most_common = counter.most_common(self.max_vocab_size - 2) # reserve for PAD, UNK
        
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.vocab_size = len(self.word2idx)
        
    def encode(self, text, max_len):
        words = text.split()
        indices = [self.word2idx.get(w, 1) for w in words] # 1 is UNK
        
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices)) # 0 is PAD
        else:
            indices = indices[:max_len]
            
        return indices

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, is_transformer=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_transformer = is_transformer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.is_transformer:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.float)
            }
        else:
            # For RNNs
            input_ids = self.tokenizer.encode(text, self.max_len)
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }

def load_and_process_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    # Clean column names if necessary
    df.columns = df.columns.str.strip()
    
    # Map classes to 0 and 1
    # 'Class' column with 'P' and 'N'
    label_map = {'P': 1, 'N': 0}
    df['label'] = df['Class'].map(label_map)
    
    # Filter out any rows that didn't map correctly (if any)
    df = df.dropna(subset=['label'])
    
    texts = df['Tweet'].tolist()
    labels = df['label'].tolist()
    
    return texts, labels

def create_data_loaders(texts, labels, tokenizer, batch_size, max_len, is_transformer=False):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42
    )
    
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, max_len, is_transformer)
    test_ds = SentimentDataset(test_texts, test_labels, tokenizer, max_len, is_transformer)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, test_loader
