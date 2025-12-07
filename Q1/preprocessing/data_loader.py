import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class UrduSentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
        if self.vocab is None:
            self.vocab = self.build_vocab(texts)
    
    def build_vocab(self, texts):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab
    
    def text_to_indices(self, text):
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
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

class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    
    df = df[df['Class'].isin(['P', 'N'])]
    
    df['Tweet'] = df['Tweet'].apply(clean_text)
    
    df = df[df['Tweet'].str.len() > 0]
    
    df['Label'] = df['Class'].map({'P': 1, 'N': 0})
    
    texts = df['Tweet'].tolist()
    labels = df['Label'].tolist()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    return train_texts, test_texts, train_labels, test_labels
