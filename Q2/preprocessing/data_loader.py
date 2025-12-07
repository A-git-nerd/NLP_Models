import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

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
