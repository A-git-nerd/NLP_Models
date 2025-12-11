import torch
from torch.utils.data import Dataset
import re
import os

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=50):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def text_to_indices(self, text, vocab):
        words = text.split()
        indices = [vocab.get('<SOS>', 2)]
        
        for word in words:
            indices.append(vocab.get(word, vocab.get('<UNK>', 1)))
        
        indices.append(vocab.get('<EOS>', 3))
        
        if len(indices) < self.max_len:
            indices += [vocab.get('<PAD>', 0)] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return indices
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        tgt_indices = self.text_to_indices(tgt_text, self.tgt_vocab)
        
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

def build_vocab(texts, min_freq=1):
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def load_translation_data():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    english_path = os.path.join(project_root, 'Data', 'EngUrdu', 'english-corpus.txt')
    urdu_path = os.path.join(project_root, 'Data', 'EngUrdu', 'urdu-corpus.txt')
    
    # Read the English corpus
    with open(english_path, 'r', encoding='utf-8') as f:
        english_sentences = [line.strip() for line in f if line.strip()]
    
    # Read the Urdu corpus
    with open(urdu_path, 'r', encoding='utf-8') as f:
        urdu_sentences = [line.strip() for line in f if line.strip()]
    
    # Ensure both corpora have the same number of lines
    min_len = min(len(english_sentences), len(urdu_sentences))
    english_sentences = english_sentences[:min_len]
    urdu_sentences = urdu_sentences[:min_len]
    
    # Clean the English sentences
    english_sentences = [clean_text(sent) for sent in english_sentences]
    
    # Split into train and test (80-20 split)
    split_idx = int(0.8 * len(english_sentences))
    
    train_en = english_sentences[:split_idx]
    train_ur = urdu_sentences[:split_idx]
    test_en = english_sentences[split_idx:]
    test_ur = urdu_sentences[split_idx:]
    
    return train_en, train_ur, test_en, test_ur
