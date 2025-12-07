import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import load_and_preprocess_data
from embeddings.word2vec_embedding import Word2VecEmbedding
from embeddings.glove_embedding import GloveEmbedding
from embeddings.fasttext_embedding import FastTextEmbedding
from embeddings.elmo_embedding import ELMoEmbedding
from models.lstm_no_embedding import LSTMWithoutEmbedding
from models.lstm_pretrained import LSTMWithPretrainedEmbedding
from models.lstm_elmo import LSTMWithELMo
from utils.dataset import EmbeddingDataset, ELMoDataset
from training.trainer import train_model
from evaluation.metrics import evaluate_model

def train_lstm_without_embedding(train_texts, train_labels, test_texts, test_labels, device):
    print("\n" + "="*80)
    print("Training LSTM without embeddings")
    print("="*80)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=1000)
    train_features = vectorizer.fit_transform(train_texts).toarray()
    test_features = vectorizer.transform(test_texts).toarray()
    
    train_features_expanded = np.expand_dims(train_features, axis=1)
    test_features_expanded = np.expand_dims(test_features, axis=1)
    
    train_dataset = list(zip(train_features_expanded, train_labels))
    test_dataset = list(zip(test_features_expanded, test_labels))
    
    train_loader = DataLoader([(torch.FloatTensor(x), torch.LongTensor([y])[0]) for x, y in train_dataset], 
                              batch_size=32, shuffle=True)
    test_loader = DataLoader([(torch.FloatTensor(x), torch.LongTensor([y])[0]) for x, y in test_dataset], 
                             batch_size=32, shuffle=False)
    
    model = LSTMWithoutEmbedding(input_dim=1000, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)
    
    results = evaluate_model(model, test_loader, device)
    
    hyperparams = {
        'input_dim': 1000,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'optimizer': 'Adam'
    }
    
    return results, hyperparams

def train_lstm_with_word2vec(train_texts, train_labels, test_texts, test_labels, device):
    print("\n" + "="*80)
    print("Training LSTM with Word2Vec embeddings")
    print("="*80)
    
    w2v = Word2VecEmbedding(vector_size=128, window=5, min_count=1, epochs=10)
    w2v.train(train_texts)
    
    embedding_matrix, word_to_idx = w2v.get_embedding_matrix()
    
    train_dataset = EmbeddingDataset(train_texts, train_labels, word_to_idx, max_len=100)
    test_dataset = EmbeddingDataset(test_texts, test_labels, word_to_idx, max_len=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = LSTMWithPretrainedEmbedding(embedding_matrix, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)
    
    results = evaluate_model(model, test_loader, device)
    
    hyperparams = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'optimizer': 'Adam',
        'w2v_window': 5,
        'w2v_min_count': 1
    }
    
    return results, hyperparams

def train_lstm_with_glove(train_texts, train_labels, test_texts, test_labels, device):
    print("\n" + "="*80)
    print("Training LSTM with GloVe embeddings")
    print("="*80)
    
    glove = GloveEmbedding(vector_size=128, learning_rate=0.05, epochs=10)
    glove.train(train_texts)
    
    embedding_matrix, word_to_idx = glove.get_embedding_matrix()
    
    train_dataset = EmbeddingDataset(train_texts, train_labels, word_to_idx, max_len=100)
    test_dataset = EmbeddingDataset(test_texts, test_labels, word_to_idx, max_len=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = LSTMWithPretrainedEmbedding(embedding_matrix, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)
    
    results = evaluate_model(model, test_loader, device)
    
    hyperparams = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'optimizer': 'Adam',
        'glove_learning_rate': 0.05,
        'x_max': 100,
        'alpha': 0.75
    }
    
    return results, hyperparams

def train_lstm_with_fasttext(train_texts, train_labels, test_texts, test_labels, device):
    print("\n" + "="*80)
    print("Training LSTM with FastText embeddings")
    print("="*80)
    
    ft = FastTextEmbedding(vector_size=128, window=5, min_count=1, epochs=10)
    ft.train(train_texts)
    
    embedding_matrix, word_to_idx = ft.get_embedding_matrix()
    
    train_dataset = EmbeddingDataset(train_texts, train_labels, word_to_idx, max_len=100)
    test_dataset = EmbeddingDataset(test_texts, test_labels, word_to_idx, max_len=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = LSTMWithPretrainedEmbedding(embedding_matrix, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)
    
    results = evaluate_model(model, test_loader, device)
    
    hyperparams = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'optimizer': 'Adam',
        'ft_window': 5,
        'ft_min_count': 1
    }
    
    return results, hyperparams

def train_lstm_with_elmo(train_texts, train_labels, test_texts, test_labels, device):
    print("\n" + "="*80)
    print("Training LSTM with ELMo embeddings")
    print("="*80)
    
    elmo = ELMoEmbedding()
    
    print("Generating ELMo embeddings for training data...")
    train_embeddings = elmo.get_sentence_embedding(train_texts, batch_size=16)
    
    print("Generating ELMo embeddings for test data...")
    test_embeddings = elmo.get_sentence_embedding(test_texts, batch_size=16)
    
    train_dataset = ELMoDataset(train_embeddings, train_labels)
    test_dataset = ELMoDataset(test_embeddings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = LSTMWithELMo(input_dim=elmo.get_embedding_dim(), hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)
    
    results = evaluate_model(model, test_loader, device)
    
    hyperparams = {
        'embedding_dim': elmo.get_embedding_dim(),
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'optimizer': 'Adam'
    }
    
    return results, hyperparams

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = '../Data/Tweet/urdu-sentiment-corpus-v1.tsv'
    train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data(data_path)
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    results_dict = {}
    hyperparams_dict = {}
    
    result, params = train_lstm_without_embedding(train_texts, train_labels, test_texts, test_labels, device)
    results_dict['LSTM (without embeddings)'] = result
    hyperparams_dict['LSTM (without embeddings)'] = params
    
    result, params = train_lstm_with_word2vec(train_texts, train_labels, test_texts, test_labels, device)
    results_dict['LSTM with Word2Vec'] = result
    hyperparams_dict['LSTM with Word2Vec'] = params
    
    result, params = train_lstm_with_glove(train_texts, train_labels, test_texts, test_labels, device)
    results_dict['LSTM with Glove'] = result
    hyperparams_dict['LSTM with Glove'] = params
    
    result, params = train_lstm_with_fasttext(train_texts, train_labels, test_texts, test_labels, device)
    results_dict['LSTM with Fasttext'] = result
    hyperparams_dict['LSTM with Fasttext'] = params
    
    result, params = train_lstm_with_elmo(train_texts, train_labels, test_texts, test_labels, device)
    results_dict['LSTM with Elmo'] = result
    hyperparams_dict['LSTM with Elmo'] = params
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F-score']
    print("\nPerformance Metrics:")
    print(results_df.to_string())
    
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)
    
    for model_name, params in hyperparams_dict.items():
        print(f"\n{model_name}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/performance_metrics.csv')
    
    hyperparams_df = pd.DataFrame(hyperparams_dict).T
    hyperparams_df.to_csv('results/hyperparameters.csv')
    
    print("\nResults saved to 'results/' directory")

if __name__ == "__main__":
    main()
