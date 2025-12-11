import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import load_and_preprocess_data, UrduSentimentDataset, TransformerDataset
from models.rnn_model import RNNModel
from models.gru_model import GRUModel
from models.lstm_model import LSTMModel
from models.bilstm_model import BiLSTMModel
from models.mbert_model import MBERTModel
from models.xlm_roberta_model import XLMRobertaModel_Custom
from training.trainer import train_model
from evaluation.metrics import evaluate_model
from utils.plotting import plot_training_history

def train_sequence_model(model_class, model_name, train_dataset, test_dataset, hyperparams, device, class_weights=None):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
    
    vocab_size = len(train_dataset.vocab)
    model = model_class(
        vocab_size=vocab_size,
        embedding_dim=hyperparams['embedding_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    ).to(device)
    
    # Use class weights if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    model, history = train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                       hyperparams['num_epochs'], is_transformer=False)
    
    results = evaluate_model(model, test_loader, device, is_transformer=False)
    
    return results, hyperparams, history

def train_transformer_model(model_class, model_name, tokenizer_class, tokenizer_name, 
                           train_texts, train_labels, test_texts, test_labels, 
                           hyperparams, device, class_weights=None):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    
    train_dataset = TransformerDataset(train_texts, train_labels, tokenizer, hyperparams['max_len'])
    test_dataset = TransformerDataset(test_texts, test_labels, tokenizer, hyperparams['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
    
    model = model_class(dropout=hyperparams['dropout']).to(device)
    
    # Use class weights if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'])
    
    model, history = train_model(model, train_loader, test_loader, criterion, optimizer, device, 
                       hyperparams['num_epochs'], is_transformer=True)
    
    results = evaluate_model(model, test_loader, device, is_transformer=True)
    
    return results, hyperparams, history

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = '../Data/Tweet/urdu-sentiment-corpus-v1.tsv'
    train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data(data_path)
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Print class distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"\nClass distribution in training data:")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(train_labels)*100:.1f}%)")
    
    # Calculate class weights for handling imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights: {class_weights}")
    
    train_dataset = UrduSentimentDataset(train_texts, train_labels, vocab=None, max_len=100)
    test_dataset = UrduSentimentDataset(test_texts, test_labels, vocab=train_dataset.vocab, max_len=100)
    
    results_dict = {}
    hyperparams_dict = {}
    histories = {}
    
    sequence_hyperparams = {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'learning_rate': 0.005,  # Increased from 0.001
        'num_epochs': 20  # Increased from 10
    }
    
    # LSTM-specific hyperparameters (much more conservative to prevent mode collapse)
    lstm_hyperparams = {
        'embedding_dim': 128,
        'hidden_dim': 128,  # Reduced from 256 to prevent overfitting
        'num_layers': 1,  # Reduced from 2 - simpler model
        'dropout': 0.5,  # Increased from 0.3 for more regularization
        'batch_size': 16,  # Smaller batches for more stable gradients
        'learning_rate': 0.0005,  # Much lower learning rate
        'num_epochs': 30  # More epochs to compensate
    }
    
    models_to_train = [
        (RNNModel, 'RNN', sequence_hyperparams),
        (GRUModel, 'GRU', sequence_hyperparams),
        (LSTMModel, 'LSTM', lstm_hyperparams),  # Use LSTM-specific params
        (BiLSTMModel, 'BiLSTM', sequence_hyperparams)
    ]
    
    for model_class, model_name, hyperparams in models_to_train:
        results, params, history = train_sequence_model(model_class, model_name, train_dataset, 
                                              test_dataset, hyperparams, device, class_weights)
        results_dict[model_name] = results
        hyperparams_dict[model_name] = params
        histories[model_name] = history
    
    transformer_hyperparams = {
        'max_len': 128,
        'dropout': 0.3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 5
    }
    
    results, params, history = train_transformer_model(
        MBERTModel, 'mBERT', BertTokenizer, 'bert-base-multilingual-cased',
        train_texts, train_labels, test_texts, test_labels, 
        transformer_hyperparams, device, class_weights
    )
    results_dict['mBERT'] = results
    hyperparams_dict['mBERT'] = params
    histories['mBERT'] = history
    
    results, params, history = train_transformer_model(
        XLMRobertaModel_Custom, 'XLM-RoBERTa', XLMRobertaTokenizer, 'xlm-roberta-base',
        train_texts, train_labels, test_texts, test_labels, 
        transformer_hyperparams, device, class_weights
    )
    results_dict['XLM-RoBERTa'] = results
    hyperparams_dict['XLM-RoBERTa'] = params
    histories['XLM-RoBERTa'] = history
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results_dict).T
    results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    print("\nPerformance Metrics:")
    print(results_df.to_string())
    
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)
    
    for model_name, params in hyperparams_dict.items():
        print(f"\n{model_name}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    results_df.to_csv('results/performance_metrics.csv')
    
    hyperparams_df = pd.DataFrame(hyperparams_dict).T
    hyperparams_df.to_csv('results/hyperparameters.csv')
    
    # Plot training metrics
    model_names = list(histories.keys())
    plot_training_history(histories, model_names, save_dir='results')
    
    print("\nResults and plots saved to 'results/' directory")

if __name__ == "__main__":
    main()
