import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import load_translation_data, build_vocab, TranslationDataset
from models.rnn_seq2seq import RNNEncoder, RNNDecoder, RNNSeq2Seq
from models.birnn_seq2seq import BiRNNEncoder, BiRNNDecoder, BiRNNSeq2Seq
from models.lstm_seq2seq import LSTMEncoder, LSTMDecoder, LSTMSeq2Seq
from models.transformer_seq2seq import TransformerSeq2Seq
from training.trainer import train_model
from evaluation.metrics import calculate_bleu, translate_sentence

def train_rnn_model(train_loader, val_loader, src_vocab_size, tgt_vocab_size, device, hyperparams):
    print("\n" + "="*80)
    print("Training RNN Seq2Seq Model")
    print("="*80)
    
    encoder = RNNEncoder(
        src_vocab_size, 
        hyperparams['embedding_dim'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_layers'], 
        hyperparams['dropout']
    )
    decoder = RNNDecoder(
        tgt_vocab_size, 
        hyperparams['embedding_dim'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_layers'], 
        hyperparams['dropout']
    )
    
    model = RNNSeq2Seq(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        hyperparams['num_epochs'], hyperparams['clip'], device, is_transformer=False
    )
    
    return model

def train_birnn_model(train_loader, val_loader, src_vocab_size, tgt_vocab_size, device, hyperparams):
    print("\n" + "="*80)
    print("Training BiRNN Seq2Seq Model")
    print("="*80)
    
    encoder = BiRNNEncoder(
        src_vocab_size, 
        hyperparams['embedding_dim'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_layers'], 
        hyperparams['dropout']
    )
    decoder = BiRNNDecoder(
        tgt_vocab_size, 
        hyperparams['embedding_dim'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_layers'], 
        hyperparams['dropout']
    )
    
    model = BiRNNSeq2Seq(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        hyperparams['num_epochs'], hyperparams['clip'], device, is_transformer=False
    )
    
    return model

def train_lstm_model(train_loader, val_loader, src_vocab_size, tgt_vocab_size, device, hyperparams):
    print("\n" + "="*80)
    print("Training LSTM Seq2Seq Model")
    print("="*80)
    
    encoder = LSTMEncoder(
        src_vocab_size, 
        hyperparams['embedding_dim'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_layers'], 
        hyperparams['dropout']
    )
    decoder = LSTMDecoder(
        tgt_vocab_size, 
        hyperparams['embedding_dim'], 
        hyperparams['hidden_dim'], 
        hyperparams['num_layers'], 
        hyperparams['dropout']
    )
    
    model = LSTMSeq2Seq(encoder, decoder, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        hyperparams['num_epochs'], hyperparams['clip'], device, is_transformer=False
    )
    
    return model

def train_transformer_model(train_loader, val_loader, src_vocab_size, tgt_vocab_size, device, hyperparams):
    print("\n" + "="*80)
    print("Training Transformer Model")
    print("="*80)
    
    model = TransformerSeq2Seq(
        src_vocab_size,
        tgt_vocab_size,
        embedding_dim=hyperparams['embedding_dim'],
        nhead=hyperparams['nhead'],
        num_encoder_layers=hyperparams['num_layers'],
        num_decoder_layers=hyperparams['num_layers'],
        dim_feedforward=hyperparams['dim_feedforward'],
        dropout=hyperparams['dropout'],
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        hyperparams['num_epochs'], hyperparams['clip'], device, is_transformer=True
    )
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_en, train_ur, test_en, test_ur = load_translation_data()
    
    print(f"Train samples: {len(train_en)}")
    print(f"Test samples: {len(test_en)}")
    
    src_vocab = build_vocab(train_en)
    tgt_vocab = build_vocab(train_ur)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    train_dataset = TranslationDataset(train_en, train_ur, src_vocab, tgt_vocab, max_len=50)
    val_dataset = TranslationDataset(test_en, test_ur, src_vocab, tgt_vocab, max_len=50)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    rnn_hyperparams = {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'clip': 1,
        'optimizer': 'Adam'
    }
    
    transformer_hyperparams = {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 3,
        'nhead': 8,
        'dim_feedforward': 512,
        'dropout': 0.3,
        'learning_rate': 0.0001,
        'num_epochs': 50,
        'clip': 1,
        'optimizer': 'Adam'
    }
    
    models = {}
    hyperparams_dict = {}
    
    rnn_model = train_rnn_model(train_loader, val_loader, len(src_vocab), len(tgt_vocab), device, rnn_hyperparams)
    models['RNN Seq2Seq'] = rnn_model
    hyperparams_dict['RNN Seq2Seq'] = rnn_hyperparams
    
    birnn_model = train_birnn_model(train_loader, val_loader, len(src_vocab), len(tgt_vocab), device, rnn_hyperparams)
    models['BiRNN Seq2Seq'] = birnn_model
    hyperparams_dict['BiRNN Seq2Seq'] = rnn_hyperparams
    
    lstm_model = train_lstm_model(train_loader, val_loader, len(src_vocab), len(tgt_vocab), device, rnn_hyperparams)
    models['LSTM Seq2Seq'] = lstm_model
    hyperparams_dict['LSTM Seq2Seq'] = rnn_hyperparams
    
    transformer_model = train_transformer_model(train_loader, val_loader, len(src_vocab), len(tgt_vocab), device, transformer_hyperparams)
    models['Transformer'] = transformer_model
    hyperparams_dict['Transformer'] = transformer_hyperparams
    
    print("\n" + "="*80)
    print("EVALUATION - BLEU SCORES")
    print("="*80)
    
    results = {}
    test_data = list(zip(test_en, test_ur))
    
    for model_name, model in models.items():
        is_transformer = (model_name == 'Transformer')
        bleu_score, hypotheses, references = calculate_bleu(model, test_data, src_vocab, tgt_vocab, device, is_transformer)
        results[model_name] = {
            'BLEU Score': bleu_score,
            'Hypotheses': hypotheses,
            'References': references
        }
        print(f"{model_name}: BLEU = {bleu_score:.2f}")
    
    print("\n" + "="*80)
    print("INFERENCE EXAMPLES")
    print("="*80)
    
    test_sentences = test_en[:5]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nExample {i+1}:")
        print(f"Source (English): {sentence}")
        print(f"Reference (Urdu): {test_ur[i]}")
        print("Translations:")
        
        for model_name, model in models.items():
            is_transformer = (model_name == 'Transformer')
            translation = translate_sentence(model, sentence, src_vocab, tgt_vocab, device, is_transformer=is_transformer)
            print(f"  {model_name}: {translation}")
    
    print("\n" + "="*80)
    print("HYPERPARAMETERS")
    print("="*80)
    
    for model_name, params in hyperparams_dict.items():
        print(f"\n{model_name}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    os.makedirs('results', exist_ok=True)
    
    bleu_scores = {name: res['BLEU Score'] for name, res in results.items()}
    results_df = pd.DataFrame([bleu_scores])
    results_df.to_csv('results/bleu_scores.csv', index=False)
    
    hyperparams_df = pd.DataFrame(hyperparams_dict).T
    hyperparams_df.to_csv('results/hyperparameters.csv')
    
    with open('results/inference_examples.txt', 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(test_sentences):
            f.write(f"Example {i+1}:\n")
            f.write(f"Source (English): {sentence}\n")
            f.write(f"Reference (Urdu): {test_ur[i]}\n")
            f.write("Translations:\n")
            
            for model_name in models.keys():
                translation = results[model_name]['Hypotheses'][i]
                f.write(f"  {model_name}: {translation}\n")
            f.write("\n")
    
    print("\nResults saved to 'results/' directory")

if __name__ == "__main__":
    main()
