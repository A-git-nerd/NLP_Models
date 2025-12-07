import torch
import torch.nn as nn

class RNNModel(nn.Module):
    #embedding 128 words from vocab, hidden layers(memory), num_layers (2 stacked RNN),rand drop 30% connections(forget) 
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout) #regularization to prevent overfitting
        self.fc = nn.Linear(hidden_dim, 2) #classification
    
    #forward pass
    def forward(self, x): #x is list of array + zero padding
        embedded = self.embedding(x) #emb layer
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(hidden[-1]) #set 30% rand samples to zero
        out = self.fc(hidden)
        return out
