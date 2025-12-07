import torch
import torch.nn as nn

class LSTMWithPretrainedEmbedding(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, num_layers=2, dropout=0.3, freeze_embedding=False):
        super(LSTMWithPretrainedEmbedding, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_matrix))
        
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        out = self.fc(hidden)
        return out
