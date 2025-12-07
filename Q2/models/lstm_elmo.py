import torch
import torch.nn as nn

class LSTMWithELMo(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super(LSTMWithELMo, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        output, (hidden, cell) = self.lstm(x)
        hidden = self.dropout(hidden[-1])
        out = self.fc(hidden)
        return out
