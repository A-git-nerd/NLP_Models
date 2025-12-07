import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=256, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, device='cpu'):
        super(TransformerSeq2Seq, self).__init__()
        
        self.device = device
        self.embedding_dim = embedding_dim
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout)
        
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def make_src_mask(self, src):
        src_mask = (src == 0).to(self.device)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        tgt_len = tgt.shape[1]
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)
        tgt_padding_mask = (tgt == 0).to(self.device)
        return tgt_mask, tgt_padding_mask
    
    def forward(self, src, tgt):
        src_embedded = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.embedding_dim)))
        tgt_embedded = self.dropout(self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.embedding_dim)))
        
        src_padding_mask = self.make_src_mask(src)
        tgt_mask, tgt_padding_mask = self.make_tgt_mask(tgt)
        
        output = self.transformer(
            src_embedded,
            tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        output = self.fc_out(output)
        return output
