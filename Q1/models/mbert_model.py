import torch
import torch.nn as nn
from transformers import BertModel

class MBERTModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(MBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        out = self.fc(pooled_output)
        return out
