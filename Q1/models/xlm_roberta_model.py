import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class XLMRobertaModel_Custom(nn.Module):
    def __init__(self, dropout=0.3):
        super(XLMRobertaModel_Custom, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        out = self.fc(pooled_output)
        return out
