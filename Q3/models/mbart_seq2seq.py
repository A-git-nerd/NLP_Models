import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class MBARTSeq2Seq(nn.Module):
    def __init__(self, model_name='facebook/mbart-large-50-many-to-many-mmt', device='cpu'):
        super(MBARTSeq2Seq, self).__init__()
        self.device = device
        self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids=None):
        if tgt_input_ids is not None:
            # Training mode
            outputs = self.model(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                labels=tgt_input_ids
            )
            return outputs.loss
        else:
            # Inference mode
            urdu_code = None
            for code in ["ur_PK", "ur_IN", "ur"]:
                if code in self.tokenizer.lang_code_to_id:
                    urdu_code = self.tokenizer.lang_code_to_id[code]
                    break
            
            if urdu_code is None:
                urdu_code = self.tokenizer.lang_code_to_id.get("hi_IN", 250001)
            
            return self.model.generate(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                forced_bos_token_id=urdu_code,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
    
    def translate(self, text):
        """Translate a single text from English to Urdu"""
        self.eval()
        
        # Set source language to English for tokenization
        self.tokenizer.src_lang = "en_XX"
        
        # Tokenize input
        encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        urdu_code = None
        for code in ["ur_PK", "ur_IN", "ur"]:
            if code in self.tokenizer.lang_code_to_id:
                urdu_code = self.tokenizer.lang_code_to_id[code]
                break
        
        if urdu_code is None:
            urdu_code = self.tokenizer.lang_code_to_id.get("hi_IN", 250001)
        
        with torch.no_grad():
            # Generate translation with Urdu as forced target language
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=urdu_code,
                max_length=50,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )
        
        # Decode the generated tokens
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
