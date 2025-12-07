import torch
from sacrebleu import corpus_bleu

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=50, is_transformer=False):
    model.eval()
    
    idx_to_tgt = {idx: word for word, idx in tgt_vocab.items()}
    
    tokens = sentence.lower().split()
    tokens = [src_vocab.get('<SOS>', 2)] + [src_vocab.get(token, src_vocab.get('<UNK>', 1)) for token in tokens] + [src_vocab.get('<EOS>', 3)]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_transformer:
            trg_indexes = [tgt_vocab.get('<SOS>', 2)]
            
            for i in range(max_len):
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
                
                output = model(src_tensor, trg_tensor)
                pred_token = output.argmax(2)[:, -1].item()
                trg_indexes.append(pred_token)
                
                if pred_token == tgt_vocab.get('<EOS>', 3):
                    break
        else:
            if hasattr(model.encoder, 'lstm'):
                hidden, cell = model.encoder(src_tensor)
            else:
                hidden = model.encoder(src_tensor)
                cell = None
            
            trg_indexes = [tgt_vocab.get('<SOS>', 2)]
            
            for i in range(max_len):
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
                with torch.no_grad():
                    if cell is not None:
                        output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
                    else:
                        output, hidden = model.decoder(trg_tensor, hidden)
                
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                
                if pred_token == tgt_vocab.get('<EOS>', 3):
                    break
    
    trg_tokens = [idx_to_tgt.get(idx, '<UNK>') for idx in trg_indexes]
    
    if '<SOS>' in trg_tokens:
        trg_tokens.remove('<SOS>')
    if '<EOS>' in trg_tokens:
        trg_tokens.remove('<EOS>')
    
    return ' '.join(trg_tokens)

def calculate_bleu(model, test_data, src_vocab, tgt_vocab, device, is_transformer=False):
    references = []
    hypotheses = []
    
    for src_text, tgt_text in test_data:
        translation = translate_sentence(model, src_text, src_vocab, tgt_vocab, device, is_transformer=is_transformer)
        
        references.append([tgt_text])
        hypotheses.append(translation)
    
    bleu = corpus_bleu(hypotheses, references)
    
    return bleu.score, hypotheses, references
