import torch
from sacrebleu import corpus_bleu

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    
    idx_to_tgt = {idx: word for word, idx in tgt_vocab.items()}
    
    tokens = sentence.lower().split()
    tokens = [src_vocab.get('<SOS>', 2)] + [src_vocab.get(token, src_vocab.get('<UNK>', 1)) for token in tokens] + [src_vocab.get('<EOS>', 3)]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
        
        trg_indexes = [tgt_vocab.get('<SOS>', 2)]
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            with torch.no_grad():
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

def calculate_bleu(model, test_data, src_vocab, tgt_vocab, device):
    references = []
    hypotheses = []
    
    for src_text, tgt_text in test_data:
        translation = translate_sentence(model, src_text, src_vocab, tgt_vocab, device)
        
        references.append([tgt_text])
        hypotheses.append(translation)
    
    bleu = corpus_bleu(hypotheses, references)
    
    return bleu.score, hypotheses, references
