import torch
import torch.nn as nn
from tqdm import tqdm

def train_seq2seq(model, iterator, optimizer, criterion, clip, device, is_transformer=False):
    model.train()
    epoch_loss = 0
    
    for src, trg in tqdm(iterator, desc='Training'):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        if is_transformer:
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]
            
            output = model(src, trg_input)
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)
        else:
            output = model(src, trg, teacher_forcing_ratio=0.8)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            trg_output = trg
        
        loss = criterion(output, trg_output)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate_seq2seq(model, iterator, criterion, device, is_transformer=False):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in iterator:
            src = src.to(device)
            trg = trg.to(device)
            
            if is_transformer:
                trg_input = trg[:, :-1]
                trg_output = trg[:, 1:]
                
                output = model(src, trg_input)
                
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_output = trg_output.contiguous().view(-1)
            else:
                output = model(src, trg, 0)
                
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                trg_output = trg
            
            loss = criterion(output, trg_output)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(model, train_iterator, val_iterator, optimizer, criterion, n_epochs, clip, device, is_transformer=False):
    best_valid_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(n_epochs):
        train_loss = train_seq2seq(model, train_iterator, optimizer, criterion, clip, device, is_transformer)
        valid_loss = evaluate_seq2seq(model, val_iterator, criterion, device, is_transformer)
        
        scheduler.step(valid_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | Best Val: {best_valid_loss:.3f}')
    
    return model, history
