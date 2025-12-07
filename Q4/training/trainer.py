import torch
import torch.nn as nn
from tqdm import tqdm
import time

def train_seq2seq(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    
    for src, trg in tqdm(iterator, desc='Training'):
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate_seq2seq(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in iterator:
            src = src.to(device)
            trg = trg.to(device)
            
            output = model(src, trg, 0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def train_model(model, train_iterator, val_iterator, optimizer, criterion, n_epochs, clip, device, scheduler=None):
    best_valid_loss = float('inf')
    train_losses = []
    val_losses = []
    epoch_times = []
    
    total_start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        train_loss = train_seq2seq(model, train_iterator, optimizer, criterion, clip, device)
        valid_loss = evaluate_seq2seq(model, val_iterator, criterion, device)
        
        if scheduler is not None:
            scheduler.step(valid_loss)
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')
    
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    
    return model, train_losses, val_losses, epoch_times, total_training_time
