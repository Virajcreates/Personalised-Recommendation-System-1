# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import GradScaler, autocast

def train_model(model, train_loader, test_loader, device, num_epochs=10, patience=3, learning_rate=0.001, weight_decay=1e-5):
    """
    Train the NCF model with the given data loaders and parameters.
    
    Args:
        model (nn.Module): The NCF model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the training on.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) parameter.
    
    Returns:
        nn.Module: Trained model with best validation loss.
        dict: Dictionary containing training and validation metrics.
    """
    # Define loss function and optimizer with weight decay for regularization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    trigger_times = 0
    
    # Dictionaries to store metrics
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for batch in tqdm(train_loader, desc="Training"):
            user, item, label = batch
            user = user.to(device, non_blocking=True)
            item = item.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type):
                outputs = model(user, item)
                loss = criterion(outputs, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * user.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            epoch_acc += (preds == label).sum().item()
            total += user.size(0)
        
        epoch_loss /= total
        epoch_acc /= total
        metrics['train_loss'].append(epoch_loss)
        metrics['train_accuracy'].append(epoch_acc)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                user, item, label = batch
                user = user.to(device, non_blocking=True)
                item = item.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                
                with autocast(device_type=device.type):
                    outputs = model(user, item)
                    loss = criterion(outputs, label)
                
                val_loss += loss.item() * user.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_acc += (preds == label).sum().item()
                val_total += user.size(0)
        
        val_loss /= val_total
        val_acc /= val_total
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_acc)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'models/best_ncf_model.pth')
            print("Model saved!")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered!")
                break
    
    return model, metrics
