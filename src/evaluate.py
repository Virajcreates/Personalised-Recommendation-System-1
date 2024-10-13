# src/evaluate.py

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    """
    Evaluate the NCF model on the test data.
    
    Args:
        model (nn.Module): Trained NCF model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the evaluation on.
    
    Returns:
        tuple: Test accuracy and ROC-AUC scores.
    """
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user, item, label = batch
            user = user.to(device, non_blocking=True)
            item = item.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            outputs = model(user, item)
            preds = torch.sigmoid(outputs).cpu().numpy()  # Apply sigmoid to get probabilities
            labels = label.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Compute binary predictions
    pred_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    accuracy = accuracy_score(all_labels, pred_binary)
    roc_auc = roc_auc_score(all_labels, all_preds)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    
    return accuracy, roc_auc
