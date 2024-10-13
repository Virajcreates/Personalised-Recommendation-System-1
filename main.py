# main.py

import os
import torch
from src import (
    load_data,
    preprocess_data,
    encode_ids,
    generate_negative_samples_vectorized,
    NCFModel,
    train_model,
    evaluate_model
)
from torch.utils.data import DataLoader
from src import InteractionDataset

def main():
    # Define directories
    data_dir = 'data/'
    models_dir = 'models/'
    outputs_dir = 'outputs/'
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load data
    data = load_data(data_dir)
    
    # Preprocess data
    catalog, relevant_events = preprocess_data(data)
    
    # Encode IDs
    interactions, user_encoder, item_encoder = encode_ids(relevant_events)
    
    # Save encoders
    import pickle
    with open(os.path.join(outputs_dir, 'user_encoder.pkl'), 'wb') as f:
        pickle.dump(user_encoder, f)
    
    with open(os.path.join(outputs_dir, 'item_encoder.pkl'), 'wb') as f:
        pickle.dump(item_encoder, f)
    
    # Split data into training and testing sets
    train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Generate negative samples for training
    print("Generating negative samples for training...")
    train_negative = generate_negative_samples_vectorized(train_data, num_negatives=4)
    train_positive = train_data[['user', 'item']].copy()
    train_positive['label'] = 1
    train_combined = pd.concat([train_positive, train_negative], ignore_index=True)
    train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total training samples: {train_combined.shape[0]}")
    
    # Save negative samples
    train_negative.to_pickle(os.path.join(outputs_dir, 'train_negative.pkl'))
    
    # Generate negative samples for testing
    print("Generating negative samples for testing...")
    test_negative = generate_negative_samples_vectorized(test_data, num_negatives=4)
    test_positive = test_data[['user', 'item']].copy()
    test_positive['label'] = 1
    test_combined = pd.concat([test_positive, test_negative], ignore_index=True)
    test_combined = test_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total testing samples: {test_combined.shape[0]}")
    
    # Save negative samples
    test_negative.to_pickle(os.path.join(outputs_dir, 'test_negative.pkl'))
    
    # Define Datasets and DataLoaders
    train_dataset = InteractionDataset(train_combined)
    test_dataset = InteractionDataset(test_combined)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
    
    # Instantiate the model
    num_users = interactions['user'].nunique()
    num_items = interactions['item'].nunique()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    model = NCFModel(num_users, num_items, embedding_size=50).to(device)
    
    # Train the model
    trained_model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=10,
        patience=3,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # Evaluate the model
    accuracy, roc_auc = evaluate_model(trained_model, test_loader, device)
    
    # Save user_positive_items for recommendations
    user_positive_items = defaultdict(set)
    for row in train_data.itertuples(index=False):
        user_positive_items[row.user].add(row.item)
    
    import pickle
    with open(os.path.join(outputs_dir, 'user_positive_items.pkl'), 'wb') as f:
        pickle.dump(user_positive_items, f)
    
    print("\nTraining and evaluation completed successfully.")

if __name__ == "__main__":
    main()
