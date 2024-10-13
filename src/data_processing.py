# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import defaultdict

def load_data(data_dir):
    """
    Load datasets from the specified directory.
    
    Args:
        data_dir (str): Path to the data directory.
    
    Returns:
        dict: Dictionary containing loaded DataFrames.
    """
    events = pd.read_csv(f'{data_dir}events.csv')
    category_tree = pd.read_csv(f'{data_dir}category_tree.csv')
    item_properties_part1 = pd.read_csv(f'{data_dir}item_properties_part1.csv')
    item_properties_part2 = pd.read_csv(f'{data_dir}item_properties_part2.csv')
    
    data = {
        'events': events,
        'category_tree': category_tree,
        'item_properties_part1': item_properties_part1,
        'item_properties_part2': item_properties_part2
    }
    
    return data

def preprocess_data(data):
    """
    Preprocess the loaded data by handling missing values, merging catalogs, and filtering events.
    
    Args:
        data (dict): Dictionary containing loaded DataFrames.
    
    Returns:
        tuple: Processed catalog DataFrame and interaction DataFrame.
    """
    # Concatenate item properties
    catalog_part1 = data['item_properties_part1'].copy()
    catalog_part2 = data['item_properties_part2'].copy()
    
    # Ensure consistent column names if necessary
    # Example: Uncomment and modify if column names differ
    # catalog_part2.rename(columns={'prod_id': 'itemid'}, inplace=True)
    
    # Concatenate
    catalog = pd.concat([catalog_part1, catalog_part2], ignore_index=True)
    
    # Handle duplicates
    duplicate_items = catalog[catalog.duplicated(subset=['itemid'], keep=False)]
    print(f"\nNumber of duplicate items: {duplicate_items.shape[0]}")
    catalog.drop_duplicates(subset=['itemid'], keep='first', inplace=True)
    print(f"Catalog shape after removing duplicates: {catalog.shape}")
    
    # Merge with category_tree if applicable
    if 'categoryid' in catalog.columns and 'categoryid' in data['category_tree'].columns:
        catalog = catalog.merge(
            data['category_tree'][['categoryid', 'parentid', 'category_name']],
            on='categoryid',
            how='left'
        )
        print("\nCatalog after merging with Category Tree:")
        print(catalog.head())
    else:
        print("Cannot merge catalog with category_tree.csv. Check the column names for Category IDs.")
    
    # Convert timestamp to datetime
    data['events']['timestamp'] = pd.to_datetime(data['events']['timestamp'], unit='ms')  # Adjust unit if necessary
    
    # Filter relevant events
    relevant_events = data['events'][data['events']['event'].isin(['view', 'purchase'])].copy()
    print(f"\nNumber of relevant events: {relevant_events.shape[0]}")
    
    # Handle missing values in catalog
    if 'ProductDescription' in catalog.columns:
        catalog['ProductDescription'] = catalog['ProductDescription'].fillna('')
    else:
        catalog['ProductDescription'] = ''
    
    return catalog, relevant_events

def encode_ids(interactions):
    """
    Encode user and item IDs using LabelEncoder.
    
    Args:
        interactions (pd.DataFrame): DataFrame containing user-item interactions.
    
    Returns:
        tuple: Encoded interactions DataFrame, user encoder, item encoder.
    """
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    interactions['user'] = user_encoder.fit_transform(interactions['visitorid'].astype(str))
    interactions['item'] = item_encoder.fit_transform(interactions['itemid'].astype(str))
    
    # Select relevant columns
    data = interactions[['user', 'item', 'event']].copy()
    
    # Map event types to interaction values
    interaction_mapping = {'view': 1, 'purchase': 2}
    data['interaction'] = data['event'].map(interaction_mapping)
    
    # Drop the original 'event' column
    data.drop('event', axis=1, inplace=True)
    
    return data, user_encoder, item_encoder

def generate_negative_samples_vectorized(df, num_negatives=4):
    """
    Generate negative samples for each user using vectorized operations.
    
    Args:
        df (pd.DataFrame): DataFrame containing user-item interactions.
        num_negatives (int): Number of negative samples per user.
    
    Returns:
        pd.DataFrame: DataFrame containing negative samples with columns ['user', 'item', 'label'].
    """
    users = df['user'].unique()
    all_items = df['item'].unique()
    all_items_set = set(all_items)
    
    # Create a mapping from user to set of positive items
    user_to_pos_items = df.groupby('user')['item'].apply(set).to_dict()
    
    negative_users = []
    negative_items = []
    negative_labels = []
    
    for user in tqdm(users, desc="Generating Negative Samples"):
        pos_items = user_to_pos_items[user]
        neg_candidates = list(all_items_set - pos_items)
        
        if len(neg_candidates) >= num_negatives:
            sampled_negatives = np.random.choice(neg_candidates, size=num_negatives, replace=False)
        else:
            # If not enough candidates, sample with replacement
            sampled_negatives = np.random.choice(neg_candidates, size=num_negatives, replace=True)
        
        negative_users.extend([user] * num_negatives)
        negative_items.extend(sampled_negatives)
        negative_labels.extend([0] * num_negatives)
    
    negative_df = pd.DataFrame({
        'user': negative_users,
        'item': negative_items,
        'label': negative_labels
    })
    
    return negative_df
