# src/model.py

import torch
import torch.nn as nn

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50):
        """
        Initialize the NCF model with embedding layers and fully connected layers.
        
        Args:
            num_users (int): Total number of unique users.
            num_items (int): Total number of unique items.
            embedding_size (int): Size of the embedding vectors.
        """
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, user, item):
        """
        Forward pass through the model.
        
        Args:
            user (torch.LongTensor): Tensor of user IDs.
            item (torch.LongTensor): Tensor of item IDs.
        
        Returns:
            torch.Tensor: Output logits indicating interaction likelihood.
        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)  # No sigmoid here; handled in loss function
        return x.squeeze()
