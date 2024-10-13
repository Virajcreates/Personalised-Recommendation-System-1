# NCF Recommendation System

This repository contains a trained Neural Collaborative Filtering (NCF) model for item recommendations based on user interactions.

## Files

- `best_ncf_model.pth`: Trained PyTorch model weights.
- `user_encoder.pkl`: LabelEncoder for users.
- `item_encoder.pkl`: LabelEncoder for items.
- `user_positive_items.pkl`: Mapping of users to their positive items.

## Usage

To use the model, load the weights and encoders using PyTorch and pickle, respectively.

```python
import torch
import pickle
from model import NCFModel  # Ensure model.py is accessible

# Load encoders
with open('user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

with open('item_encoder.pkl', 'rb') as f:
    item_encoder = pickle.load(f)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NCFModel(num_users, num_items, embedding_size=50)
model.load_state_dict(torch.load('best_ncf_model.pth'))
model.to(device)
model.eval()

# Make predictions...
