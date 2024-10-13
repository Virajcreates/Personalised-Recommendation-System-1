# app.py

import torch
import pickle
from model import NCFModel  # Ensure model.py is accessible
import gradio as gr

def load_model():
    # Load encoders
    with open('user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)

    with open('item_encoder.pkl', 'rb') as f:
        item_encoder = pickle.load(f)

    with open('user_positive_items.pkl', 'rb') as f:
        user_positive_items = pickle.load(f)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NCFModel(num_users, num_items, embedding_size=50).to(device)
    model.load_state_dict(torch.load('best_ncf_model.pth', map_location=device))
    model.eval()

    return model, user_encoder, item_encoder, user_positive_items, device

model, user_encoder, item_encoder, user_positive_items, device = load_model()

def recommend(user_id, num_recommendations=5):
    # Encode user
    try:
        user = user_encoder.transform([user_id])[0]
    except:
        return "User not found."

    # Get positive items
    pos_items = user_positive_items.get(user, set())

    # Get all items
    all_items = set(range(num_items))

    # Negative candidates
    neg_candidates = list(all_items - pos_items)

    # Prepare tensors
    items_to_predict = neg_candidates
    user_tensor = torch.tensor([user]*len(items_to_predict), dtype=torch.long).to(device)
    item_tensor = torch.tensor(items_to_predict, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(user_tensor, item_tensor)
        scores = torch.sigmoid(outputs).cpu().numpy()

    # Get top N
    top_indices = scores.argsort()[-num_recommendations:][::-1]
    top_items = [item_encoder.inverse_transform([items_to_predict[i]])[0] for i in top_indices]
    top_scores = scores[top_indices]

    recommendations = list(zip(top_items, top_scores))
    return recommendations

interface = gr.Interface(
    fn=recommend,
    inputs=["text", gr.inputs.Slider(minimum=1, maximum=20, default=5, label="Number of Recommendations")],
    outputs="text",
    title="NCF Recommendation System",
    description="Enter a user ID to get item recommendations."
)

interface.launch()
