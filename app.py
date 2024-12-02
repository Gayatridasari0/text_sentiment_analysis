from flask import Flask, request, render_template
import torch
import matplotlib.pyplot as plt
import io
import base64
import os
from transformers import LongformerTokenizer

# Load the Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model classes (ensure these match your trained model's architecture)
class Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = torch.nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_scores = self.attn(lstm_output).squeeze(2)
        return torch.nn.functional.softmax(attention_scores, dim=1)

class SentimentClassifierWithSoftAttention(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        embedding_dim=256,
        hidden_dim=512,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(tokenizer), embedding_dim)
        self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.act = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, return_attention_weights=False):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        attention_weights = self.attention(lstm_output)
        attention_weights = attention_weights.unsqueeze(2)
        weighted = lstm_output * attention_weights
        weighted_sum = weighted.sum(dim=1)
        dense_outputs = self.fc(weighted_sum)
        outputs = self.act(dense_outputs)
        if return_attention_weights:
            return outputs, attention_weights
        return outputs

# Function to load the trained model from the .pth file
def load_trained_model(filepath, tokenizer, device):
    model = SentimentClassifierWithSoftAttention(
        tokenizer=tokenizer,
        embedding_dim=256,
        hidden_dim=512,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model

# Path to your saved model (.pth file)
filepath = "prediction_attention_bilstm_SentimentClassifierWithSoftAttention.pth"

# Load the trained model
trained_model = load_trained_model(filepath, tokenizer, device)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def classify_review():
    if request.method == 'POST':
        # Get the user review
        review = request.form['review']

        # Tokenize the review using the Longformer tokenizer
        inputs = tokenizer(review, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to(device)

        # Get predictions and attention weights
        with torch.no_grad():
            outputs, attention_weights = trained_model(input_ids, return_attention_weights=True)
            prediction = outputs.argmax(dim=1).item()
            sentiment = "Positive" if prediction == 1 else "Negative"

        # Convert attention weights and tokens for visualization
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
        attention_weights = attention_weights.squeeze().cpu().numpy()

        # Clean tokens (optional, to remove special tokens like [CLS], [SEP])
        tokens, attention_weights = clean_special_tokens(tokens, attention_weights)

        # Visualize attention weights
        plt.figure(figsize=(12, 6))
        plt.bar(tokens, attention_weights, color='blue')
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Tokens")
        plt.ylabel("Attention Weights")
        plt.title("Attention Weights for Review")
        plt.tight_layout()

        # Convert the plot to a base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', sentiment=sentiment, plot_url=plot_url)

    return render_template('index.html')

def clean_special_tokens(tokens, attention_weights):
    """
    Remove special tokens like [CLS], [SEP], and adjust attention weights accordingly.

    Args:
        tokens: List of tokens from the tokenizer.
        attention_weights: Corresponding attention weights.

    Returns:
        Cleaned tokens and attention weights.
    """
    cleaned_tokens = []
    cleaned_weights = []
    for token, weight in zip(tokens, attention_weights):
        if token not in tokenizer.all_special_tokens:
            # Remove 'Ġ' from tokens if present
            token = token.replace('Ġ', '').strip()
            cleaned_tokens.append(token)
            cleaned_weights.append(weight)
    return cleaned_tokens, cleaned_weights

if __name__ == '__main__':
    app.run(debug=True)
