import streamlit as st
import torch
import matplotlib.pyplot as plt
from transformers import LongformerTokenizer

# Load the tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model definition (similar to before)
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


# Load the trained model
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


filepath = "prediction_attention_bilstm_SentimentClassifierWithSoftAttention.pth"
trained_model = load_trained_model(filepath, tokenizer, device)

# Streamlit app layout
st.title("Sentiment Analysis with Attention Visualization")
st.write("Enter a review to analyze its sentiment and visualize attention weights.")

def clean_special_tokens(tokens, attention_weights):
    cleaned_tokens = []
    cleaned_weights = []
    for token, weight in zip(tokens, attention_weights):
        if token not in tokenizer.all_special_tokens:
            token = token.replace('Ġ', '').strip()
            cleaned_tokens.append(token)
            cleaned_weights.append(weight)
    return cleaned_tokens, cleaned_weights

# Input review
review = st.text_area("Enter your review here:")

if st.button("Submit"):
    if review.strip():
        # Tokenize the review
        inputs = tokenizer(review, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs["input_ids"].to(device)

        # Get predictions and attention weights
        with torch.no_grad():
            outputs, attention_weights = trained_model(input_ids, return_attention_weights=True)
            prediction = outputs.argmax(dim=1).item()
            sentiment = "Positive" if prediction == 1 else "Negative"

        # Display the sentiment
        st.subheader(f"Predicted Sentiment: {sentiment}")

        # Convert attention weights and tokens for visualization
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
        attention_weights = attention_weights.squeeze().cpu().numpy()
        tokens, attention_weights = clean_special_tokens(tokens, attention_weights)

        # Visualize attention weights
        plt.figure(figsize=(12, 6))
        plt.bar(tokens, attention_weights, color="blue")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Tokens")
        plt.ylabel("Attention Weights")
        plt.title("Attention Weights for Review")
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)
    else:
        st.error("Please enter a review to analyze.")



