import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Pandas display settings
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# Load dataset
user_data_path = 'data/Harshil-Filtered.csv'
user_data = pd.read_csv(user_data_path)

# Generate synthetic training data
def generate_training_data(user_data):
    prompts = []
    labels = []

    for _, row in user_data.iterrows():
        if row['energy'] > 0.7 and row['danceability'] > 0.7:
            prompts.append("High-energy dance track")
        elif row['energy'] < 0.4 and row['valence'] < 0.5:
            prompts.append("Chill and calm music")
        elif row['instrumentalness'] > 0.5:
            prompts.append("Relaxing instrumental music")
        else:
            prompts.append("Balanced track for casual listening")

        labels.append([row['energy'], row['danceability'], row['valence'], row['instrumentalness']])

    return pd.DataFrame({"prompt": prompts, "labels": labels})

training_data = generate_training_data(user_data)

# Define Dataset and DataLoader
class PromptFeatureDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.prompts = data["prompt"]
        self.labels = torch.tensor(data["labels"].tolist(), dtype=torch.float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts.iloc[idx]
        tokenized = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": self.labels[idx]
        }

# Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Prepare DataLoader
dataset = PromptFeatureDataset(training_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define Model
class BERTForFeaturePrediction(nn.Module):
    def __init__(self, bert_model_name, output_dim):
        super(BERTForFeaturePrediction, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.regression_head = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        predictions = self.regression_head(pooled_output)
        return predictions

model = BERTForFeaturePrediction("distilbert-base-uncased", output_dim=4)

# Training with better logging
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

from tqdm import tqdm

# Determine number of epochs for 100 total runs
batches_per_epoch = len(dataloader)  # Number of batches in the dataloader
num_epochs = 100 // batches_per_epoch  # Total runs / batches per epoch

print(f"Total Epochs: {num_epochs}, Batches per Epoch: {batches_per_epoch}, Total Runs: {num_epochs * batches_per_epoch}")

# Training with progress bar
for epoch in range(num_epochs):
    total_loss = 0
    # Add progress bar for batches
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        predictions = model(input_ids, attention_mask)
        loss = loss_fn(predictions, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Log average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}\n")


# Predict Features from Prompt
def predict_features(prompt, model, tokenizer):
    inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    with torch.no_grad():
        predictions = model(inputs["input_ids"], inputs["attention_mask"])
    return predictions.squeeze().numpy()

# Filter Songs from User Data
def filter_songs(user_data, predicted_features):
    tolerance = 0.2
    conditions = (
            (user_data["energy"] >= predicted_features[0] - tolerance) &
            (user_data["energy"] <= predicted_features[0] + tolerance) &
            (user_data["danceability"] >= predicted_features[1] - tolerance) &
            (user_data["danceability"] <= predicted_features[1] + tolerance) &
            (user_data["valence"] >= predicted_features[2] - tolerance) &
            (user_data["valence"] <= predicted_features[2] + tolerance)
    )
    filtered = user_data[conditions]

    if filtered.empty:
        print("No exact matches found. Relaxing filters...")
        tolerance += 0.1
        conditions = (
                (user_data["energy"] >= predicted_features[0] - tolerance) &
                (user_data["energy"] <= predicted_features[0] + tolerance) &
                (user_data["danceability"] >= predicted_features[1] - tolerance) &
                (user_data["danceability"] <= predicted_features[1] + tolerance)
        )
        filtered = user_data[conditions]
    return filtered

# Visualization Function
def visualize_results(prompt, predicted_features, recommended_songs):
    avg_features = recommended_songs[["energy", "danceability", "valence", "instrumentalness"]].mean()

    # Bar Chart
    plt.figure(figsize=(10, 6))
    x = ["Energy", "Danceability", "Valence", "Instrumentalness"]
    bar_width = 0.35
    index = np.arange(len(x))

    plt.bar(index, predicted_features, bar_width, label="Predicted Features", alpha=0.7)
    plt.bar(index + bar_width, avg_features, bar_width, label="Average Song Features", alpha=0.7)

    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.title(f"Feature Comparison for Prompt: '{prompt}'", fontsize=14)
    plt.xticks(index + bar_width / 2, x)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Feature Distributions
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(["energy", "danceability", "valence", "instrumentalness"]):
        plt.subplot(2, 2, i + 1)
        sns.histplot(recommended_songs[feature], kde=True, bins=10, color="skyblue", edgecolor="black")
        plt.title(f"Distribution of {feature.capitalize()}", fontsize=12)
        plt.xlabel(feature.capitalize(), fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    plt.show()

# Generate Playlist Function with Visualization
def generate_playlist(prompt, model, user_data, tokenizer):
    predicted_features = predict_features(prompt, model, tokenizer)
    print(f"Predicted Features: {predicted_features}")

    filtered_songs = filter_songs(user_data, predicted_features)
    if filtered_songs.empty:
        print("No songs match the predicted features.")
        return None

    filtered_songs = filtered_songs.head(20)
    display_columns = ["name", "artists", "energy", "danceability", "valence", "instrumentalness"]
    filtered_songs = filtered_songs[display_columns]

    visualize_results(prompt, predicted_features, filtered_songs)

    return filtered_songs

# Prompt Input and Playlist Generation
user_prompt = input("Enter your music prompt: ")
playlist = generate_playlist(user_prompt, model, user_data, tokenizer)

if playlist is not None:
    print("\nFinal Playlist:")
    print(playlist)
