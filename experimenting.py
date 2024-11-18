import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import pandas as pd

# Load dataset
user_data_path = 'data/Karthik-Filtered.csv'
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

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

for epoch in range(3):
    for batch in dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        predictions = model(input_ids, attention_mask)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

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

# Generate Playlist Function
def generate_playlist(prompt, model, user_data, tokenizer):
    """
    Generate a playlist based on the input prompt.
    """
    # Predict features based on the prompt
    predicted_features = predict_features(prompt, model, tokenizer)
    print(f"Predicted Features: {predicted_features}")

    # Filter the songs based on predicted features
    filtered_songs = filter_songs(user_data, predicted_features)

    # Check if songs were found
    if filtered_songs.empty:
        print("No songs match the predicted features.")
        return None

    # Add song features to the output
    print("\nRecommended Songs with Features:")
    print(filtered_songs[["name", "artists", "energy", "danceability", "valence", "instrumentalness"]])

    return filtered_songs[["name", "artists", "energy", "danceability", "valence", "instrumentalness"]]

# Prompt Input and Playlist Generation
user_prompt = input("Enter your music prompt: ")
playlist = generate_playlist(user_prompt, model, user_data, tokenizer)

# Display the Playlist
if playlist is not None:
    print("\nGenerated Playlist:")
    print(playlist)

