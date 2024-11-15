import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Attention, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the dataset
file_path = 'data/kaggle_dataset_filtered.csv'  # Replace with your actual dataset path
data = pd.read_csv(file_path)

print("Dataset sample:")
print(data.head())

# Step 1: Handle Duplicates and Aggregate by Song Name
# Calculate the total weight for each song by summing 'pop_trend_index'
# and average other audio features to represent each unique song
aggregated_data = data.groupby('name').agg({
    'energy': 'mean',
    'danceability': 'mean',
    'valence': 'mean',
    'instrumentalness': 'mean',
    'loudness': 'mean',
    'duration_ms': 'mean',
    'tempo': 'mean',
    'pop_trend_index': 'sum'  # Summing popularity to give weight to repeated songs
}).reset_index()

# Rename 'pop_trend_index' to 'weight' to indicate song importance
aggregated_data = aggregated_data.rename(columns={'pop_trend_index': 'weight'})

# Step 2: Normalize Audio Features
numeric_columns = ['energy', 'danceability', 'valence', 'instrumentalness', 'loudness', 'duration_ms', 'tempo']
scaler = MinMaxScaler()
aggregated_data[numeric_columns] = scaler.fit_transform(aggregated_data[numeric_columns])

# Step 3: Prepare Weighted Data for Training
# Expand rows based on weights to simulate the song's popularity influence
weighted_rows = []
for _, row in aggregated_data.iterrows():
    weighted_rows.extend([row] * int(row['weight']))  # Replicate each song by its weight

# Convert weighted_rows to DataFrame
weighted_df = pd.DataFrame(weighted_rows)
print("Weighted data sample:")
print(weighted_df.head())

# Step 4: Create Sequences for Model Training
sequence_length = 5
sequences = []

# Shuffle the weighted data to randomize sequences
weighted_df = weighted_df.sample(frac=1).reset_index(drop=True)

# Generate sequences
for i in range(len(weighted_df) - sequence_length + 1):
    sequences.append(weighted_df.iloc[i:i + sequence_length][numeric_columns].values)

# Convert the list of sequences to a NumPy array
sequences = np.array(sequences)
print("Total number of sequences:", len(sequences))

# Prepare input (X) and target (y) data
X = sequences[:, :-1]  # All steps except the last in each sequence
y = sequences[:, -1]   # The last step in each sequence as the target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define the RNN Model with Attention
sequence_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Bidirectional(LSTM(128, return_sequences=True))(sequence_input)
x = Dropout(0.3)(x)

# Apply another LSTM layer to process the data further
x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.3)(x)

# Attention layer and aggregation using GlobalAveragePooling1D
attention_data = Attention()([x, x])
x = GlobalAveragePooling1D()(attention_data)  # Aggregating the attention output

# Fully connected layers for the final prediction
x = Dense(32, activation='relu')(x)
output = Dense(y_train.shape[1])(x)

# Define the full model
model = Model(inputs=sequence_input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# Step 6: Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {test_loss}")

# You can now use the model to generate playlist recommendations based on user preferences.
