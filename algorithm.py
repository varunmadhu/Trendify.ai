import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Attention, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the dataset
file_path = 'data/Harshil-Data.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print("First few rows of the dataset:")
print(data.head())

print("\nDataset info:")
print(data.info())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop any unnecessary columns if identified (e.g., 'name' might be for reference and not needed for modeling)
data = data.drop(columns=['name'])

# Normalize relevant numeric columns
scaler = MinMaxScaler()
numeric_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                   'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

print("\nNormalized data sample:")
print(data.head())

# Sort data by ranking to create sequences if that represents the play order
data = data.sort_values(by='ranking').reset_index(drop=True)

# Creating sequences - Example: Sequence of 5 songs at a time
sequence_length = 5
sequences = []

# Convert DataFrame rows into sequences of defined length
for i in range(len(data) - sequence_length + 1):
    sequences.append(data.iloc[i:i + sequence_length].values)

# Convert the list of sequences to a NumPy array (common input format for RNNs)
sequences = np.array(sequences)
print("Sample of created sequences:")
print(sequences[:1])  # Show the first sequence for inspection
print(f"\nTotal number of sequences: {len(sequences)}")

# Prepare input (X) and target (y) data
X = sequences[:, :-1]  # All steps except the last in each sequence
y = sequences[:, -1]   # The last step in each sequence as the target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model with Attention
sequence_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Bidirectional(LSTM(128, return_sequences=True))(sequence_input)
x = Dropout(0.3)(x)

# Apply another LSTM layer to process the data further
x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.3)(x)

# Attention layer
attention_data = Attention()([x, x])

# Use GlobalAveragePooling1D instead of tf.reduce_mean to aggregate the attention output
x = GlobalAveragePooling1D()(attention_data)

# Fully connected layers for the final prediction
x = Dense(32, activation='relu')(x)
output = Dense(y_train.shape[1])(x)

# Define the full model
model_with_attention = Model(inputs=sequence_input, outputs=output)

# Compile the model
model_with_attention.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

# Train the model with attention
history_attention = model_with_attention.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model with attention on the test set
test_loss_attention = model_with_attention.evaluate(X_test, y_test, verbose=1)
print(f"\nImproved Test Loss with Attention: {test_loss_attention}")

# Predict with the model with attention
y_pred_attention = model_with_attention.predict(X_test)

# Visualization of predictions vs actual values
num_features_to_plot = 3  # Number of features to visualize
for i in range(num_features_to_plot):
    plt.figure(figsize=(8, 4))
    plt.plot(y_test[:, i], label="Actual", marker='o')
    plt.plot(y_pred_attention[:, i], label="Predicted", marker='x')
    plt.title(f"Feature {i+1}: Actual vs Predicted with Attention")
    plt.xlabel("Sample")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.show()
