from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd

# Load and preprocess the user’s data
user_data = pd.read_csv('data/Harshil-Filtered.csv')
numeric_columns = ['energy', 'danceability', 'valence', 'instrumentalness', 'loudness', 'duration_ms', 'tempo']
X = user_data[numeric_columns]

# Define and train the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(X.shape[1])  # Output layer to match feature space
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
model.fit(X, X, epochs=50, batch_size=16, validation_split=0.2)

# Save the trained model
model.save('user_model.keras')
