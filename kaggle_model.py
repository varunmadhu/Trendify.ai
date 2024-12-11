import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Attention, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

file_path = 'data/kaggle_dataset_filtered.csv'  
data = pd.read_csv(file_path)

print("Dataset sample:")
print(data.head())

aggregated_data = data.groupby('name').agg({
    'energy': 'mean',
    'danceability': 'mean',
    'valence': 'mean',
    'instrumentalness': 'mean',
    'loudness': 'mean',
    'duration_ms': 'mean',
    'tempo': 'mean',
    'pop_trend_index': 'sum'  
}).reset_index()

aggregated_data = aggregated_data.rename(columns={'pop_trend_index': 'weight'})

numeric_columns = ['energy', 'danceability', 'valence', 'instrumentalness', 'loudness', 'duration_ms', 'tempo']
scaler = MinMaxScaler()
aggregated_data[numeric_columns] = scaler.fit_transform(aggregated_data[numeric_columns])

weighted_rows = []
for _, row in aggregated_data.iterrows():
    weighted_rows.extend([row] * int(row['weight']))  

weighted_df = pd.DataFrame(weighted_rows)
print("Weighted data sample:")
print(weighted_df.head())

sequence_length = 5
sequences = []

weighted_df = weighted_df.sample(frac=1).reset_index(drop=True)

for i in range(len(weighted_df) - sequence_length + 1):
    sequences.append(weighted_df.iloc[i:i + sequence_length][numeric_columns].values)

sequences = np.array(sequences)
print("Total number of sequences:", len(sequences))

X = sequences[:, :-1]  
y = sequences[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sequence_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Bidirectional(LSTM(128, return_sequences=True))(sequence_input)
x = Dropout(0.3)(x)

x = LSTM(64, return_sequences=True)(x)
x = Dropout(0.3)(x)

attention_data = Attention()([x, x])
x = GlobalAveragePooling1D()(attention_data) 

x = Dense(32, activation='relu')(x)
output = Dense(y_train.shape[1])(x)

model = Model(inputs=sequence_input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)

test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {test_loss}")

