import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

user_data_path = 'data/Varun-Filtered.csv'  
user_data = pd.read_csv(user_data_path)

kaggle_data_path = 'data/kaggle_dataset_filtered_new.csv' 
kaggle_data = pd.read_csv(kaggle_data_path)

song_names = kaggle_data['name']
artists = kaggle_data['artists']

user_data = user_data.drop(columns=['name'], errors='ignore')
kaggle_data = kaggle_data.drop(columns=['name', 'artists'], errors='ignore')

minmax_columns = ['energy', 'danceability', 'valence', 'instrumentalness', 'tempo', 'loudness', 'duration_ms']

def scale_features(dataframe):
    minmax_scaler = MinMaxScaler()
    dataframe[minmax_columns] = minmax_scaler.fit_transform(dataframe[minmax_columns])

    return dataframe

user_data = scale_features(user_data)
kaggle_data = scale_features(kaggle_data)

sequence_length = 5
sequences = []

for i in range(len(user_data) - sequence_length):
    sequences.append(user_data[minmax_columns].iloc[i:i + sequence_length].values)

sequences = np.array(sequences)

X = sequences[:, :-1] 
y = sequences[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1]))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

last_sequence = user_data[minmax_columns].iloc[-sequence_length:].values.reshape(1, sequence_length, len(minmax_columns))
predicted_features = model.predict(last_sequence)

kaggle_features = kaggle_data[minmax_columns].values
similarities = np.linalg.norm(kaggle_features - predicted_features, axis=1)
kaggle_data['similarity'] = similarities

kaggle_data_with_names = pd.read_csv(kaggle_data_path)
kaggle_data['name'] = kaggle_data_with_names['name']
kaggle_data['artists'] = kaggle_data_with_names['artists']
recommendations = kaggle_data.sort_values(by='similarity').drop_duplicates(subset='name', keep='first')

recommended_songs_df = recommendations[['name', 'artists'] + minmax_columns].drop_duplicates(subset='name', keep='first').head(25)

print("Top 10 recommended songs based on user's listening habits:")
print(recommended_songs_df)

def plot_features_comparison(user_df, recommended_df, feature):
    plt.figure(figsize=(10, 6))
    plt.scatter(user_df[feature], range(len(user_df)), label='User Songs', color='blue', alpha=0.6)
    plt.scatter(recommended_df[feature], range(len(user_df)), label='Recommended Songs', color='orange', alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel("Songs (Index)")
    plt.title(f"Comparison of {feature} Between User Songs and Recommendations")
    plt.legend()
    plt.grid()
    plt.show()
    
user_data_temp = user_data.head(len(recommended_songs_df))

for feature in minmax_columns:
    plot_features_comparison(user_data_temp, recommended_songs_df, feature)
