import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

user_data = pd.read_csv('data/Harshil-Filtered.csv')

numeric_columns = ['energy', 'danceability', 'valence', 'instrumentalness', 'loudness', 'duration_ms', 'tempo']
categorical_column = 'country'

label_encoder = LabelEncoder()
user_data['country_encoded'] = label_encoder.fit_transform(user_data[categorical_column])

scaler = MinMaxScaler()
user_data[numeric_columns] = scaler.fit_transform(user_data[numeric_columns])

X = pd.concat([user_data[numeric_columns], pd.get_dummies(user_data['country_encoded'])], axis=1)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(X.shape[1])  
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

model.fit(X, X, epochs=50, batch_size=16, validation_split=0.2)

model.save('user_model_with_country.keras')

