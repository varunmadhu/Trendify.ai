import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('user_model_with_country.keras')

kaggle_dataset = pd.read_csv('data/kaggle_dataset_filtered_new.csv')

user_data = pd.read_csv('data/Harshil-Filtered.csv')

def filter_numeric_columns(data):
    """Select only numeric columns from the dataset."""
    return data.select_dtypes(include=[np.number])

def encode_country(data, encoder=None):
    """Encode the 'country' column using a LabelEncoder, handling unseen labels."""
    if encoder is None:
        encoder = LabelEncoder()
        data['country_encoded'] = encoder.fit_transform(data['country'].astype(str))
    else:
        unseen_label = len(encoder.classes_)
        encoder_classes = list(encoder.classes_)
        for country in data['country'].unique():
            if country not in encoder_classes:
                encoder_classes.append(country)
        encoder.classes_ = np.array(encoder_classes)
        data['country_encoded'] = data['country'].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else unseen_label
        )
    return data, encoder

def preprocess_user_data(user_data):
    """Preprocess user data: align columns and filter numeric features."""
    numeric_data = filter_numeric_columns(user_data)
    numeric_data = numeric_data.dropna()
    return numeric_data

def generate_user_profile(user_data):
    """Generate the user profile using the trained model."""
    return model.predict(user_data).mean(axis=0)

def calculate_country_weights(user_data):
    """Calculate weights for each country based on user listening history."""
    country_counts = user_data['country'].value_counts(normalize=True).to_dict()
    return country_counts

def recommend_songs(user_profile, kaggle_dataset, country_weights, top_n=10):
    """Generate song recommendations based on user profile."""
    kaggle_numeric_data = filter_numeric_columns(kaggle_dataset)
    kaggle_numeric_data = kaggle_numeric_data.dropna() 
    X_kaggle = kaggle_numeric_data.values

    kaggle_dataset['country_weight'] = kaggle_dataset['country'].map(country_weights).fillna(0.001)

    distances = np.linalg.norm(X_kaggle - user_profile, axis=1)
    kaggle_dataset['distance'] = distances

    kaggle_dataset['weighted_distance'] = kaggle_dataset['distance'] / kaggle_dataset['country_weight']

    recommendations = kaggle_dataset.sort_values('weighted_distance').drop_duplicates(subset=['name']).head(top_n)
    return recommendations

user_data, country_encoder = encode_country(user_data)
kaggle_dataset, _ = encode_country(kaggle_dataset, country_encoder)

aligned_user_data = preprocess_user_data(user_data)

user_profile = generate_user_profile(aligned_user_data)

country_weights = calculate_country_weights(user_data)

recommendations = recommend_songs(user_profile, kaggle_dataset, country_weights)

print("Recommended Songs:")
print(recommendations[['name', 'artists', 'country', 'weighted_distance']])
