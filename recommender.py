from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the user-specific model
user_model = load_model('user_model.keras', custom_objects={'mse': 'mean_squared_error'})

# Load the user's data
user_data = pd.read_csv('data/Harshil-Filtered.csv')
numeric_columns = ['energy', 'danceability', 'valence', 'instrumentalness', 'loudness', 'duration_ms', 'tempo']
X = user_data[numeric_columns]

# Step 1: Cluster the user's data to find different listening "moods"
n_clusters = 3  # Adjust this based on the diversity in user listening habits
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Step 2: Generate preference profiles from cluster centroids
preference_profiles = kmeans.cluster_centers_

# Load the larger song dataset for recommendations
song_database = pd.read_csv('data/kaggle_dataset_filtered.csv')

# Define a function to find unique similar songs for each profile, avoiding duplicates across all profiles
def find_similar_songs(profile, song_database, top_n=5, global_seen_songs=None):
    if global_seen_songs is None:
        global_seen_songs = set()

    # Calculate the distance between the profile and each song in the dataset
    song_database['distance'] = np.linalg.norm(song_database[numeric_columns].values - profile, axis=1)
    close_matches = song_database.sort_values(by='distance')
    unique_recommendations = []

    # Loop through the sorted close matches and add unique songs to recommendations
    for _, row in close_matches.iterrows():
        song_name = row['name']
        if song_name not in global_seen_songs:
            unique_recommendations.append(song_name)
            global_seen_songs.add(song_name)

        # Stop if we've collected enough unique recommendations
        if len(unique_recommendations) >= top_n:
            break

    return unique_recommendations

# Step 3: Generate recommendations for each profile and combine them
all_recommendations = []
global_seen_songs = set()  # Track unique songs across all clusters

for profile in preference_profiles:
    recommendations = find_similar_songs(profile, song_database, top_n=5, global_seen_songs=global_seen_songs)
    all_recommendations.extend(recommendations)

# Display the recommended playlist with unique songs from each profile
print("Generated Personalized Playlist with Unique Songs Across All Listening Moods:")
for i, song_name in enumerate(all_recommendations, 1):
    print(f"{i}. {song_name}")
