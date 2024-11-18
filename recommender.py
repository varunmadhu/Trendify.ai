from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the user-specific model
user_model = load_model('user_model_with_country.keras', custom_objects={'mse': 'mean_squared_error'})

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
song_database = pd.read_csv('data/kaggle_dataset_filtered_new.csv')

# Define a function to find unique similar songs for each profile, avoiding duplicates across all profiles
def find_similar_songs_with_features(profile, song_database, top_n=5, global_seen_songs=None):
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
            # Add the full row (or selected features) to the recommendations
            unique_recommendations.append(row)
            global_seen_songs.add(song_name)

        # Stop if we've collected enough unique recommendations
        if len(unique_recommendations) >= top_n:
            break

    return unique_recommendations

# Step 3: Generate recommendations for each profile and combine them
all_recommendations = []
global_seen_songs = set()  # Track unique songs across all clusters

for profile in preference_profiles:
    recommendations = find_similar_songs_with_features(profile, song_database, top_n=10, global_seen_songs=global_seen_songs)
    all_recommendations.extend(recommendations)

# Convert recommendations to a DataFrame for better display
recommendations_df = pd.DataFrame(all_recommendations)

# Display the recommended playlist with features
print("Generated Personalized Playlist with Unique Songs Across All Listening Moods:")
for i, (_, song) in enumerate(recommendations_df.iterrows(), 1):
    print(f"{i}. {song['name']} - Features: {song[numeric_columns].to_dict()}")


from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Define columns for different scalers
minmax_columns = ['energy', 'danceability', 'valence', 'instrumentalness']
standard_columns = ['loudness', 'duration_ms', 'tempo']

# Scale recommended songs' features
def scale_features(dataframe):
    # MinMax scaling
    minmax_scaler = MinMaxScaler()
    dataframe[minmax_columns] = minmax_scaler.fit_transform(dataframe[minmax_columns])

    # Standard scaling
    standard_scaler = StandardScaler()
    dataframe[standard_columns] = standard_scaler.fit_transform(dataframe[standard_columns])

    return dataframe

# Prepare DataFrame of recommended songs
scaled_recommendations_df = recommendations_df.copy()  # Copy to avoid modifying original
scaled_recommendations_df = scale_features(scaled_recommendations_df)

# Scale user songs for comparison
scaled_user_data = user_data.copy()  # Copy to avoid modifying original
scaled_user_data = scale_features(scaled_user_data)

# Plot scatter comparison between recommended and user songs
def plot_features_comparison(user_df, recommended_df, feature):
    plt.figure(figsize=(10, 6))
    plt.scatter(user_df[feature], range(len(user_df)), label='User Songs', color='blue', alpha=0.6)
    plt.scatter(recommended_df[feature], range(len(recommended_df)), label='Recommended Songs', color='orange', alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel("Songs (Index)")
    plt.title(f"Comparison of {feature} Between User Songs and Recommendations")
    plt.legend()
    plt.grid()
    plt.show()

# Compare features (loop over all features)
for feature in minmax_columns + standard_columns:
    plot_features_comparison(scaled_user_data, scaled_recommendations_df, feature)