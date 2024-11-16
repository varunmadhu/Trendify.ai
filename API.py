import musicbrainzngs
import pandas as pd
from collections import Counter
import numpy as np

# Set up MusicBrainz API client
musicbrainzngs.set_useragent("MusicCountryFinder", "1.0", "your-email@example.com")

# Function to get country of an artist using MusicBrainz API
def get_country_by_artist(artist_name):
    """Fetch the country of an artist using MusicBrainz."""
    try:
        # Search for the artist
        result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
        if result['artist-list']:
            artist = result['artist-list'][0]
            if 'area' in artist:
                return artist['area']['name']  # Return country name
        return 'Unknown'
    except Exception as e:
        print(f"Error fetching country for {artist_name}: {e}")
        return 'Unknown'

# Function to calculate country weights dynamically
def calculate_country_weights(user_data):
    """Calculate the weight of each country dynamically based on the user's listening history."""
    # Fetch country for each artist in the user's dataset
    print("Fetching countries for user's listening history...")
    user_data['country'] = user_data['artists'].apply(get_country_by_artist)

    # Count occurrences of each country
    country_counts = Counter(user_data['country'])
    total_songs = sum(country_counts.values())

    # Calculate weights as a percentage
    country_weights = {country: count / total_songs for country, count in country_counts.items()}
    return country_weights

# Function to find similar songs based on country weights
def find_similar_songs_by_country(profile, song_database, country_weights, top_n=10):
    """Find similar songs, prioritizing countries by weight."""
    # Calculate distance between user profile and song features
    song_database['distance'] = np.linalg.norm(
        song_database[['energy', 'danceability', 'valence', 'instrumentalness']].values - profile,
        axis=1
    )

    # Apply country weights to adjust distances
    def weighted_distance(row):
        country_weight = country_weights.get(row['country'], 0.001)  # Default small weight for unknown countries
        return row['distance'] / country_weight

    song_database['weighted_distance'] = song_database.apply(weighted_distance, axis=1)

    # Sort by weighted distance and keep only unique songs
    recommendations = song_database.sort_values('weighted_distance').drop_duplicates(subset=['name']).head(top_n)

    return recommendations

# Load user dataset
print("Loading user data...")
user_data = pd.read_csv('data/Harshil-Filtered.csv')

# Calculate country weights
print("Calculating country weights...")
country_weights = calculate_country_weights(user_data)
print("Calculated Country Weights:", country_weights)

# Load the Kaggle dataset with country data
print("Loading song database...")
song_database = pd.read_csv('data/kaggle_dataset_filtered_new.csv')

# Generate a preference profile from the user's data
print("Generating user profile...")
user_profile = user_data[['energy', 'danceability', 'valence', 'instrumentalness']].mean().values

# Generate song recommendations
print("Generating recommendations...")
recommended_songs = find_similar_songs_by_country(user_profile, song_database, country_weights)
print("Recommended Songs:\n", recommended_songs[['name', 'artists', 'country']])
