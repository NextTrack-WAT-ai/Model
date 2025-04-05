# %% [markdown]
# This is version 00 of the playlist shuffle or you can think of this as version 02 of "*nextTrack_recommender_basic_*" files. The following is true, and changes from previous version are highlighted in **bold**:
# 
# 1. **The songs are shuffled in the playlist one after another. As in, the first song is chosen (randomly or by the user), the second song is the song with the highest similarity to 1st. The third song is the one with the highest similarity to the 2nd song, and so on.**
# 2. There are no weights for the features for now. Everything matters the same amount.
# 3. User can either pick the first song or it can be random.
# 4. It does not implement any "learning".
# 5. It does not consider tags.

# %%
# %%capture
import pandas as pd
import numpy as np
import random


# for preprocess
# from sklearn.preprocessing import scale

# for accuracy measures
# from sklearn import metrics

# ML algorithms

from sklearn.metrics.pairwise import cosine_similarity

# hierarchical clustering
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# %%
# %%capture
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
def load_dataset():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("undefinenull/million-song-dataset-spotify-lastfm")

    csv_files = [f for f in os.listdir(path) if f.endswith("Info.csv")]

    # Check if any CSV file exists
    if csv_files:
        file_path = os.path.join(path, csv_files[0])

        df = pd.read_csv(file_path)

        # print(f"Reading file: {file_path}")
        # display(df.head())
    else:
        print("No CSV files found in the directory.")
    return df

# %% [markdown]
# The Task at hand is as follows:
# 
# 1. User inputs a playlist (dictionary) which has the song title (`name`) and artist's name (`artist`).
# 2. We can either randomly select a song as the first or ask the user to pick the song to start with.
# 3. Search through the database for the song - if not found, error. Else, move on to step 4.
# 4. Create a new playlist.
# 5. Add the first song's name and artist information in it.
# 6. Remove this song from user's playlist, then from the remaining songs, find the song with the highest cosine similarity with song already in the new playlist.
# 6. Repeat steps 5 and 6 until the user's playlist is empty.
# 7. Return the shuffled playlist.

# %%
user_playlist = ["06UfBBDISthj1ZJAtX4xjj",
 "09ZQ5TmUG8TSL56n0knqrj",
 "01QoK9DA7VTeTSE3MNzp4I",
 "0keNu0t0tqsWtExGM3nT1D",
 "28YZkdihqt3e37t1IcqJIu",
 "0GG7ei637NDIN2w11TWtLC",
 "08A1lZeyLMWH58DT6aYjnC",
 "005lwxGU1tms6HGELIcUv9",
 "1jHhOrH4kIhjLFvagzvw1s"]



# %%
df = load_dataset()
# def shufflePlaylist(playlist, first_song_choice):
#   # see if the songs exist in the database
#   matched_indices = []

#   for song in playlist:
#     matches = df[(df['name'] == song['name']) & (df['artist'] == song['artist'])]

#     if not matches.empty:
#       matched_indices.append(matches.index[0])

#     else:
#       print(f"Warning: Song '{song['name']}' by '{song['artist']}' not found in database.")

#     if not matched_indices:
#         raise ValueError("No songs from user's playlist were found in the database.")


#     # choose the user songs' information from the df
#     user_df = df.loc[matched_indices].copy()

#     # Select numerical features for similarity calculation ---> will add tags later
#     numerical_features = ['year','duration_ms','danceability', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

#     # sanity check
#     numerical_features = [feat for feat in numerical_features if feat in user_df.columns]

#     # Normalize numerical features
#     features_df = user_df[numerical_features].copy()
#     for feature in numerical_features:
#         if features_df[feature].std() > 0:  # Avoid division by zero
#             features_df[feature] = (features_df[feature] - features_df[feature].mean()) / features_df[feature].std()

#     if first_song_choice:
#         first_song_matches = user_df[
#             (user_df['name'] == first_song_choice['name']) &
#             (user_df['artist'] == first_song_choice['artist'])
#         ]
#         if first_song_matches.empty:
#             raise ValueError("The selected first song is not found in the user's playlist.")
#         current_song_idx = first_song_matches.index[0]
#     else:
#         current_song_idx = random.choice(user_df.index)  # <<< this is now the true index

#     # Step 5: Initialize
#     shuffled_playlist = []
#     remaining_indices = list(user_df.index)

#     # Step 6: Nearest Neighbor Loop
#     while remaining_indices:
#         # Add current song
#         song_info = {
#             'name': user_df.loc[current_song_idx, 'name'],
#             'artist': user_df.loc[current_song_idx, 'artist']
#         }
#         shuffled_playlist.append(song_info)

#         # Remove it
#         remaining_indices.remove(current_song_idx)
#         if not remaining_indices:
#             break

#         # Compute similarity
#         current_features = features_df.loc[current_song_idx].values.reshape(1, -1)
#         remaining_features = features_df.loc[remaining_indices].values

#         similarities = cosine_similarity(current_features, remaining_features)[0]

#         # Find next song
#         next_index_in_remaining = np.argmax(similarities)
#         current_song_idx = remaining_indices[next_index_in_remaining]

#     return shuffled_playlist


# %%
feature_weights = {
        'key': 3.0,           # Key is 3x more important
        'tempo': 2.0,         # Tempo is 2x more important
        'speechiness': 0.5,   # Speechiness is half as important
        'danceability': 2.5   # Danceability is 2.5x more important
    }


# shufflePlaylist(user_playlist, None)

# %%
# Claude fix

from sklearn.preprocessing import StandardScaler

def shufflePlaylist(df, user_playlist):
    """
    Shuffle a playlist based on song similarities in the database.

    Parameters:
    -----------
    df : pandas.DataFrame
        The database of songs containing features for similarity calculation
    user_playlist : list
        List of spotify ids
    first_song_choice : dict, optional
        Specific song to start the playlist with

    Returns:
    --------
    list
        Shuffled playlist with song details
    """
    # Step 1: Validate and match user playlist songs with database
    matched_indices = []
    song_name_mapping = dict(zip(df['spotify_id'], df['name']))
    for song in user_playlist:
        matches = df[
            df['spotify_id'] == song
        ]

        if matches.empty:
            print(f"Warning: Song '{song_name_mapping[song]}' not found in database.")
        else:
            matched_indices.append(matches.index[0])
    # Raise error if no songs found
    if not matched_indices:
        raise ValueError("No songs from user's playlist were found in the database.")

    # Step 2: Create subset of matched songs
    user_df = df.loc[matched_indices].copy()

    # Step 3: Select and normalize numerical features
    numerical_features = [
        'year', 'duration_ms', 'danceability', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]

    # Sanity check for features
    numerical_features = [
        feat for feat in numerical_features if feat in user_df.columns
    ]

    # Normalize features
    features_df = user_df[numerical_features].copy()
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df)
    features_df_normalized = pd.DataFrame(
        features_normalized,
        columns=numerical_features,
        index=features_df.index
    )

    # Step 4: Select first song
    current_song_idx = random.choice(user_df.index)

    # Initialize shuffled playlist and remaining songs
    shuffled_playlist = []
    remaining_indices = list(user_df.index)

    # Shuffle playlist
    while remaining_indices:
        # Add current song to shuffled playlist
        song_info = {
            'spotify_id': user_df.loc[current_song_idx, 'spotify_id'],
        }
        shuffled_playlist.append(song_info)

        # Remove current song from remaining indices
        remaining_indices.remove(current_song_idx)

        # Exit if no more songs
        if not remaining_indices:
            break

        # Compute cosine similarities
        current_features = features_df_normalized.loc[current_song_idx].values.reshape(1, -1)
        remaining_features = features_df_normalized.loc[remaining_indices].values

        # Calculate similarities
        similarities = cosine_similarity(current_features, remaining_features)[0]

        # Find next most similar song
        next_index_in_remaining = np.argmax(similarities)
        current_song_idx = remaining_indices[next_index_in_remaining]

    return shuffled_playlist

# Example usage function
def demonstrate_playlist_shuffler(df, user_playlist):
    try:
        # Option 1: Randomly select first song
        shuffled_playlist_random = shufflePlaylist(df, user_playlist)
        song_names = dict(zip(df['spotify_id'], df['name']))
        print("Randomly Selected First Song Playlist:")
        for i, song in enumerate(shuffled_playlist_random, 1):
            print(f"{i}. {song_names[song['spotify_id']]}, Spotify ID: {song['spotify_id']}")

        print("\n")

        # Option 2: Specify first song
        # first_song_choice = user_playlist[0]  # Example: use first song from playlist
        # shuffled_playlist_specific = shufflePlaylist(
        #     df,
        #     user_playlist,
        #     first_song_choice
        # )
        # print(f"Playlist Starting with {song_names[first_song_choice]}:")
        # for i, song in enumerate(shuffled_playlist_specific, 1):
        #     print(f"{i}. {song_names[song['spotify_id']]}")

    except Exception as e:
        print(f"An error occurred: {e}")

def translateSpotifyID(df, playlist):
    """
    Translate Spotify IDs in the shuffled playlist to song names.

    Parameters:
    -----------
    df : pandas.DataFrame
        The database of songs containing features for similarity calculation
    shuffled_playlist : list
        Shuffled playlist with Spotify IDs

    Returns:
    --------
    list
        Shuffled playlist with song names
    """
    song_names = dict(zip(df['spotify_id'], df['name']))
    song_artists = dict(zip(df['spotify_id'], df['artist']))
    if (isinstance(playlist[0], dict)):
        for i, song in enumerate(playlist, 1):
            print(f"{i}. {song_names[song['spotify_id']]} by {song_artists[song['spotify_id']]}")
    else:
        for i, song in enumerate(playlist, 1):
            print(f"{i}. {song_names[song]} by {song_artists[song]}")

# %%
# demonstrate_playlist_shuffler(df, user_playlist)

# %%


# %% [markdown]
# 

# %%
def calculate_similarity_metrics(shuffled_playlist, original_df):
    """
    Calculate various metrics to evaluate playlist shuffling quality.

    Parameters:
    -----------
    shuffled_playlist : list
        Shuffled playlist of songs
    original_df : pandas.DataFrame
        Original dataset containing song features

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Extract numerical features
    numerical_features = [
        'year', 'duration_ms', 'danceability', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]

    # Ensure features exist in the dataframe
    numerical_features = [
        feat for feat in numerical_features if feat in original_df.columns
    ]

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(original_df[numerical_features])

    # Create metrics tracking
    metrics = {
        'feature_distances': [],
        'consecutive_similarities': []
    }

    # Calculate feature distances between consecutive songs
    for i in range(len(shuffled_playlist) - 1):
        # Find indices of current and next songs
        current_song = original_df[
            original_df['spotify_id'] == shuffled_playlist[i]['spotify_id']
        ].index[0]

        next_song = original_df[
            original_df['spotify_id'] == shuffled_playlist[i+1]['spotify_id']
        ].index[0]

        # Calculate cosine similarity between consecutive songs
        current_features = features_normalized[current_song].reshape(1, -1)
        next_features = features_normalized[next_song].reshape(1, -1)

        similarity = cosine_similarity(current_features, next_features)[0][0]
        metrics['consecutive_similarities'].append(similarity)

        # Calculate Euclidean distance between features
        distance = np.linalg.norm(
            features_normalized[current_song] - features_normalized[next_song]
        )
        metrics['feature_distances'].append(distance)

    # Calculate additional metrics
    metrics['avg_consecutive_similarity'] = np.mean(metrics['consecutive_similarities'])
    metrics['std_consecutive_similarity'] = np.std(metrics['consecutive_similarities'])
    metrics['avg_feature_distance'] = np.mean(metrics['feature_distances'])
    metrics['std_feature_distance'] = np.std(metrics['feature_distances'])

    return metrics

# def visualize_playlist_evaluation(metrics):
    """
    Create visualizations for playlist shuffling evaluation.

    Parameters:
    -----------
    metrics : dict
        Metrics dictionary from calculate_similarity_metrics()
    """
    plt.figure(figsize=(12, 5))

    # Consecutive Similarities Plot
    plt.subplot(1, 2, 1)
    plt.plot(metrics['consecutive_similarities'], marker='o')
    plt.title('Consecutive Song Similarities')
    plt.xlabel('Song Transition')
    plt.ylabel('Cosine Similarity')
    plt.axhline(y=metrics['avg_consecutive_similarity'], color='r', linestyle='--',
                label=f'Mean: {metrics["avg_consecutive_similarity"]:.4f}')
    plt.legend()

    # Feature Distances Plot
    plt.subplot(1, 2, 2)
    plt.plot(metrics['feature_distances'], marker='o')
    plt.title('Feature Distances Between Consecutive Songs')
    plt.xlabel('Song Transition')
    plt.ylabel('Euclidean Distance')
    plt.axhline(y=metrics['avg_feature_distance'], color='r', linestyle='--',
                label=f'Mean: {metrics["avg_feature_distance"]:.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()

def run_multiple_shuffles(df, user_playlist, num_shuffles=100):
    """
    Run multiple shuffles to analyze consistency and variability.

    Parameters:
    -----------
    df : pandas.DataFrame
        Song database
    user_playlist : list
        Original playlist
    num_shuffles : int
        Number of shuffles to perform

    Returns:
    --------
    list
        List of metrics from each shuffle
    """
    all_metrics = []

    for _ in range(num_shuffles):
        # Shuffle the playlist
        shuffled_playlist = shufflePlaylist(df, user_playlist)

        # Calculate metrics
        metrics = calculate_similarity_metrics(shuffled_playlist, df)
        all_metrics.append(metrics)

    return all_metrics

# def comprehensive_evaluation(df, user_playlist):
    """
    Perform comprehensive evaluation of playlist shuffler.

    Parameters:
    -----------
    df : pandas.DataFrame
        Song database
    user_playlist : list
        Original playlist
    """
    # Run multiple shuffles
    all_metrics = run_multiple_shuffles(df, user_playlist)

    # Aggregate metrics
    aggregated_metrics = {
        'avg_consecutive_similarities': [m['avg_consecutive_similarity'] for m in all_metrics],
        'std_consecutive_similarities': [m['std_consecutive_similarity'] for m in all_metrics],
        'avg_feature_distances': [m['avg_feature_distance'] for m in all_metrics],
        'std_feature_distances': [m['std_feature_distance'] for m in all_metrics]
    }

    # Print summary statistics
    print("Shuffle Evaluation Summary:")
    print(f"Average Consecutive Similarity: {np.mean(aggregated_metrics['avg_consecutive_similarities']):.4f} ± {np.std(aggregated_metrics['avg_consecutive_similarities']):.4f}")
    print(f"Average Feature Distance: {np.mean(aggregated_metrics['avg_feature_distances']):.4f} ± {np.std(aggregated_metrics['avg_feature_distances']):.4f}")

    # Visualize distribution of metrics
    plt.figure(figsize=(12, 5))

    # Consecutive Similarities Distribution
    plt.subplot(1, 2, 1)
    plt.hist(aggregated_metrics['avg_consecutive_similarities'], bins=20)
    plt.title('Distribution of Average Consecutive Similarities')
    plt.xlabel('Average Similarity')
    plt.ylabel('Frequency')

    # Feature Distances Distribution
    plt.subplot(1, 2, 2)
    plt.hist(aggregated_metrics['avg_feature_distances'], bins=20)
    plt.title('Distribution of Average Feature Distances')
    plt.xlabel('Average Distance')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    return aggregated_metrics


# Perform a single shuffle evaluation
shuffled_playlist = shufflePlaylist(df, user_playlist)
metrics = calculate_similarity_metrics(shuffled_playlist, df)
# visualize_playlist_evaluation(metrics)

# Perform comprehensive evaluation
# comprehensive_metrics = comprehensive_evaluation(df, user_playlist)

# %%



