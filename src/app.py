from dotenv import load_dotenv
load_dotenv()
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify
from reshuffler import shufflePlaylist, load_dataset, learn_from_reordered_playlist
from utils import display_playlist
import pandas as pd
from pymongo import MongoClient

app = Flask(__name__)
# df = load_dataset()
MONGO_URI = os.environ.get('MONGO_DB_URI')

@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200

@app.route('/shuffle', methods=['POST'])
def shuffle():
    data = request.json
    user_playlist = data.get('playlist')
    user_email = data.get('email')
    try:
        client = MongoClient(MONGO_URI)
        db = client.next_track
        user = db.users.find_one({"email": user_email})
        feature_weights = None
        numerical_features = [
            'year', 'duration_ms', 'danceability', 'key', 'loudness',
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature'
        ]
        if user:
            feature_weights = user["songFeaturePreferences"]
        else:
            feature_weights = {feature: 1.0 for feature in numerical_features}
        
        shuffled_playlist = shufflePlaylist(user_playlist, feature_weights=feature_weights)
        print("User Playlist:")
        display_playlist(user_playlist)
        print("Shuffled Playlist:")
        display_playlist(shuffled_playlist)
        return jsonify(shuffled_playlist), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred."}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_reordered_playlist = data.get('playlist')
    user_email = data.get('email')
    try:
        original_order_playlist = sorted(user_reordered_playlist, key=lambda x: x['trackIndex'])
        # Retrieve user from database
        client = MongoClient(MONGO_URI)
        db = client.next_track
        user = db.users.find_one({"email": user_email})
        feature_weights = None
        numerical_features = [
            'year', 'duration_ms', 'danceability', 'key', 'loudness',
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature'
        ]
        if user:
            feature_weights = user["songFeaturePreferences"]
        else:
            feature_weights = {feature: 1.0 for feature in numerical_features}
        # Learn from the user's reordered playlist
        updated_weights = learn_from_reordered_playlist(original_order_playlist, user_reordered_playlist, feature_weights)
        db.users.update_one({"email": user_email}, {"$set": {"songFeaturePreferences": updated_weights}})
        return jsonify({"updated_weights": updated_weights}), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)