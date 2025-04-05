import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Flask, request, jsonify
from reshuffler import shufflePlaylist, load_dataset
from utils import validate_playlist, display_playlist
import pandas as pd

app = Flask(__name__)

@app.route('/shuffle', methods=['POST'])
def shuffle():
    user_playlist = request.json
    validate_playlist(user_playlist)
    try:
        df = load_dataset()
        shuffled_playlist = shufflePlaylist(df, user_playlist)
        print("User Playlist:")
        display_playlist(user_playlist)
        print("Shuffled Playlist:")
        display_playlist(shuffled_playlist)
        return jsonify(shuffled_playlist), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)