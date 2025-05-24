from typing import List

def display_playlist(playlist):
    for i, song in enumerate(playlist, start=1):
        print(f"{i}. {song['name']} by {song['artists']}")