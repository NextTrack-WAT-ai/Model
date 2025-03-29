def validate_playlist(playlist):
    if not isinstance(playlist, list):
        raise ValueError("Playlist must be a list.")
    
    for song in playlist:
        if not isinstance(song, dict) or 'name' not in song or 'artist' not in song:
            raise ValueError("Each song must be a dictionary with 'name' and 'artist' keys.")
    
    return True

def format_response(shuffled_playlist):
    return {
        "shuffled_playlist": shuffled_playlist
    }