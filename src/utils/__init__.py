def validate_playlist(playlist):
    if not isinstance(playlist, list):
        raise ValueError("Playlist must be a list.")
    
    return True

def format_response(shuffled_playlist):
    return {
        "shuffled_playlist": shuffled_playlist
    }