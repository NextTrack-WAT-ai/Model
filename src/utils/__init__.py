def validate_playlist(playlist):
    if not isinstance(playlist, list):
        raise ValueError("Playlist must be a list.")
    
    return True

def display_playlist(playlist):
    for i, song in enumerate(playlist, start=1):
        print(f"{i}. {song['name']} by {song['artist']}")