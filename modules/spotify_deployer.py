import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Use Client Credentials flow instead of OAuth (no user login required)
client_credentials_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def search_songs(query, limit=20):
    """
    Search for songs on Spotify and return formatted results
    Args:
        query (str): Search query (song name, artist, etc.)
        limit (int): Number of results to return
    
    Returns:
        list: List of song dictionaries with id, name, artist, album, url
    """
    try:
        results = spotify.search(q=query, type='track', limit=limit)
        
        if results['tracks']['total'] > 0:
            tracks = results['tracks']['items']
            
            formatted_tracks = []
            for track in tracks:
                track_data = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url'],  # 30-second preview
                    'cover_url': track['album']['images'][0]['url'] or 'https://via.placeholder.com/300x300?text=No+Image',
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity']
                }
                formatted_tracks.append(track_data)
            
            return formatted_tracks
        else:
            return []
            
    except Exception as e:
        print(f"Error searching: {str(e)}")
        return []

def get_track_by_id(track_id):
    """
    Get detailed track information by Spotify track ID
    Args:
        track_id (str): Spotify track ID
    
    Returns:
        dict: Track information or None if not found
    """
    try:
        track = spotify.track(track_id)
        
        return {
            'id': track['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'url': track['external_urls']['spotify'],
            'preview_url': track['preview_url'],
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity']
        }
        
    except Exception as e:
        print(f"Error getting track: {str(e)}")
        return None

def create_urls_file(selected_songs, filename="urls.txt"):
    """
    Create URLs file from selected songs for music_downloader.py
    Args:
        selected_songs (list): List of song dictionaries with 'url' key
        filename (str): Output filename
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for song in selected_songs:
                if isinstance(song, dict) and 'url' in song:
                    file.write(song['url'] + '\n')
                elif isinstance(song, str):
                    file.write(song + '\n')
        
        print(f"Successfully saved {len(selected_songs)} URLs to {filename}")
        return True
        
    except Exception as e:
        print(f"Error saving URLs file: {str(e)}")
        return False

def get_multiple_tracks(track_ids):
    """
    Get multiple tracks by their IDs
    Args:
        track_ids (list): List of Spotify track IDs
    
    Returns:
        list: List of track dictionaries
    """
    try:
        tracks = spotify.tracks(track_ids)
        
        formatted_tracks = []
        for track in tracks['tracks']:
            if track:  # Check if track exists
                track_data = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity']
                }
                formatted_tracks.append(track_data)
        
        return formatted_tracks
        
    except Exception as e:
        print(f"Error getting multiple tracks: {str(e)}")
        return []

def validate_spotify_url(url):
    """
    Validate if a URL is a valid Spotify track URL
    Args:
        url (str): URL to validate
    
    Returns:
        bool: True if valid Spotify track URL
    """
    return url.startswith('https://open.spotify.com/track/')

def extract_track_id_from_url(url):
    """
    Extract track ID from Spotify URL
    Args:
        url (str): Spotify track URL
    
    Returns:
        str: Track ID or None if invalid
    """
    try:
        if validate_spotify_url(url):
            # Extract ID from URL like: https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh
            track_id = url.split('/track/')[-1].split('?')[0]
            return track_id
        return None
    except:
        return None

# Test function (optional - for debugging)
def test_search():
    """
    Test function to verify Spotify API connection
    """
    try:
        results = search_songs("test", limit=1)
        if results:
            print("Spotify API connection successful!")
            return True
        else:
            print("Spotify API connection failed - no results")
            return False
    except Exception as e:
        print(f"Spotify API connection failed: {str(e)}")
        return False

# For backward compatibility with your existing workflow
def save_selected_urls(selected_urls, filename="urls.txt"):
    """
    Legacy function - converts URL list to song dictionaries and saves
    """
    songs = []
    for url in selected_urls:
        if isinstance(url, str):
            songs.append({'url': url})
        else:
            songs.append(url)
    
    return create_urls_file(songs, filename)

# Main execution (for testing only)
if __name__ == "__main__":
    # Test the API connection
    if test_search():
        print("spotify_deployer.py is ready for Flask integration!")
    else:
        print("Please check your Spotify API credentials in .env file")
