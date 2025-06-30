import os
import subprocess
import sys
from pathlib import Path
import shutil
import glob
import time
import json

def run_freyr_command(cmd_args, **kwargs):
    """Run freyr command with Windows compatibility"""
    if os.name == 'nt':  # Windows
        return subprocess.run(['freyr'] + cmd_args, shell=True, **kwargs)
    else:  # Unix/Linux/Mac
        return subprocess.run(['freyr'] + cmd_args, **kwargs)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    if not shutil.which('node'):
        print("❌ Node.js is not installed")
        return False
    else:
        print("✓ Node.js is installed")
    
    if not shutil.which('npm'):
        print("❌ npm is not installed")
        return False
    else:
        print("✓ npm is installed")
    
    try:
        result = run_freyr_command(['--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ freyr is installed and working")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("❌ freyr is installed but not working properly")
            # Try to install freyr if it's missing
            try:
                print("🔄 Attempting to install freyr...")
                install_result = subprocess.run(['npm', 'install', '-g', 'freyr'], 
                                                capture_output=True, text=True, timeout=60)
                if install_result.returncode == 0:
                    print("✓ freyr installed successfully")
                    return True
                else:
                    print(f"❌ Failed to install freyr: {install_result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error installing freyr: {e}")
                return False
    except Exception as e:
        print(f"❌ freyr check failed: {e}")
        return False
def read_urls_from_file(file_path):
    """Read URLs from a text file, one URL per line"""
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                url = line.strip()
                if url and not url.startswith('#'):
                    urls.append(url)
        return urls
    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []

def flatten_directory(source_dir, target_dir):
    """Move all audio files from nested folders to target directory"""
    audio_extensions = ['.m4a', '.mp3', '.flac', '.wav', '.ogg']
    moved_files = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                
                # Handle duplicate filenames
                counter = 1
                original_target = target_path
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(original_target)
                    target_path = f"{name}_{counter}{ext}"
                    counter += 1
                
                try:
                    shutil.move(source_path, target_path)
                    moved_files.append(target_path)
                    print(f"📁 Moved: {os.path.basename(target_path)}")
                except Exception as e:
                    print(f"❌ Failed to move {file}: {e}")
    
    return moved_files

def clean_empty_directories(directory):
    """Remove empty directories recursively"""
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    print(f"🗑️  Removed empty directory: {dir_name}")
            except:
                pass

def extract_metadata_from_filename(filepath):
    """Extract artist and title from filename"""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    if ' - ' in filename:
        parts = filename.split(' - ', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
    elif ' by ' in filename:
        parts = filename.split(' by ', 1)
        title = parts[0].strip()
        artist = parts[1].strip()
    else:
        artist = "Unknown Artist"
        title = filename
    
    return artist, title

def get_spotify_metadata(url):
    """Extract Spotify track ID and get basic metadata"""
    try:
        if 'spotify.com/track/' in url:
            track_id = url.split('track/')[-1].split('?')[0]
            return {
                'source': 'spotify',
                'track_id': track_id,
                'url': url
            }
        elif 'music.apple.com' in url:
            return {
                'source': 'apple_music',
                'url': url
            }
        elif 'deezer.com' in url:
            return {
                'source': 'deezer',
                'url': url
            }
    except:
        pass
    return {'source': 'unknown', 'url': url}

def save_metadata_to_json(songs_metadata, output_folder):
    """Save all metadata to a JSON file"""
    metadata_file = os.path.join(output_folder, 'songs_metadata.json')
    
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(songs_metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Metadata saved to: songs_metadata.json")
    except Exception as e:
        print(f"❌ Failed to save metadata: {e}")

def download_songs(urls, output_folder):
    """Download songs in m4a format and organize them properly"""
    if not urls:
        print("❌ No URLs to process")
        return False
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output folder: {os.path.abspath(output_folder)}")
    print(f"🎵 Found {len(urls)} URLs to download")
    print(f"🎧 Audio format: M4A (default high quality)")
    
    successful_downloads = 0
    failed_downloads = 0
    all_songs_metadata = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        print(f"🎵 Starting download in M4A format...")
        
        try:
            original_dir = os.getcwd()
            
            # Create temporary download directory
            temp_download_dir = os.path.join(output_folder, f'temp_download_{i}')
            Path(temp_download_dir).mkdir(exist_ok=True)
            
            # Change to temp directory
            os.chdir(temp_download_dir)
            
            # Use freyr default command (downloads in m4a format)
            freyr_cmd = [url]
            
            print(f"🚀 Running freyr (m4a format)...")
            result = run_freyr_command(freyr_cmd, timeout=300)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            if result.returncode == 0:
                # Flatten the directory structure
                moved_files = flatten_directory(temp_download_dir, output_folder)
                
                if moved_files:
                    for file_path in moved_files:
                        artist, title = extract_metadata_from_filename(file_path)
                        filename = os.path.basename(file_path)
                        file_extension = os.path.splitext(filename)[1].lower()
                        
                        # Get additional metadata
                        source_metadata = get_spotify_metadata(url)
                        
                        song_metadata = {
                            'filename': filename,
                            'artist': artist,
                            'title': title,
                            'album': 'Unknown Album',
                            'source_url': url,
                            'source_info': source_metadata,
                            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'file_format': file_extension.replace('.', ''),
                            'file_size_mb': round(os.path.getsize(file_path) / (1024*1024), 2)
                        }
                        
                        all_songs_metadata.append(song_metadata)
                        
                        print(f"✓ Successfully downloaded: {title} by {artist}")
                        print(f"📁 File: {filename}")
                        print(f"🎧 Format: {file_extension.upper()}")
                    
                    successful_downloads += 1
                else:
                    print("❌ No files were moved")
                    failed_downloads += 1
                
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_download_dir)
                except:
                    pass
                
            else:
                print("❌ Download failed")
                failed_downloads += 1
                
        except subprocess.TimeoutExpired:
            print("❌ Download timed out")
            failed_downloads += 1
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            failed_downloads += 1
        finally:
            try:
                os.chdir(original_dir)
                # Clean up any remaining temp directories
                if os.path.exists(temp_download_dir):
                    shutil.rmtree(temp_download_dir)
            except:
                pass
    
    # Clean up any empty directories
    clean_empty_directories(output_folder)
    
    # Save all metadata to JSON file
    if all_songs_metadata:
        save_metadata_to_json(all_songs_metadata, output_folder)
    
    print(f"\n📊 Download Summary:")
    print(f"✓ Successful: {successful_downloads}")
    print(f"❌ Failed: {failed_downloads}")
    print(f"📁 Files saved to: {os.path.abspath(output_folder)}")
    
    if all_songs_metadata:
        print(f"\n🎵 Downloaded Songs:")
        print("=" * 50)
        for song in all_songs_metadata:
            print(f"🎤 Artist: {song['artist']}")
            print(f"🎵 Song: {song['title']}")
            print(f"📁 File: {song['filename']}")
            print(f"🎧 Format: {song['file_format'].upper()}")
            print(f"💾 Size: {song['file_size_mb']} MB")
            print("-" * 30)
    
    # Return success status and metadata for Flask
    return {
        'success': successful_downloads > 0,
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'total_urls': len(urls),
        'metadata': all_songs_metadata,
        'output_folder': os.path.abspath(output_folder)
    }

# Flask-compatible functions
def download_from_urls(urls_file="urls.txt", output_folder="downloaded_music"):
    """
    Flask-compatible function to download songs from URLs file
    
    Args:
        urls_file (str): Path to file containing URLs
        output_folder (str): Output directory for downloaded songs
    
    Returns:
        dict: Download results with success status and metadata
    """
    # Check dependencies first
    if not check_dependencies():
        return {
            'success': False,
            'error': 'Missing dependencies (Node.js, npm, or freyr)',
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_urls': 0,
            'metadata': [],
            'output_folder': output_folder
        }
    
    # Read URLs from file
    urls = read_urls_from_file(urls_file)
    
    if not urls:
        return {
            'success': False,
            'error': f'No URLs found in {urls_file}',
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_urls': 0,
            'metadata': [],
            'output_folder': output_folder
        }
    
    # Download songs
    return download_songs(urls, output_folder)

def download_from_url_list(url_list, output_folder="downloaded_music"):
    """
    Flask-compatible function to download songs from a list of URLs
    
    Args:
        url_list (list): List of URLs to download
        output_folder (str): Output directory for downloaded songs
    
    Returns:
        dict: Download results with success status and metadata
    """
    # Check dependencies first
    if not check_dependencies():
        return {
            'success': False,
            'error': 'Missing dependencies (Node.js, npm, or freyr)',
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_urls': 0,
            'metadata': [],
            'output_folder': output_folder
        }
    
    if not url_list:
        return {
            'success': False,
            'error': 'No URLs provided',
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_urls': 0,
            'metadata': [],
            'output_folder': output_folder
        }
    
    # Download songs
    return download_songs(url_list, output_folder)

def get_download_status(output_folder="downloaded_music"):
    """
    Get status of downloaded files
    
    Args:
        output_folder (str): Directory to check for downloaded files
    
    Returns:
        dict: Status information about downloaded files
    """
    try:
        if not os.path.exists(output_folder):
            return {
                'folder_exists': False,
                'audio_files': [],
                'metadata_file': None,
                'total_files': 0
            }
        
        files = os.listdir(output_folder)
        audio_files = [f for f in files if f.endswith(('.m4a', '.mp3', '.flac', '.wav', '.ogg'))]
        metadata_file = 'songs_metadata.json' if 'songs_metadata.json' in files else None
        
        return {
            'folder_exists': True,
            'audio_files': audio_files,
            'metadata_file': metadata_file,
            'total_files': len(audio_files),
            'output_folder': os.path.abspath(output_folder)
        }
    
    except Exception as e:
        return {
            'folder_exists': False,
            'error': str(e),
            'audio_files': [],
            'metadata_file': None,
            'total_files': 0
        }

# Keep original main function for standalone usage
def main():
    urls_file = "urls.txt"
    output_folder = "downloaded_music"
    
    print("🎵 Freyr Music Downloader - M4A Format")
    print("=" * 45)
    
    result = download_from_urls(urls_file, output_folder)
    
    if not result['success'] and 'error' in result:
        print(f"\n❌ {result['error']}")
        sys.exit(1)
    
    print(f"\n📂 Final folder structure:")
    status = get_download_status(output_folder)
    
    if status['folder_exists']:
        print(f"🎵 Audio files ({status['total_files']}):")
        for file in status['audio_files']:
            print(f"  📄 {file}")
        
        if status['metadata_file']:
            print(f"📋 Metadata file: {status['metadata_file']}")
    else:
        print("  (folder not accessible)")

if __name__ == "__main__":
    main()
