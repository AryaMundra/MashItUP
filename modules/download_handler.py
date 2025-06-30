import os
import json
import shutil
from typing import Dict, List, Optional
import logging
from .session_manager import session_manager
from . import music_downloader, spotify_deployer

logger = logging.getLogger(__name__)

class IsolatedDownloadHandler:
    """Handle downloads in isolated user sessions"""
    
    def __init__(self):
        self.active_downloads = {}
    
    def download_for_session(self, session_id: str) -> Dict:
        """Download songs for a specific session"""
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return {'success': False, 'error': 'Session not found'}
        
        # Check if already downloaded
        if session_data.get('songs_downloaded', False):
            logger.info(f"Songs already downloaded for session {session_id}")
            return {'success': True, 'message': 'Songs already available'}
        
        try:
            # Mark as downloading
            session_manager.update_session_status(session_id, 'downloading')
            self.active_downloads[session_id] = {'status': 'downloading', 'progress': 0}
            
            # Create URLs file for this session
            selected_songs = session_data['selected_songs']
            urls_file = os.path.join(session_data['session_folder'], 'urls.txt')
            
            # Create URLs file
            success = spotify_deployer.create_urls_file(selected_songs)
            if not success:
                raise Exception("Failed to create URLs file")
            
            # Download to session's music folder
            download_result = music_downloader.download_from_urls(
                urls_file=urls_file,
                output_folder=session_data['music_folder']
            )
            
            if download_result.get('success', False):
                # Mark as downloaded
                session_manager.update_session_status(
                    session_id, 
                    'downloaded',
                    songs_downloaded=True,
                    download_metadata=download_result.get('metadata', {})
                )
                
                # Clean up URLs file
                if os.path.exists(urls_file):
                    os.remove(urls_file)
                
                logger.info(f"Successfully downloaded songs for session {session_id}")
                
                return {
                    'success': True,
                    'downloaded_count': download_result.get('successful_downloads', 0),
                    'failed_count': download_result.get('failed_downloads', 0),
                    'music_folder': session_data['music_folder']
                }
            else:
                raise Exception(download_result.get('error', 'Download failed'))
                
        except Exception as e:
            logger.error(f"Download failed for session {session_id}: {e}")
            session_manager.update_session_status(session_id, 'download_failed')
            return {'success': False, 'error': str(e)}
        
        finally:
            # Remove from active downloads
            if session_id in self.active_downloads:
                del self.active_downloads[session_id]

# Global download handler instance
download_handler = IsolatedDownloadHandler()
