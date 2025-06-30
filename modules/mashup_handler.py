import os
import shutil
from typing import Dict, Optional
import logging
from .session_manager import session_manager
from .download_handler import download_handler
from . import mashup

logger = logging.getLogger(__name__)

class IsolatedMashupHandler:
    """Handle mashup creation in isolated user sessions"""
    
    def __init__(self):
        self.active_mashups = {}
    
    def create_mashup_for_session(self, session_id: str, target_duration: float = 180.0) -> Dict:
        """Create mashup for a specific session"""
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return {'success': False, 'error': 'Session not found'}
        
        try:
            # Mark as creating mashup
            session_manager.update_session_status(session_id, 'creating_mashup')
            self.active_mashups[session_id] = {'status': 'creating', 'progress': 0}
            
            # Ensure songs are downloaded
            if not session_data.get('songs_downloaded', False):
                logger.info(f"Downloading songs for session {session_id}")
                download_result = download_handler.download_for_session(session_id)
                if not download_result['success']:
                    raise Exception(f"Download failed: {download_result.get('error', 'Unknown error')}")
            
            # Create mashup using session's folders
            mashup_file, background_file = mashup.create_mashup(
                demo_folder=session_data['music_folder'],
                target_duration=target_duration,
                include_background=False
            )
            
            if mashup_file:
                # Move mashup to session's mashup folder
                session_mashup_file = os.path.join(
                    session_data['mashup_folder'], 
                    os.path.basename(mashup_file)
                )
                
                # Copy to session folder
                shutil.copy2(mashup_file, session_mashup_file)
                
                # Also copy to Final_Mashup for global access
                global_mashup_file = os.path.join('Final_Mashup', os.path.basename(mashup_file))
                shutil.copy2(mashup_file, global_mashup_file)
                
                # Update session
                session_manager.update_session_status(
                    session_id,
                    'completed',
                    mashup_file=session_mashup_file,
                    global_mashup_file=global_mashup_file
                )
                
                logger.info(f"Mashup created successfully for session {session_id}")
                
                return {
                    'success': True,
                    'mashup_file': global_mashup_file,
                    'session_id': session_id
                }
            else:
                raise Exception("Mashup creation returned no file")
                
        except Exception as e:
            logger.error(f"Mashup creation failed for session {session_id}: {e}")
            session_manager.update_session_status(session_id, 'mashup_failed')
            return {'success': False, 'error': str(e)}
        
        finally:
            # Remove from active mashups
            if session_id in self.active_mashups:
                del self.active_mashups[session_id]

# Global mashup handler instance
mashup_handler = IsolatedMashupHandler()
