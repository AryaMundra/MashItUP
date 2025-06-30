import os
import uuid
import time
import json
import shutil
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class UserSessionManager:
    """Manage isolated user sessions for mashup creation"""
    
    def __init__(self, base_folder="user_sessions", cleanup_interval=300):
        self.base_folder = base_folder
        self.sessions = {}
        self.cleanup_interval = cleanup_interval
        self.lock = threading.Lock()
        
        # Ensure base folder exists
        os.makedirs(base_folder, exist_ok=True)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("UserSessionManager initialized")
    
    def create_session(self, selected_songs: List[Dict]) -> str:
        """Create a new user session with isolated folders"""
        session_id = str(uuid.uuid4())
        
        # Create session folders
        session_folder = os.path.join(self.base_folder, session_id)
        music_folder = os.path.join(session_folder, "music")
        mashup_folder = os.path.join(session_folder, "mashup")
        
        os.makedirs(music_folder, exist_ok=True)
        os.makedirs(mashup_folder, exist_ok=True)
        
        # Session data
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'selected_songs': selected_songs,
            'session_folder': session_folder,
            'music_folder': music_folder,
            'mashup_folder': mashup_folder,
            'status': 'created',
            'mashup_file': None,
            'is_editing': False,
            'songs_downloaded': False
        }
        
        with self.lock:
            self.sessions[session_id] = session_data
        
        logger.info(f"Created session {session_id} for {len(selected_songs)} songs")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data and update last accessed time"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_accessed'] = datetime.now()
                return self.sessions[session_id].copy()
        return None
    
    def update_session_status(self, session_id: str, status: str, **kwargs):
        """Update session status and additional data"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['status'] = status
                self.sessions[session_id]['last_accessed'] = datetime.now()
                
                for key, value in kwargs.items():
                    self.sessions[session_id][key] = value
                
                logger.info(f"Session {session_id} status updated to {status}")
    
    def cleanup_session(self, session_id: str, force: bool = False):
        """Clean up a session's files and data"""
        with self.lock:
            if session_id not in self.sessions:
                return
            
            session_data = self.sessions[session_id]
            
            # Don't cleanup if currently editing (unless forced)
            if session_data.get('is_editing', False) and not force:
                logger.info(f"Skipping cleanup for editing session {session_id}")
                return
            
            # Remove session folder
            try:
                if os.path.exists(session_data['session_folder']):
                    shutil.rmtree(session_data['session_folder'])
                    logger.info(f"Cleaned up session folder for {session_id}")
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
            
            # Remove from memory
            del self.sessions[session_id]
            logger.info(f"Session {session_id} cleaned up")
    
    def _cleanup_worker(self):
        """Background worker to clean up expired sessions"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    def _cleanup_expired_sessions(self):
        """Clean up sessions that haven't been accessed recently"""
        current_time = datetime.now()
        expired_sessions = []
        
        with self.lock:
            for session_id, session_data in self.sessions.items():
                # Different timeouts based on status
                if session_data.get('is_editing', False):
                    timeout = timedelta(hours=2)
                elif session_data['status'] == 'completed':
                    timeout = timedelta(minutes=30)
                else:
                    timeout = timedelta(minutes=15)
                
                if current_time - session_data['last_accessed'] > timeout:
                    expired_sessions.append(session_id)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session {session_id}")
            self.cleanup_session(session_id, force=True)

# Global session manager instance
session_manager = UserSessionManager()
