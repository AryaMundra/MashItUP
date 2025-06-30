"""
Mashup Website Modules

This package contains all the core modules for the mashup creation website:
- Session management for isolated user sessions
- Download handling for audio files
- Mashup creation and processing
- Spotify API integration
- Music downloading utilities
- Audio editing capabilities
"""

__version__ = "1.0.0"
__author__ = "Mashup Creator"
__description__ = "AI-Powered Music Mashup Creation Platform"

import os
import sys
import logging

# Set up logging for the modules package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Ensure required directories exist
def ensure_directories():
    """Ensure all required directories exist"""
    required_dirs = [
        'user_sessions',
        'Final_Mashup', 
        'downloaded_music',
        'static/temp'
    ]
    
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

# Initialize directories on import
ensure_directories()

# Import core modules with error handling
try:
    from .session_manager import session_manager, UserSessionManager
    logger.info("Session manager imported successfully")
except ImportError as e:
    logger.error(f"Failed to import session_manager: {e}")
    session_manager = None
    UserSessionManager = None

try:
    from .download_handler import download_handler, IsolatedDownloadHandler
    logger.info("Download handler imported successfully")
except ImportError as e:
    logger.error(f"Failed to import download_handler: {e}")
    download_handler = None
    IsolatedDownloadHandler = None

try:
    from .mashup_handler import mashup_handler, IsolatedMashupHandler
    logger.info("Mashup handler imported successfully")
except ImportError as e:
    logger.error(f"Failed to import mashup_handler: {e}")
    mashup_handler = None
    IsolatedMashupHandler = None

# Import other modules
try:
    from . import mashup
    logger.info("Mashup module imported successfully")
except ImportError as e:
    logger.error(f"Failed to import mashup: {e}")
    mashup = None

try:
    from . import spotify_deployer
    logger.info("Spotify deployer imported successfully")
except ImportError as e:
    logger.error(f"Failed to import spotify_deployer: {e}")
    spotify_deployer = None

try:
    from . import music_downloader
    logger.info("Music downloader imported successfully")
except ImportError as e:
    logger.error(f"Failed to import music_downloader: {e}")
    music_downloader = None

try:
    from . import mashup_editor
    logger.info("Mashup editor imported successfully")
except ImportError as e:
    logger.error(f"Failed to import mashup_editor: {e}")
    mashup_editor = None

# Define what gets imported with "from modules import *"
__all__ = [
    # Core handlers
    'session_manager',
    'download_handler', 
    'mashup_handler',
    
    # Classes
    'UserSessionManager',
    'IsolatedDownloadHandler',
    'IsolatedMashupHandler',
    
    # Modules
    'mashup',
    'spotify_deployer',
    'music_downloader',
    'mashup_editor',
    
    # Utility functions
    'ensure_directories',
    'get_module_status'
]

def get_module_status():
    """Get the status of all imported modules"""
    status = {
        'session_manager': session_manager is not None,
        'download_handler': download_handler is not None,
        'mashup_handler': mashup_handler is not None,
        'mashup': mashup is not None,
        'spotify_deployer': spotify_deployer is not None,
        'music_downloader': music_downloader is not None,
        'mashup_editor': mashup_editor is not None
    }
    
    logger.info(f"Module status: {status}")
    return status

def initialize_modules():
    """Initialize all modules and check dependencies"""
    logger.info("Initializing mashup website modules...")
    
    # Check module status
    status = get_module_status()
    
    # Count successful imports
    successful_imports = sum(status.values())
    total_modules = len(status)
    
    logger.info(f"Successfully imported {successful_imports}/{total_modules} modules")
    
    if successful_imports < total_modules:
        failed_modules = [name for name, success in status.items() if not success]
        logger.warning(f"Failed to import modules: {failed_modules}")
    
    # Ensure directories are created
    ensure_directories()
    
    logger.info("Module initialization complete")
    return status

# Auto-initialize on import
try:
    initialize_modules()
except Exception as e:
    logger.error(f"Failed to initialize modules: {e}")

# Version information
def get_version_info():
    """Get detailed version information"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'python_version': sys.version,
        'modules_loaded': get_module_status()
    }

# Export version info
VERSION_INFO = get_version_info()

logger.info(f"Mashup Website Modules v{__version__} loaded successfully")
