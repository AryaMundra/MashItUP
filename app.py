import os
import sys
import threading
import uuid
import time
import logging
from datetime import timedelta
import shutil

# Add modules directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import your modules
from modules.mashup_editor import MashupEditor
from modules import spotify_deployer, music_downloader, mashup
from modules.session_manager import session_manager
from modules.download_handler import download_handler
from modules.mashup_handler import mashup_handler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# CORRECT - Check for production environment
if os.environ.get('PYTHON_ENV') == 'production' or os.environ.get('PORT'):
    app.config['DEBUG'] = False
    logging.basicConfig(level=logging.INFO)
else:
    app.config['DEBUG'] = True
    logging.basicConfig(level=logging.DEBUG)

# Initialize Socket.IO
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',
    logger=False,
    engineio_logger=False
)

# Global variables
processing_statuses = {}  # user_id -> status
user_sessions = {}  # user_id -> session_id
editor_instances = {}

def ensure_directories():
    directories = ['user_sessions', 'Final_Mashup', 'downloaded_music', 'static/temp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

ensure_directories()

def get_user_id():
    """Get or create a unique user ID for this browser session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session.permanent = True
    return session['user_id']

def process_mashup(selected_songs, user_id):
    """Process mashup for a specific user"""
    global processing_statuses, user_sessions
    
    try:
        # Create isolated session for this user
        session_id = session_manager.create_session(selected_songs)
        
        # Store session ID for this user
        user_sessions[user_id] = session_id
        
        logger.info(f"Created session {session_id} for user {user_id}")
        
        processing_statuses[user_id] = {
            "status": "processing", 
            "message": "Creating mashup...This may take some time", 
            "progress": 5,
            "session_id": session_id
        }
        
        # Create mashup using isolated session
        result = mashup_handler.create_mashup_for_session(session_id)
        
        if result.get('success', False):
            processing_statuses[user_id] = {
                "status": "completed", 
                "message": "Mashup created successfully!", 
                "progress": 100,
                "session_id": session_id,
                "files": {
                    "mashup": os.path.basename(result['mashup_file'])
                }
            }
            
            logger.info(f"Mashup completed for user {user_id}, session {session_id}")
        else:
            raise Exception(result.get('error', 'Unknown error'))
        
    except Exception as e:
        logger.error(f"Process mashup error for user {user_id}: {e}")
        processing_statuses[user_id] = {
            "status": "error", 
            "message": f"Error: {str(e)}", 
            "progress": 0
        }

# ===== EXISTING ROUTES =====

@app.route('/')
def index():
    get_user_id()  # Ensure user has an ID
    return render_template('index.html')

@app.route('/search')
def search():
    get_user_id()  # Ensure user has an ID
    return render_template('search.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        search_query = request.json.get('query', '')
        if not search_query:
            return jsonify({'error': 'No search query provided'}), 400
        
        results = spotify_deployer.search_songs(search_query, limit=20)
        
        if not results:
            return jsonify({'error': 'No songs found for your search'}), 404
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def api_process():
    """Process selected songs into mashup"""
    try:
        user_id = get_user_id()
        data = request.get_json()
        selected_songs = data.get('songs', [])
        
        if not selected_songs:
            return jsonify({'error': 'No songs provided'}), 400
        
        if len(selected_songs) < 2:
            return jsonify({'error': 'At least 2 songs required'}), 400
        
        # Start processing for this specific user
        thread = threading.Thread(target=process_mashup, args=(selected_songs, user_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'processing',
            'message': 'Mashup creation started',
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"API process error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """Get processing status for current user"""
    try:
        user_id = get_user_id()
        user_status = processing_statuses.get(user_id, {'status': 'unknown'})
        
        # If completed, store session ID in Flask session
        if user_status.get('status') == 'completed':
            session_id = user_sessions.get(user_id)
            if session_id:
                session['editor_session_id'] = session_id
                session['current_mashup_session'] = session_id
                logger.info(f"Stored session ID {session_id} in Flask session for user {user_id}")
        
        return jsonify(user_status)
        
    except Exception as e:
        logger.error(f"API status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    """Show user's mashups from both session and global storage"""
    try:
        user_id = get_user_id()
        session_id = user_sessions.get(user_id)
        
        # Also store in Flask session for template access
        if session_id:
            session['editor_session_id'] = session_id
            session['current_mashup_session'] = session_id
        
        logger.info(f"Results page - User: {user_id}, Session ID: {session_id}")
        
        # Get files from session folder if session exists
        session_files = []
        if session_id:
            session_data = session_manager.get_session(session_id)
            if session_data and 'mashup_folder' in session_data:
                mashup_folder = session_data['mashup_folder']
                if os.path.exists(mashup_folder):
                    session_files = [f for f in os.listdir(mashup_folder) 
                                   if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
        
        # Get files from global Final_Mashup folder
        global_files = []
        if os.path.exists('Final_Mashup'):
            global_files = [f for f in os.listdir('Final_Mashup') 
                          if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
        
        # Combine both lists
        all_files = {
            'session_files': session_files,
            'global_files': global_files,
            'session_id': session_id
        }
        
        return render_template('results.html', 
                             audio_files=all_files, 
                             session_id=session_id)
    
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return render_template('results.html', 
                             audio_files={'session_files': [], 'global_files': []},
                             session_id=None)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        if not any(filename.lower().endswith(ext) for ext in audio_extensions):
            return jsonify({'error': 'Only audio files allowed'}), 403
        
        file_path = os.path.abspath(os.path.join('Final_Mashup', filename))
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    except:
        return None

@app.route('/download-session/<session_id>/<filename>')
def download_session_file(session_id, filename):
    """Download file from user session"""
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return "Session not found", 404
        
        mashup_folder = session_data['mashup_folder']
        file_path = os.path.join(mashup_folder, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return "File not found", 404
    
    except Exception as e:
        logger.error(f"Error downloading session file: {e}")
        return "Download failed", 500

@app.route('/preview-session/<session_id>/<filename>')
def preview_session_file(session_id, filename):
    """Preview file from user session"""
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            return "Session not found", 404
        
        mashup_folder = session_data['mashup_folder']
        file_path = os.path.join(mashup_folder, filename)
        
        if os.path.exists(file_path):
            return send_file(file_path, mimetype="audio/wav")
        else:
            return "File not found", 404
    
    except Exception as e:
        logger.error(f"Error previewing session file: {e}")
        return "Preview failed", 500

@app.route('/api/session-keepalive', methods=['POST'])
def session_keepalive():
    """Keep session alive during editing"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_manager:
            # Update last accessed time
            session_data = session_manager.get_session(session_id)
            if session_data:
                session_manager.update_session_status(session_id, 'editing', is_editing=True)
                return jsonify({'success': True})
        
        return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== EDITOR ROUTES (EXACT SAME AS BEFORE) =====

@app.route('/edit/<filename>')
def edit_mashup(filename):
    try:
        # Get session ID from query params
        session_id = request.args.get('session_id')
        print(f"✔ Debug: Requested session {session_id}")
        
        if session_id:
            # Try to get existing session
            session_data = session_manager.get_session(session_id)
            print(f"✔ Debug: Session exists: {session_data is not None}")
            
            if not session_data:
                # Session was cleaned up, recreate it
                print(f"⚠️ Debug: Session {session_id} was cleaned up, recreating for editing")
                logger.info(f"Session {session_id} was cleaned up, recreating for editing")
                
                # Find the user who owns this session
                user_id = None
                print(f"🔍 Debug: Searching for user in {len(user_sessions)} user sessions")
                for uid, sid in user_sessions.items():
                    print(f"   - User {uid[:8]}... has session {sid[:8]}...")
                    if sid == session_id:
                        user_id = uid
                        break
                
                if user_id:
                    print(f"✔ Debug: Found user {user_id[:8]}... for session, recreating")
                    # Recreate session for editing
                    new_session_id = session_manager.create_session([])
                    user_sessions[user_id] = new_session_id
                    session_id = new_session_id
                    session_data = session_manager.get_session(session_id)
                    
                    # Mark as editing to prevent cleanup
                    session_manager.update_session_status(session_id, 'editing', is_editing=True)
                    print(f"✔ Debug: Created new session {session_id[:8]}... for editing")
                else:
                    print(f"⚠️ Debug: No user found for session, creating fallback session")
                    # Create completely new session
                    session_id = str(uuid.uuid4())
                    session_data = {
                        'music_folder': 'downloaded_music',
                        'mashup_folder': 'Final_Mashup'
                    }
                    print(f"✔ Debug: Created fallback session {session_id[:8]}...")
            else:
                print(f"✔ Debug: Session exists, marking as editing")
                # Session exists, mark as editing
                session_manager.update_session_status(session_id, 'editing', is_editing=True)
                print(f"✔ Debug: Session {session_id[:8]}... marked as editing")
        else:
            print(f"⚠️ Debug: No session ID provided, creating new session")
            # No session ID provided, create new
            session_id = str(uuid.uuid4())
            session_data = {
                'music_folder': 'downloaded_music',
                'mashup_folder': 'Final_Mashup'
            }
            print(f"✔ Debug: Created new session {session_id[:8]}... without session ID")
        
        print(f"🔍 Debug: Session data folders:")
        print(f"   - Music folder: {session_data.get('music_folder', 'N/A')}")
        print(f"   - Mashup folder: {session_data.get('mashup_folder', 'N/A')}")
        
        # Create editor instance
        print(f"🔧 Debug: Creating MashupEditor instance")
        editor = MashupEditor(
            music_folder=session_data.get('music_folder', 'downloaded_music'),
            mashup_folder=session_data.get('mashup_folder', 'Final_Mashup'),
            session_id=session_id
        )
        
        print(f"✔ Debug: MashupEditor created successfully")
        
        editor_instances[session_id] = editor
        session['editor_session_id'] = session_id
        
        print(f"🔍 Debug: Stored editor in instances, total editors: {len(editor_instances)}")
        
        # Get data for template
        print(f"📊 Debug: Getting songs and segments data")
        songs_data = editor.get_songs_data() or {}
        segments_data = editor.get_segments_data() or []
        
        print(f"✔ Debug: Data retrieved:")
        print(f"   - Songs: {len(songs_data)} songs")
        print(f"   - Segments: {len(segments_data)} segments")
        
        print(f"✔ Debug: Editor loaded successfully for session {session_id[:8]}...")
        
        return render_template('editor.html', 
                             songs=songs_data,
                             segments=segments_data,
                             filename=filename,
                             session_id=session_id)
    
    except Exception as e:
        print(f"❌ Debug: Error loading editor: {e}")
        print(f"❌ Debug: Error type: {type(e).__name__}")
        import traceback
        print(f"❌ Debug: Full traceback:")
        traceback.print_exc()
        
        logger.error(f"Error loading editor: {e}")
        return render_template('error.html', 
                             error_code=500, 
                             error_message=f"Failed to load editor: {str(e)}"), 500


@app.route('/api/editor/songs')
def get_editor_songs():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        songs_data = editor.get_songs_data()
        
        return jsonify({'songs': songs_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/segments')
def get_editor_segments():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        segments_data = editor.get_segments_data()
        
        return jsonify({'segments': segments_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/used-segments/<song_id>')
def get_used_segments(song_id):
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        used_segments = editor.get_used_segments_for_song(song_id)
        
        return jsonify({'used_segments': used_segments})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/add-segment', methods=['POST'])
def add_segment():
    """Simple add segment to end of mashup"""
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        section = data.get('section', 'body')
        
        segment_id = editor.add_segment(song_id, start_time, end_time, section)
        
        if segment_id:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'segment_id': segment_id, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'error': 'Failed to add segment'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/replacement-options', methods=['POST'])
def get_replacement_options():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        options = editor.get_replacement_options(segment_id, song_id, start_time, end_time)
        
        if options:
            return jsonify({'success': True, 'options': options})
        else:
            return jsonify({'success': False, 'error': 'Segment not found'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/replace-segment-advanced', methods=['POST'])
def replace_segment_advanced():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        replacement_option = data.get('replacement_option', 'adjust_timeline')
        
        success, message = editor.replace_segment_with_options(
            segment_id, song_id, start_time, end_time, replacement_option
        )
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'message': message, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'error': message})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/replace-segment', methods=['POST'])
def replace_segment():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        success = editor.replace_segment(segment_id, song_id, start_time, end_time)
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'error': 'Failed to replace segment'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/delete-segment-advanced', methods=['POST'])
def delete_segment_advanced():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        adjust_timeline = data.get('adjust_timeline', True)
        
        success, duration = editor.delete_segment_with_timeline_adjustment(segment_id, adjust_timeline)
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({
                'success': True, 
                'segments': segments_data,
                'deleted_duration': duration / 1000,
                'timeline_adjusted': adjust_timeline
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to delete segment'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/timeline-positions')
def get_timeline_positions():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        positions = editor.get_timeline_positions()
        
        return jsonify({'positions': positions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/add-segment-at-position', methods=['POST'])
def add_segment_at_position():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        position_ms = data.get('position_ms')
        section = data.get('section', 'body')
        
        segment_id = editor.add_segment_at_position(song_id, start_time, end_time, position_ms, section)
        
        if segment_id:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'segment_id': segment_id, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'error': 'Failed to add segment'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/delete-segment', methods=['POST'])
def delete_segment():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        
        success = editor.delete_segment(segment_id)
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete segment'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/check-overlap', methods=['POST'])
def check_overlap():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        new_position = data.get('new_position')
        
        # Find the segment
        target_segment = None
        for segment in editor.segments:
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            return jsonify({'error': 'Segment not found'}), 400
        
        # Calculate new end position
        duration = target_segment['mashup_end'] - target_segment['mashup_start']
        new_end = new_position + duration
        
        # Check for overlaps
        overlapping_segments = []
        for segment in editor.segments:
            if (segment['id'] != segment_id and 
                new_position < segment['mashup_end'] and 
                new_end > segment['mashup_start']):
                overlapping_segments.append({
                    'id': segment['id'],
                    'song_id': segment['song_id']
                })
        
        return jsonify({
            'has_overlap': len(overlapping_segments) > 0,
            'overlapping_segments': overlapping_segments
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/move-segment', methods=['POST'])
def move_segment():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        segment_id = data.get('segment_id')
        new_position = data.get('new_position')
        overlap_option = data.get('overlap_option', 'shift_timeline')
        
        success, message = editor.move_segment_with_overlap_handling(
            segment_id, new_position, overlap_option
        )
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({
                'success': True, 
                'message': message,
                'segments': segments_data
            })
        else:
            return jsonify({'success': False, 'error': message})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/check-timeline-shift', methods=['POST'])
def check_timeline_shift():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        position_ms = data.get('position_ms')
        
        needs_shift, segments_after = editor.check_timeline_shift_needed(position_ms)
        
        return jsonify({
            'needs_shift': needs_shift,
            'segments_count': len(segments_after),
            'segments_after': [{'song_id': seg['song_id'], 'start': seg['mashup_start']} for seg in segments_after]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/add-segment-with-shift-option', methods=['POST'])
def add_segment_with_shift_option():
    """Add segment with user choice for timeline shifting"""
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        position_ms = data.get('position_ms')
        section = data.get('section', 'body')
        shift_timeline = data.get('shift_timeline', True)
        
        segment_id = editor.add_segment_at_position(
            song_id, start_time, end_time, position_ms, section, shift_timeline
        )
        
        if segment_id:
            segments_data = editor.get_segments_data()
            return jsonify({
                'success': True, 
                'segment_id': segment_id, 
                'segments': segments_data,
                'timeline_shifted': shift_timeline
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to add segment'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/preview-segment', methods=['POST'])
def preview_segment():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        data = request.json
        song_id = data.get('song_id')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        preview_file = editor.get_segment_preview(song_id, start_time, end_time)
        
        if preview_file:
            return send_file(preview_file, as_attachment=False, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Failed to create preview'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/preview-mashup')
def preview_mashup():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        preview_file = editor.generate_mashup_preview()
        
        if preview_file:
            return send_file(preview_file, as_attachment=False, mimetype='audio/wav')
        else:
            return jsonify({'error': 'Failed to generate preview'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
from flask import send_file, abort

@app.route('/preview-original/<song_id>')
def preview_original(song_id):
    """
    Stream the original song audio for preview in the editor.
    """
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return abort(404, description="No active editor session")
        
        editor = editor_instances[session_id]
        
        # Try exact match
        song_data = editor.songs.get(song_id)
        
        # If not found, try without extension
        if not song_data:
            song_id_no_ext = os.path.splitext(song_id)[0]
            song_data = editor.songs.get(song_id_no_ext)
        
        if not song_data:
            return abort(404, description="Song not found in session")
        
        file_path = song_data['file_path']
        if not os.path.exists(file_path):
            return abort(404, description="File does not exist")
        
        ext = os.path.splitext(file_path)[1].lower()
        mimetype = "audio/mpeg" if ext == ".mp3" else "audio/wav"
        
        return send_file(file_path, mimetype=mimetype)
    
    except Exception as e:
        logger.error(f"Error in preview-original: {e}")
        return abort(500, description=str(e))

@app.route('/api/editor/export', methods=['POST'])
def export_mashup():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        
        output_filename = f"edited_mashup_{int(time.time())}.mp3"
        output_path = os.path.join('Final_Mashup', output_filename)
        
        success = editor.export_mashup(output_path)
        
        if success:
            return jsonify({
                'success': True,
                'filename': output_filename,
                'download_url': f'/download/{output_filename}'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to export mashup'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/undo', methods=['POST'])
def undo_action():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        success, message = editor.undo()
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'message': message, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/editor/redo', methods=['POST'])
def redo_action():
    try:
        session_id = session.get('editor_session_id')
        if not session_id or session_id not in editor_instances:
            return jsonify({'error': 'No editor session found'}), 400
        
        editor = editor_instances[session_id]
        success, message = editor.redo()
        
        if success:
            segments_data = editor.get_segments_data()
            return jsonify({'success': True, 'message': message, 'segments': segments_data})
        else:
            return jsonify({'success': False, 'message': message})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== SOCKET.IO EVENTS =====

@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    session['socket_session_id'] = session_id
    join_room(session_id)
    emit('connected', {'session_id': session_id})
    print(f'Client connected: {session_id}')

@socketio.on('disconnect')
def handle_disconnect():
    session_id = session.get('socket_session_id')
    if session_id:
        leave_room(session_id)
    print(f'Client disconnected: {session_id}')

# ===== CLEANUP =====

@app.route('/api/cleanup', methods=['POST'])
def cleanup_editor():
    try:
        session_id = session.get('editor_session_id')
        if session_id and session_id in editor_instances:
            editor = editor_instances[session_id]
            editor.cleanup()
            del editor_instances[session_id]
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/cleanup-session', methods=['POST'])
def cleanup_session():
    """Cleanup session data"""
    try:
        user_id = get_user_id()
        data = request.get_json() or {}
        
        if user_id and user_id in user_sessions:
            session_id = user_sessions[user_id]
            
            # Cleanup session manager
            if session_manager:
                session_manager.cleanup_session(session_id, force=True)
            
            # Remove from our tracking
            del user_sessions[user_id]
            if user_id in processing_statuses:
                del processing_statuses[user_id]
            
            # Cleanup editor instance if exists
            if session_id in editor_instances:
                editor = editor_instances[session_id]
                editor.cleanup()
                del editor_instances[session_id]
            
            logger.info(f"Cleaned up session for user {user_id}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/extend-session', methods=['POST'])
def extend_session():
    """Extend session timeout for editing"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        

        
        return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===== MAIN EXECUTION =====

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_production = bool(os.environ.get('PORT'))
    
    if is_production:
        # Production mode on Render
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=False,
            log_output=True
        )
    else:
        # Development mode
        print("\n🎵 Mashup Editor Starting...")
        print(f"🔗 Access the app at: http://localhost:{port}")
        socketio.run(
            app,
            debug=True,
            host='0.0.0.0',
            port=port,
            allow_unsafe_werkzeug=True
        )
