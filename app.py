import os
import sys

# Add modules directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import time
import json
import uuid
import logging

# Import your modules
from modules.mashup_editor import MashupEditor
from modules import spotify_deployer, music_downloader, mashup

# IMPORT DATA HANDLERS - THIS IS CRUCIAL
from modules.session_manager import session_manager
from modules.download_handler import download_handler  
from modules.mashup_handler import mashup_handler

# Rest of your app.py code...

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here-change-this')

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

def ensure_directories():
    directories = ['user_sessions', 'Final_Mashup', 'downloaded_music', 'static/temp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

ensure_directories()

# Global variables
processing_status = {"status": "idle", "message": "", "progress": 0}
editor_instances = {}

# ===== EXISTING ROUTES =====

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
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
    global processing_status
    
    try:
        selected_songs = request.json.get('songs', [])
        if not selected_songs:
            return jsonify({'error': 'No songs selected'}), 400
        
        if len(selected_songs) < 2:
            return jsonify({'error': 'Please select at least 2 songs'}), 400
        
        processing_status = {"status": "idle", "message": "", "progress": 0}
        
        thread = threading.Thread(target=process_mashup, args=(selected_songs,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Processing started successfully', 
            'status': 'processing',
            'songs_count': len(selected_songs)
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to start processing: {str(e)}'}), 500

def process_mashup(selected_songs):
    global processing_status
    
    try:
        # Create isolated session for this user
        session_id = session_manager.create_session(selected_songs)
        
        processing_status = {
            "status": "processing", 
            "message": "Creating isolated session...", 
            "progress": 5,
            "session_id": session_id
        }
        
        # Create mashup using isolated session
        result = mashup_handler.create_mashup_for_session(session_id)
        
        if result['success']:
            processing_status = {
                "status": "completed", 
                "message": "Mashup created successfully!", 
                "progress": 100,
                "session_id": session_id,
                "files": {
                    "mashup": os.path.basename(result['mashup_file'])
                }
            }
        else:
            raise Exception(result.get('error', 'Unknown error'))
        
    except Exception as e:
        processing_status = {
            "status": "error", 
            "message": f"Error: {str(e)}", 
            "progress": 0
        }


@app.route('/api/status')
def api_status():
    return jsonify(processing_status)

@app.route('/results')
def results():
    try:
        mashup_folder = 'Final_Mashup'
        audio_files = []
        
        if os.path.exists(mashup_folder):
            all_files = os.listdir(mashup_folder)
            audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']
            audio_files = [f for f in all_files 
                          if any(f.lower().endswith(ext) for ext in audio_extensions)]
        
        return render_template('results.html', 
                             audio_files=audio_files,
                             total_files=len(audio_files),
                             error=None)
    
    except Exception as e:
        return render_template('results.html', 
                             audio_files=[], 
                             total_files=0,
                             error=str(e))

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
    
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/preview/<filename>')
def preview_file(filename):
    try:
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        if not any(filename.lower().endswith(ext) for ext in audio_extensions):
            return jsonify({'error': 'Only audio files allowed'}), 403
        
        file_path = os.path.join('Final_Mashup', filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_from_directory(
            'Final_Mashup', 
            filename, 
            as_attachment=False,
            mimetype='audio/wav'
        )
    
    except Exception as e:
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500
    
@app.route('/preview-original/<path:filename>')
def preview_original_file(filename):
    """Stream original audio files for preview"""
    try:
        # Remove any extension from filename first
        base_filename = os.path.splitext(filename)[0]
        
        # Try different extensions
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        
        # First, try to find exact filename match
        if os.path.exists(os.path.join('downloaded_music', filename)):
            return send_from_directory(
                'downloaded_music', 
                filename, 
                as_attachment=False,
                mimetype='audio/mpeg'
            )
        
        # Try with different extensions
        for ext in audio_extensions:
            test_filename = base_filename + ext
            file_path = os.path.join('downloaded_music', test_filename)
            
            if os.path.exists(file_path):
                return send_from_directory(
                    'downloaded_music', 
                    test_filename, 
                    as_attachment=False,
                    mimetype=f'audio/{ext[1:]}'
                )
        
        # Debug: List all files in downloaded_music folder
        if os.path.exists('downloaded_music'):
            available_files = os.listdir('downloaded_music')
            logger.info(f"Available files in downloaded_music: {available_files}")
            logger.info(f"Looking for: {filename} or {base_filename}")
        
        return jsonify({'error': f'Audio file not found: {filename}'}), 404
    
    except Exception as e:
        logger.error(f"Preview failed for {filename}: {e}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500



# ===== EDITOR ROUTES =====

@app.route('/edit/<filename>')
def edit_mashup(filename):
    try:
        # Get session ID from query params
        session_id = request.args.get('session_id')
        
        if session_id:
            # Prepare existing session for editing
            result = mashup_handler.prepare_for_editing(session_id, filename)
        else:
            # Create new session for editing
            result = mashup_handler.prepare_for_editing(None, filename)
            session_id = result.get('session_id')
        
        if not result['success']:
            return render_template('error.html', 
                                 error_code=500, 
                                 error_message=result['error']), 500
        
        # Create editor instance with session-specific folders
        session_data = session_manager.get_session(session_id)
        editor = MashupEditor(
            music_folder=session_data['music_folder'],
            mashup_folder=session_data['mashup_folder']
        )
        
        editor_instances[session_id] = editor
        session['editor_session_id'] = session_id
        
        # Get data for template
        songs_data = editor.get_songs_data() or {}
        segments_data = editor.get_segments_data() or []
        
        return render_template('editor.html', 
                             songs=songs_data,
                             segments=segments_data,
                             filename=filename,
                             session_id=session_id)
    
    except Exception as e:
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

# ===== INITIALIZATION =====

def initialize_app():
    try:
        directories = ['downloaded_music', 'Final_Mashup', 'static/temp']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        
        print("✓ Application initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Application initialization failed: {e}")
        return False

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

