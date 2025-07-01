import os
import json
import glob
import uuid
import copy
import tempfile
import logging
import time
import shutil
from datetime import datetime
from pydub import AudioSegment
from typing import List, Dict, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MashupEditor:
    def __init__(self, music_folder="downloaded_music", mashup_folder="Final_Mashup", session_id=None):
        """Initialize the mashup editor with session-aware folders"""
        logger.info("Initializing Mashup Editor...")
        
        # Store session information
        self.session_id = session_id
        self.music_folder = music_folder
        self.mashup_folder = mashup_folder
        
        # Ensure folders exist
        os.makedirs(music_folder, exist_ok=True)
        os.makedirs(mashup_folder, exist_ok=True)
        
        # Load songs and segments
        self.songs = self._load_songs()
        self.segments = self._load_segments()
        self.temp_files = []
        
        # Undo/Redo system
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
        # Timeline management
        self.timeline_positions = []
        
        logger.info(f"Loaded {len(self.songs)} songs and {len(self.segments)} segments")
        logger.info(f"Music folder: {self.music_folder}")
        logger.info(f"Mashup folder: {self.mashup_folder}")
    
    def update_session_folders(self, music_folder: str, mashup_folder: str, session_id: str = None):
        """Update folders to new session folders"""
        logger.info(f"Updating session folders: music={music_folder}, mashup={mashup_folder}")
        
        self.session_id = session_id
        self.music_folder = music_folder
        self.mashup_folder = mashup_folder
        
        # Ensure new folders exist
        os.makedirs(self.music_folder, exist_ok=True)
        os.makedirs(self.mashup_folder, exist_ok=True)
        
        # Reload songs and segments from new folders
        self.songs = self._load_songs()
        self.segments = self._load_segments()
        
        logger.info(f"Updated to session {session_id} with {len(self.songs)} songs")
    
    def _load_songs(self) -> Dict[str, Dict]:
        """Load all audio files from session's music folder"""
        songs = {}
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        
        if not os.path.exists(self.music_folder):
            logger.warning(f"Music folder {self.music_folder} does not exist")
            return songs
        
        for file in os.listdir(self.music_folder):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(self.music_folder, file)
                song_name = os.path.splitext(file)[0]
                
                try:
                    audio = AudioSegment.from_file(file_path)
                    songs[song_name] = {
                        'audio': audio,
                        'duration': len(audio),
                        'file_path': file_path
                    }
                    logger.info(f"Loaded song: {song_name} ({len(audio)/1000:.1f}s)")
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")
        
        return songs
    
    def _load_segments(self) -> List[Dict]:
        """Load segments from JSON metadata in session's mashup folder"""
        segments = []
        
        # Look for metadata files in the mashup folder
        json_files = glob.glob(os.path.join(self.mashup_folder, "*.json"))
        
        if json_files:
            try:
                # Use the first JSON file found
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    timeline_data = data.get('timeline_data', [])
                    
                    current_position = 0
                    for entry in timeline_data:
                        segment = {
                            'id': str(uuid.uuid4()),
                            'song_id': entry['song_name'],
                            'song_start': int(entry['start_time'] * 1000),
                            'song_end': int(entry['end_time'] * 1000),
                            'mashup_start': current_position,
                            'mashup_end': current_position + int((entry['end_time'] - entry['start_time']) * 1000),
                            'section': entry.get('section', 'body')
                        }
                        segments.append(segment)
                        current_position = segment['mashup_end']
                        
                logger.info(f"Loaded {len(segments)} segments from metadata")
                        
            except Exception as e:
                logger.error(f"Failed to load segments: {e}")
        else:
            logger.info("No metadata files found, starting with empty segments")
        
        return segments
    
    def save_session_metadata(self) -> bool:
        """Save current segments as metadata in session folder"""
        try:
            metadata = {
                'session_id': self.session_id,
                'created_at': time.time(),
                'timeline_data': [],
                'total_duration': self.get_total_duration() / 1000,
                'structure_info': self.get_structure_info()
            }
            
            # Convert segments to timeline data
            for segment in self.segments:
                timeline_entry = {
                    'song_name': segment['song_id'],
                    'start_time': segment['song_start'] / 1000,
                    'end_time': segment['song_end'] / 1000,
                    'duration': (segment['song_end'] - segment['song_start']) / 1000,
                    'section': segment.get('section', 'body'),
                    'mashup_position': segment['mashup_start'] / 1000
                }
                metadata['timeline_data'].append(timeline_entry)
            
            # Save to session's mashup folder
            metadata_file = os.path.join(self.mashup_folder, 'session_metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved session metadata to {metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session metadata: {e}")
            return False
    
    def get_structure_info(self) -> Dict:
        """Get structure information for metadata"""
        if not self.segments:
            return {"total_segments": 0, "structure": "empty"}
        
        structure_info = {
            "total_segments": len(self.segments),
            "total_duration": self.get_total_duration(),
            "segments_by_type": {},
            "average_segment_duration": self.get_total_duration() / len(self.segments),
            "songs_used": list(set(seg.get('song_id', 'Unknown') for seg in self.segments))
        }
        
        # Count segments by type
        for segment in self.segments:
            seg_type = segment.get('section', 'unknown')
            structure_info["segments_by_type"][seg_type] = structure_info["segments_by_type"].get(seg_type, 0) + 1
        
        return structure_info
    
    def get_songs_data(self) -> Dict[str, Dict]:
        """Get songs data for frontend"""
        songs_data = {}
        for song_id, song_info in self.songs.items():
            songs_data[song_id] = {
                'duration': song_info['duration'],
                'file_path': song_info['file_path'],
                'session_folder': self.music_folder
            }
        return songs_data
    
    def get_segments_data(self) -> List[Dict]:
        """Get segments data for frontend"""
        return [{
            'id': seg['id'],
            'song_id': seg['song_id'],
            'song_start': seg['song_start'],
            'song_end': seg['song_end'],
            'mashup_start': seg['mashup_start'],
            'mashup_end': seg['mashup_end'],
            'duration': seg['song_end'] - seg['song_start'],
            'section': seg.get('section', 'body'),
            'session_id': self.session_id
        } for seg in self.segments]
    
    def get_timeline_positions(self) -> List[Dict]:
        positions = []
        if not self.segments:
            positions.append({'position': 0, 'type': 'start'})
            return positions

        valid_segments = [seg for seg in self.segments 
                        if seg.get('mashup_start') is not None and seg.get('mashup_end') is not None]

        sorted_segments = sorted(valid_segments, key=lambda x: x['mashup_start'])

        positions.append({'position': 0, 'type': 'start'})

        for i in range(len(sorted_segments) - 1):
            current_end = int(sorted_segments[i]['mashup_end'])
            next_start = int(sorted_segments[i + 1]['mashup_start'])
            if current_end < next_start:
                positions.append({
                    'position': current_end,
                    'type': 'gap',
                    'gap_duration': next_start - current_end
                })

        if sorted_segments:
            last_end = int(sorted_segments[-1]['mashup_end'])
            positions.append({'position': last_end, 'type': 'end'})

        # Defensive: Filter out any invalid positions
        positions = [p for p in positions if p.get('position') is not None and not isinstance(p.get('position'), str)]
        return positions

    
    def check_timeline_shift_needed(self, position_ms: int) -> Tuple[bool, List[Dict]]:
        """Check if timeline shift is needed when adding at position"""
        segments_after = []
        
        for segment in self.segments:
            if segment['mashup_start'] >= position_ms:
                segments_after.append(segment)
        
        return len(segments_after) > 0, segments_after
    
    def check_overlap(self, segment_id: str, new_position: int) -> Dict:
        """Check for overlaps when moving a segment"""
        # Find the segment
        target_segment = None
        for segment in self.segments:
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            return {'has_overlap': False, 'overlapping_segments': []}
        
        # Calculate new end position
        duration = target_segment['mashup_end'] - target_segment['mashup_start']
        new_end = new_position + duration
        
        # Check for overlaps
        overlapping_segments = []
        for segment in self.segments:
            if (segment['id'] != segment_id and 
                new_position < segment['mashup_end'] and 
                new_end > segment['mashup_start']):
                overlapping_segments.append({
                    'id': segment['id'],
                    'song_id': segment['song_id']
                })
        
        return {
            'has_overlap': len(overlapping_segments) > 0,
            'overlapping_segments': overlapping_segments
        }
    
    def add_segment(self, song_id: str, song_start: float, song_end: float, section: str = 'body') -> Optional[str]:
        """Add a new segment to the end of the mashup"""
        if song_id not in self.songs:
            logger.error(f"Song {song_id} not found in session")
            return None
        
        # Find the end of the current mashup
        if self.segments:
            mashup_end = max([seg['mashup_end'] for seg in self.segments])
        else:
            mashup_end = 0
        
        # Add segment at the end
        segment_id = self.add_segment_at_position(song_id, song_start, song_end, mashup_end, section, shift_timeline=False)
        
        return segment_id
    
    def add_segment_at_position(self, song_id: str, song_start: float, song_end: float, 
                               mashup_position: int, section: str = 'body', 
                               shift_timeline: bool = True) -> Optional[str]:
        """Add segment at a specific position with proper timeline management"""
        if song_id not in self.songs:
            logger.error(f"Song {song_id} not found in session")
            return None
        
        # Save state for undo
        self._save_state(f"Add segment from {song_id} at position {mashup_position/1000:.1f}s")
        
        duration = int((song_end - song_start) * 1000)
        new_segment = {
            'id': str(uuid.uuid4()),
            'song_id': song_id,
            'song_start': int(song_start * 1000),
            'song_end': int(song_end * 1000),
            'mashup_start': mashup_position,
            'mashup_end': mashup_position + duration,
            'section': section
        }
        
        if shift_timeline:
            # Shift all segments that start at or after this position
            for segment in self.segments:
                if segment['mashup_start'] >= mashup_position:
                    segment['mashup_start'] += duration
                    segment['mashup_end'] += duration
        
        # Add the new segment
        self.segments.append(new_segment)
        self.segments.sort(key=lambda x: x['mashup_start'])
        
        # Save metadata and rebuild preview
        self.save_session_metadata()
        self._rebuild_preview_mashup()
        
        logger.info(f"Added segment from {song_id} to session {self.session_id}")
        return new_segment['id']
    
    def replace_segment(self, segment_id: str, new_song_id: str, start_time: float, end_time: float) -> bool:
        """Replace an existing segment with proper validation"""
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                if new_song_id not in self.songs:
                    logger.error(f"Song {new_song_id} not found in session")
                    return False
                
                # Save state for undo
                self._save_state(f"Replace segment with {new_song_id}")
                
                # Calculate new duration
                new_duration = int((end_time - start_time) * 1000)
                old_duration = segment['mashup_end'] - segment['mashup_start']
                
                # IMPORTANT: Preserve the mashup_start position
                mashup_start = segment['mashup_start']
                
                # Update segment with validated values
                self.segments[i].update({
                    'song_id': new_song_id,
                    'song_start': int(start_time * 1000),
                    'song_end': int(end_time * 1000),
                    'mashup_start': mashup_start,  # Keep original position
                    'mashup_end': mashup_start + new_duration  # Calculate new end
                })
                
                # Validate the updated segment
                updated_segment = self.segments[i]
                if (updated_segment['mashup_start'] is None or 
                    updated_segment['mashup_end'] is None or
                    updated_segment['mashup_start'] < 0 or
                    updated_segment['mashup_end'] <= updated_segment['mashup_start']):
                    
                    logger.error(f"Invalid segment data after replacement: {updated_segment}")
                    return False
                
                # Save metadata and rebuild preview
                self.save_session_metadata()
                self._rebuild_preview_mashup()
                
                logger.info(f"Replaced segment with {new_song_id} in session {self.session_id}")
                return True
        
        logger.error(f"Segment {segment_id} not found")
        return False

    
    def replace_segment_with_options(self, segment_id: str, song_id: str, start_time: float, 
                               end_time: float, replacement_option: str = 'adjust_timeline') -> Tuple[bool, str]:
        """Replace segment with advanced options and proper validation"""
        target_segment = None
        target_index = -1
        
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                target_segment = segment
                target_index = i
                break
        
        if not target_segment:
            return False, "Segment not found"
        
        if song_id not in self.songs:
            return False, f"Song {song_id} not found"
        
        # Save state for undo
        self._save_state(f"Replace segment with options: {replacement_option}")
        
        # Calculate durations
        old_duration = target_segment['mashup_end'] - target_segment['mashup_start']
        new_duration = int((end_time - start_time) * 1000)
        duration_diff = new_duration - old_duration
        
        # Preserve original mashup position
        original_start = target_segment['mashup_start']
        
        # Update the target segment with validation
        self.segments[target_index].update({
            'song_id': song_id,
            'song_start': int(start_time * 1000),
            'song_end': int(end_time * 1000),
            'mashup_start': original_start,
            'mashup_end': original_start + new_duration
        })
        
        # Validate the replacement
        if (self.segments[target_index]['mashup_start'] is None or 
            self.segments[target_index]['mashup_end'] is None):
            return False, "Invalid replacement data"
        
        # Handle timeline adjustment
        if replacement_option == 'adjust_timeline' and duration_diff != 0:
            # Adjust timeline for segments after this one
            for segment in self.segments:
                if segment['mashup_start'] > original_start and segment['id'] != segment_id:
                    segment['mashup_start'] += duration_diff
                    segment['mashup_end'] += duration_diff
                    
                    # Validate adjusted segments
                    if segment['mashup_start'] < 0:
                        segment['mashup_start'] = 0
                        segment['mashup_end'] = segment['mashup_end'] - segment['mashup_start']
        
        self.save_session_metadata()
        self._rebuild_preview_mashup()
        
        return True, f"Segment replaced with {replacement_option}"

    
    def get_replacement_options(self, segment_id: str, song_id: str, start_time: float, end_time: float) -> Optional[Dict]:
        """Get replacement options for a segment"""
        target_segment = None
        for segment in self.segments:
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            return None
        
        old_duration = target_segment['mashup_end'] - target_segment['mashup_start']
        new_duration = int((end_time - start_time) * 1000)
        duration_diff = new_duration - old_duration
        
        options = {
            'current_duration': old_duration / 1000,
            'new_duration': new_duration / 1000,
            'duration_difference': duration_diff / 1000,
            'replacement_options': [
                {
                    'id': 'adjust_timeline',
                    'name': 'Adjust Timeline',
                    'description': 'Shift all following segments to accommodate new duration'
                },
                {
                    'id': 'keep_timeline',
                    'name': 'Keep Timeline',
                    'description': 'Replace segment without affecting other segments'
                }
            ]
        }
        
        return options
    
    def delete_segment(self, segment_id: str) -> bool:
        """Delete a segment"""
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                # Save state for undo
                self._save_state(f"Delete segment from {segment['song_id']}")
                
                self.segments.pop(i)
                
                # Save metadata and rebuild preview
                self.save_session_metadata()
                self._rebuild_preview_mashup()
                
                logger.info(f"Deleted segment {segment_id} from session {self.session_id}")
                return True
        
        logger.error(f"Segment {segment_id} not found")
        return False
    
    def delete_segment_with_timeline_adjustment(self, segment_id: str, adjust_timeline: bool = True) -> Tuple[bool, int]:
        """Delete segment with timeline adjustment option"""
        target_segment = None
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            return False, 0
        
        # Save state for undo
        self._save_state(f"Delete segment with timeline adjustment: {adjust_timeline}")
        
        deleted_duration = target_segment['mashup_end'] - target_segment['mashup_start']
        deleted_start = target_segment['mashup_start']
        
        # Remove the segment
        self.segments.remove(target_segment)
        
        if adjust_timeline:
            # Shift all segments that start after the deleted segment
            for segment in self.segments:
                if segment['mashup_start'] > deleted_start:
                    segment['mashup_start'] -= deleted_duration
                    segment['mashup_end'] -= deleted_duration
        
        self.save_session_metadata()
        self._rebuild_preview_mashup()
        
        return True, deleted_duration
    
    def move_segment_with_overlap_handling(self, segment_id: str, new_position: int, 
                                         overlap_option: str = 'shift_timeline') -> Tuple[bool, str]:
        """Move segment with overlap handling"""
        target_segment = None
        for segment in self.segments:
            if segment['id'] == segment_id:
                target_segment = segment
                break
        
        if not target_segment:
            return False, "Segment not found"
        
        # Save state for undo
        self._save_state(f"Move segment to position {new_position/1000:.1f}s")
        
        old_start = target_segment['mashup_start']
        duration = target_segment['mashup_end'] - target_segment['mashup_start']
        new_end = new_position + duration
        
        # Check for overlaps
        overlap_info = self.check_overlap(segment_id, new_position)
        
        if overlap_info['has_overlap'] and overlap_option == 'shift_timeline':
            # Shift overlapping segments
            shift_amount = new_end - new_position
            for segment in self.segments:
                if (segment['id'] != segment_id and 
                    segment['mashup_start'] >= new_position):
                    segment['mashup_start'] += shift_amount
                    segment['mashup_end'] += shift_amount
        
        # Update target segment position
        target_segment['mashup_start'] = new_position
        target_segment['mashup_end'] = new_end
        
        # Sort segments by position
        self.segments.sort(key=lambda x: x['mashup_start'])
        
        self.save_session_metadata()
        self._rebuild_preview_mashup()
        
        return True, f"Segment moved with {overlap_option}"
    
    def get_segment_preview(self, song_id: str, start_time: float, end_time: float) -> Optional[str]:
        """Create a preview file for a segment"""
        if song_id not in self.songs:
            logger.error(f"Song {song_id} not found in session")
            return None
        
        try:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            segment_audio = self.songs[song_id]['audio'][start_ms:end_ms]
            
            # Create temporary file in session's temp folder
            temp_dir = os.path.join(self.mashup_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)
            segment_audio.export(temp_file.name, format="wav")
            self.temp_files.append(temp_file.name)
            
            logger.info(f"Created segment preview for {song_id} in session {self.session_id}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creating segment preview: {e}")
            return None
    
    def generate_mashup_preview(self) -> Optional[str]:
        """Generate a preview of the complete mashup"""
        if not self.segments:
            logger.warning("No segments to generate preview")
            return None
        
        try:
            # Find total duration
            total_duration = max([seg['mashup_end'] for seg in self.segments])
            
            # Create silent base
            mashup = AudioSegment.silent(duration=total_duration)
            
            # Sort segments by mashup start time
            sorted_segments = sorted(self.segments, key=lambda x: x['mashup_start'])
            
            # Add each segment
            for i, segment in enumerate(sorted_segments):
                if segment['song_id'] in self.songs:
                    song_audio = self.songs[segment['song_id']]['audio']
                    segment_audio = song_audio[segment['song_start']:segment['song_end']]
                    
                    if len(segment_audio) == 0:
                        continue
                    
                    # Apply fade transitions
                    fade_duration = min(500, len(segment_audio) // 4)
                    
                    if fade_duration > 0:
                        if i > 0:
                            segment_audio = segment_audio.fade_in(fade_duration)
                        if i < len(sorted_segments) - 1:
                            segment_audio = segment_audio.fade_out(fade_duration)
                    
                    # Overlay the segment
                    mashup = mashup.overlay(segment_audio, position=segment['mashup_start'])
            
            # Export to temporary file in session folder
            temp_dir = os.path.join(self.mashup_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=temp_dir)
            mashup.export(temp_file.name, format="wav")
            self.temp_files.append(temp_file.name)
            
            logger.info(f"Generated mashup preview for session {self.session_id}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to generate mashup preview: {e}")
            return None
    
    def export_mashup(self, output_path: Optional[str] = None) -> bool:
        """Export the final mashup to session's mashup folder"""
        if output_path is None:
            timestamp = int(time.time())
            filename = f"edited_mashup_{timestamp}.mp3"
            output_path = os.path.join(self.mashup_folder, filename)
        
        preview_file = self.generate_mashup_preview()
        if preview_file:
            try:
                # Copy to final location in session folder
                mashup = AudioSegment.from_wav(preview_file)
                mashup.export(output_path, format="mp3")
                
                # Also save metadata
                self.save_session_metadata()
                
                logger.info(f"Exported mashup to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to export mashup: {e}")
                return False
        return False
    
    def _rebuild_preview_mashup(self):
        """Rebuild the preview mashup based on current segments"""
        # This method is called after segment modifications
        # The actual preview is generated on-demand via generate_mashup_preview()
        pass
    
    def _save_state(self, action: str):
        """Save current state for undo"""
        state = {
            'segments': copy.deepcopy(self.segments),
            'action': action,
            'timestamp': time.time()
        }
        
        # Remove future history if we're not at the end
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(state)
        self.history_index += 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        
        logger.info(f"Saved state {self.history_index}: {action}")
    
    def undo(self) -> Tuple[bool, str]:
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.segments = copy.deepcopy(state['segments'])
            self.save_session_metadata()
            logger.info(f"Undid action: {state['action']}")
            return True, state['action']
        return False, "Nothing to undo"
    
    def redo(self) -> Tuple[bool, str]:
        """Redo last undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.segments = copy.deepcopy(state['segments'])
            self.save_session_metadata()
            logger.info(f"Redid action: {state['action']}")
            return True, state['action']
        return False, "Nothing to redo"
    
    def get_total_duration(self) -> int:
        """Get total mashup duration in milliseconds"""
        if not self.segments:
            return 0
        return max([seg['mashup_end'] for seg in self.segments])
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
                logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
        self.temp_files = []
        
        # Clean up temp directory in session folder
        temp_dir = os.path.join(self.mashup_folder, 'temp')
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

# Session-aware factory function
def create_session_editor(session_id: str, music_folder: str, mashup_folder: str) -> MashupEditor:
    """Create a MashupEditor instance for a specific session"""
    return MashupEditor(
        music_folder=music_folder,
        mashup_folder=mashup_folder,
        session_id=session_id
    )
