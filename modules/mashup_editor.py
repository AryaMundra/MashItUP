import os
import json
import glob
import uuid
import copy
import tempfile
import logging
import time
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MashupEditor:
    def __init__(self, music_folder="downloaded_music", mashup_folder="Final_Mashup"):
        """Initialize the mashup editor"""
        logger.info("Initializing Mashup Editor...")
        
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
        self.max_history= 50
        
        logger.info(f"Loaded {len(self.songs)} songs and {len(self.segments)} segments")
    
    def _load_songs(self):
        """Load all audio files from music folder"""
        songs = {}
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        
        if not os.path.exists(self.music_folder):
            logger.warning(f"Music folder {self.music_folder} does not exist")
            return songs
        
        for file in os.listdir(self.music_folder):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(self.music_folder, file)
                song_name = os.path.splitext(file)[0]  # Fixed: removed extra [0]
                
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
    
    def _load_segments(self):
        """Load segments from JSON metadata"""
        segments = []
        json_files = glob.glob(os.path.join(self.mashup_folder, "*.json"))
        
        if json_files:
            try:
                with open(json_files[0], 'r', encoding='utf-8') as f:  # Fixed: added [0]
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
                        
            except Exception as e:
                logger.error(f"Failed to load segments: {e}")
        
        return segments
    
    def get_songs_data(self):
        """Get songs data for frontend"""
        songs_data = {}
        for song_id, song_info in self.songs.items():
            songs_data[song_id] = {
                'duration': song_info['duration'],
                'file_path': song_info['file_path']
            }
        return songs_data
    
    def get_segments_data(self):
        """Get segments data for frontend"""
        return [{
            'id': seg['id'],
            'song_id': seg['song_id'],
            'song_start': seg['song_start'],
            'song_end': seg['song_end'],
            'mashup_start': seg['mashup_start'],
            'mashup_end': seg['mashup_end'],
            'duration': seg['song_end'] - seg['song_start'],
            'section': seg.get('section', 'body')
        } for seg in self.segments]
    
    def get_used_segments_for_song(self, song_id):
        """Get all used segments for a specific song"""
        used_segments = []
        for segment in self.segments:
            if segment['song_id'] == song_id:
                used_segments.append({
                    'start': segment['song_start'] / 1000,
                    'end': segment['song_end'] / 1000,
                    'section': segment['section']
                })
        return used_segments
    
    def add_segment(self, song_id, song_start, song_end, section='body'):
        """Add a new segment to the end of the mashup (simple version)"""
        if song_id not in self.songs:
            logger.error(f"Song {song_id} not found")
            return None
        
        # Find the end of the current mashup
        if self.segments:
            mashup_end = max([seg['mashup_end'] for seg in self.segments])
        else:
            mashup_end = 0
        
        # Add segment at the end
        return self.add_segment_at_position(song_id, song_start, song_end, mashup_end, section, shift_timeline=False)
    def add_segment_at_position(self, song_id, song_start, song_end, mashup_position, section='body', shift_timeline=True):
        """Add segment at a specific position with proper timeline management"""
        if song_id not in self.songs:
            logger.error(f"Song {song_id} not found")
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
            # SHIFT TIMELINE MODE: Move everything after this position
            
            # Step 1: Shift all segments that start at or after this position
            for segment in self.segments:
                if segment['mashup_start'] >= mashup_position:
                    segment['mashup_start'] += duration
                    segment['mashup_end'] += duration
            
            # Step 2: Add the new segment
            self.segments.append(new_segment)
            
            # Step 3: Sort segments by start time to maintain order
            self.segments.sort(key=lambda x: x['mashup_start'])
            
        else:
            # OVERLAY MODE: Just add without shifting (background music)
            self.segments.append(new_segment)
            self.segments.sort(key=lambda x: x['mashup_start'])
        
        # Rebuild the preview
        self._rebuild_preview_mashup()
        
        logger.info(f"Added segment from {song_id} at {mashup_position/1000:.1f}s, shift_timeline={shift_timeline}")
        return new_segment['id']

    def check_timeline_shift_needed(self, mashup_position):
        """Check if there are segments after the given position"""
        segments_after = [seg for seg in self.segments if seg['mashup_start'] >= mashup_position]
        return len(segments_after) > 0, segments_after

    def replace_segment_with_options(self, segment_id, new_song_id, start_time, end_time, replacement_option='adjust_timeline'):
        """
        Replace segment with user-chosen option for length differences
        
        Args:
            segment_id: ID of segment to replace
            new_song_id: New song to use
            start_time: Start time in seconds
            end_time: End time in seconds
            replacement_option: 'adjust_timeline', 'keep_background', or 'trim_to_fit'
        """
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                if new_song_id not in self.songs:
                    logger.error(f"Song {new_song_id} not found")
                    return False, "Song not found"
                
                # Save state for undo
                self._save_state(f"Replace segment with {new_song_id} ({replacement_option})")
                
                old_duration = segment['mashup_end'] - segment['mashup_start']
                new_duration = int((end_time - start_time) * 1000)
                duration_diff = new_duration - old_duration
                
                # Update the segment
                self.segments[i].update({
                    'song_id': new_song_id,
                    'song_start': int(start_time * 1000),
                    'song_end': int(end_time * 1000),
                    'mashup_end': segment['mashup_start'] + new_duration
                })
                
                # Handle different replacement options
                if replacement_option == 'adjust_timeline':
                    # Shift all subsequent segments
                    self._adjust_subsequent_segments(i, duration_diff)
                    
                elif replacement_option == 'keep_background':
                    # Keep original timeline, excess audio plays in background
                    if duration_diff > 0:
                        # New segment is longer - it will overlay with next segments
                        self.segments[i]['overlay_mode'] = True
                    else:
                        # New segment is shorter - add silence or keep original timing
                        self.segments[i]['mashup_end'] = segment['mashup_start'] + old_duration
                        
                elif replacement_option == 'trim_to_fit':
                    # Force new segment to fit original duration
                    if duration_diff != 0:
                        # Adjust end time to match original duration
                        new_end_time = start_time + (old_duration / 1000)
                        self.segments[i]['song_end'] = int(new_end_time * 1000)
                        self.segments[i]['mashup_end'] = segment['mashup_start'] + old_duration
                
                logger.info(f"Replaced segment with {new_song_id} using {replacement_option}")
                return True, f"Segment replaced using {replacement_option} mode"
        
        return False, "Segment not found"

    def _adjust_subsequent_segments(self, segment_index, duration_diff):
        """Adjust all segments after the given index by duration_diff"""
        for i in range(segment_index + 1, len(self.segments)):
            self.segments[i]['mashup_start'] += duration_diff
            self.segments[i]['mashup_end'] += duration_diff

    def get_replacement_options(self, segment_id, new_song_id, start_time, end_time):
        """Get information about replacement options"""
        for segment in self.segments:
            if segment['id'] == segment_id:
                old_duration = (segment['mashup_end'] - segment['mashup_start']) / 1000
                new_duration = end_time - start_time
                duration_diff = new_duration - old_duration
                
                return {
                    'old_duration': old_duration,
                    'new_duration': new_duration,
                    'duration_diff': duration_diff,
                    'needs_decision': abs(duration_diff) > 0.5,  # More than 0.5 second difference
                    'options': {
                        'adjust_timeline': {
                            'description': 'Adjust entire timeline (recommended)',
                            'effect': f'Timeline will be {"extended" if duration_diff > 0 else "shortened"} by {abs(duration_diff):.1f} seconds'
                        },
                        'keep_background': {
                            'description': 'Keep original timeline',
                            'effect': 'Excess audio will play in background' if duration_diff > 0 else 'Segment will be padded with silence'
                        },
                        'trim_to_fit': {
                            'description': 'Trim to fit original duration',
                            'effect': f'New segment will be {"trimmed" if duration_diff > 0 else "extended"} to fit exactly'
                        }
                    }
                }
        
        return None

    
    def replace_segment(self, segment_id, new_song_id, start_time, end_time):
        """Replace an existing segment"""
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                if new_song_id not in self.songs:
                    logger.error(f"Song {new_song_id} not found")
                    return False
                
                # Save state for undo
                self._save_state(f"Replace segment with {new_song_id}")
                
                # Keep the same mashup position but update content
                duration = int((end_time - start_time) * 1000)
                self.segments[i].update({
                    'song_id': new_song_id,
                    'song_start': int(start_time * 1000),
                    'song_end': int(end_time * 1000),
                    'mashup_end': segment['mashup_start'] + duration
                })
                
                logger.info(f"Replaced segment with {new_song_id}")
                return True
        logger.error(f"Segment {segment_id} not found")
        return False
    
    def delete_segment(self, segment_id):
        """Delete a segment"""
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                # Save state for undo
                self._save_state(f"Delete segment from {segment['song_id']}")
                
                self.segments.pop(i)
                logger.info(f"Deleted segment {segment_id}")
                return True
        logger.error(f"Segment {segment_id} not found")
        return False
    

    def delete_segment_with_timeline_adjustment(self, segment_id, adjust_timeline=True):
        """Delete segment with option to adjust timeline"""
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                # Save state for undo
                self._save_state(f"Delete segment from {segment['song_id']}")
                
                # Get segment duration for timeline adjustment
                segment_duration = segment['mashup_end'] - segment['mashup_start']
                segment_start = segment['mashup_start']
                
                # Remove the segment
                self.segments.pop(i)
                
                # Adjust subsequent segments if requested
                if adjust_timeline:
                    self._shift_segments_after_position(segment_start, -segment_duration)
                
                logger.info(f"Deleted segment {segment_id} with timeline adjustment: {adjust_timeline}")
                return True, segment_duration
        
        return False, 0

    

    
    def check_timeline_shift_needed(self, mashup_position):
        """Check if timeline shift is needed when adding at position"""
        segments_after = [seg for seg in self.segments if seg['mashup_start'] >= mashup_position]
        return len(segments_after) > 0, segments_after

    def _shift_segments_after_position(self, position, time_shift):
        """Shift all segments after a given position by time_shift amount without creating gaps"""
        # Get all segments that start at or after the position, sorted by start time
        segments_to_shift = sorted(
            [seg for seg in self.segments if seg['mashup_start'] >= position], 
            key=lambda x: x['mashup_start']
        )
        
        # Shift each segment by the time_shift amount
        for segment in segments_to_shift:
            segment['mashup_start'] += time_shift
            segment['mashup_end'] += time_shift
        
        logger.info(f"Shifted {len(segments_to_shift)} segments by {time_shift}ms")
    def _rebuild_preview_mashup(self):
        """Rebuild the preview mashup based on current segments"""
        if not self.segments:
            self.preview_mashup = AudioSegment.silent(duration=10000)
            return
        
        try:
            # Find the maximum end time
            max_end = max([seg['mashup_end'] for seg in self.segments])
            
            # Create silent base track
            self.preview_mashup = AudioSegment.silent(duration=max_end)
            
            # Sort segments by mashup start time
            sorted_segments = sorted(self.segments, key=lambda x: x['mashup_start'])
            
            for i, segment in enumerate(sorted_segments):
                if segment['song_id'] not in self.songs:
                    logger.warning(f"Song {segment['song_id']} not found in loaded songs")
                    continue
                
                try:
                    # Extract audio from source song
                    song_audio = self.songs[segment['song_id']]['audio'][segment['song_start']:segment['song_end']]
                    
                    if len(song_audio) == 0:
                        logger.warning(f"Empty audio segment for {segment['song_id']}")
                        continue
                    
                    # Apply fade transitions
                    fade_duration = min(500, len(song_audio) // 4)  # Max 0.5s or 25% of segment
                    
                    if fade_duration > 0:
                        if i > 0:
                            song_audio = song_audio.fade_in(fade_duration)
                        
                        if i < len(sorted_segments) - 1:
                            song_audio = song_audio.fade_out(fade_duration)
                    
                    # Check for overlaps and adjust volume
                    overlapping_segments = [
                        seg for seg in sorted_segments 
                        if seg != segment and (
                            (segment['mashup_start'] < seg['mashup_end'] and 
                            segment['mashup_end'] > seg['mashup_start'])
                        )
                    ]
                    
                    if overlapping_segments:
                        # Reduce volume for overlapping segments (background music)
                        song_audio = song_audio - 14  # Reduce by ~14dB (≈20% volume)
                    
                    # Ensure mashup is long enough
                    required_length = segment['mashup_end']
                    if len(self.preview_mashup) < required_length:
                        extension = AudioSegment.silent(duration=required_length - len(self.preview_mashup))
                        self.preview_mashup = self.preview_mashup + extension
                    
                    # Overlay the segment
                    self.preview_mashup = self.preview_mashup.overlay(song_audio, position=segment['mashup_start'])
                    
                except Exception as e:
                    logger.error(f"Error processing segment {segment['id']}: {e}")
                    continue
            
            logger.info("Preview mashup rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding preview mashup: {e}")
            self.preview_mashup = AudioSegment.silent(duration=10000)


    def get_timeline_positions(self):
        """Get available positions for inserting segments"""
        positions = [0]  # Start of timeline
        
        for segment in sorted(self.segments, key=lambda x: x['mashup_start']):
            positions.append(segment['mashup_start'])  # Before this segment
            positions.append(segment['mashup_end'])    # After this segment
        
        # Remove duplicates and sort
        positions = sorted(list(set(positions)))
        
        return [{
            'position_ms': pos,
            'position_seconds': pos / 1000,
            'description': self._get_position_description(pos)
        } for pos in positions]

    def _get_position_description(self, position_ms):
        """Get human-readable description of timeline position"""
        if position_ms == 0:
            return "Beginning of mashup"
        
        # Find what's at this position
        for segment in self.segments:
            if segment['mashup_start'] == position_ms:
                return f"Before '{segment['song_id']}'"
            elif segment['mashup_end'] == position_ms:
                return f"After '{segment['song_id']}'"
        
        return f"At {position_ms/1000:.1f}s"

    def get_segment_preview(self, song_id, start_time, end_time):
        """Create a preview file for a segment"""
        if song_id not in self.songs:
            logger.error(f"Song {song_id} not found")
            return None
        
        try:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            segment_audio = self.songs[song_id]['audio'][start_ms:end_ms]
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            segment_audio.export(temp_file.name, format="wav")
            self.temp_files.append(temp_file.name)
            
            logger.info(f"Created segment preview for {song_id}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error creating segment preview: {e}")
            return None
    
    def generate_mashup_preview(self):
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
                        logger.warning(f"Empty audio segment for {segment['song_id']}")
                        continue
                    
                    # Apply fade transitions
                    fade_duration = min(500, len(segment_audio) // 4)  # Max 0.5s or 25% of segment
                    
                    if fade_duration > 0:
                        if i > 0:
                            segment_audio = segment_audio.fade_in(fade_duration)
                        
                        if i < len(sorted_segments) - 1:
                            segment_audio = segment_audio.fade_out(fade_duration)
                    
                    # Check for overlaps and adjust volume
                    overlapping_segments = [
                        seg for seg in sorted_segments 
                        if seg != segment and (
                            (segment['mashup_start'] < seg['mashup_end'] and 
                            segment['mashup_end'] > seg['mashup_start'])
                        )
                    ]
                    
                    if overlapping_segments:
                        # Reduce volume for overlapping segments (background music)
                        segment_audio = segment_audio - 14  # Reduce by ~14dB (≈20% volume)
                    
                    # Ensure mashup is long enough
                    required_length = segment['mashup_end']
                    if len(mashup) < required_length:
                        extension = AudioSegment.silent(duration=required_length - len(mashup))
                        mashup = mashup + extension
                    
                    # Overlay the segment
                    mashup = mashup.overlay(segment_audio, position=segment['mashup_start'])
                    
                else:
                    logger.warning(f"Song {segment['song_id']} not found for segment")
            
            # Export to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            mashup.export(temp_file.name, format="wav")
            self.temp_files.append(temp_file.name)
            
            logger.info("Generated mashup preview")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to generate mashup preview: {e}")
            return None

    def move_segment_with_overlap_handling(self, segment_id, new_position, overlap_option='shift_timeline'):
        """Move segment with overlap handling"""
        # Find segment
        segment = None
        segment_index = None
        for i, seg in enumerate(self.segments):
            if seg['id'] == segment_id:
                segment = seg
                segment_index = i
                break
        
        if segment is None:
            logger.error(f"Segment {segment_id} not found")
            return False, "Segment not found"
        
        # Save state for undo
        self._save_state(f"Move segment {segment_id} with {overlap_option}")
        
        # Calculate new end position
        duration = segment['mashup_end'] - segment['mashup_start']
        new_end = new_position + duration
        
        # Check for overlaps
        overlapping_segments = []
        for seg in self.segments:
            if (seg['id'] != segment_id and 
                new_position < seg['mashup_end'] and 
                new_end > seg['mashup_start']):
                overlapping_segments.append(seg)
        
        # Handle overlap based on option
        if overlapping_segments:
            if overlap_option == 'background':
                # Keep timeline, play overlapping parts at 20% volume
                segment['mashup_start'] = new_position
                segment['mashup_end'] = new_end
                segment['overlay_mode'] = True
                self._rebuild_preview_mashup()
                return True, "Moved with background overlap"
                
            elif overlap_option == 'shift_timeline':
                # Shift subsequent segments
                segment['mashup_start'] = new_position
                segment['mashup_end'] = new_end
                self._shift_segments_after_position(new_end, duration)
                self._rebuild_preview_mashup()
                return True, "Moved with timeline shift"
                
            elif overlap_option == 'replace':
                # Replace overlapping segments
                for overlap in overlapping_segments:
                    self.delete_segment_by_id(overlap['id'])
                segment['mashup_start'] = new_position
                segment['mashup_end'] = new_end
                self._rebuild_preview_mashup()
                return True, "Moved with replacement of overlapping segments"
            else:
                return False, "Invalid overlap option"
        else:
            # No overlap, just move
            segment['mashup_start'] = new_position
            segment['mashup_end'] = new_end
            self._rebuild_preview_mashup()
            return True, "Moved without overlap"

    def delete_segment_by_id(self, segment_id):
        """Delete segment by ID"""
        for i, segment in enumerate(self.segments):
            if segment['id'] == segment_id:
                self.segments.pop(i)
                return True
        return False

    def export_mashup(self, output_path):
        """Export the final mashup"""
        preview_file = self.generate_mashup_preview()
        if preview_file:
            try:
                # Copy to final location
                mashup = AudioSegment.from_wav(preview_file)
                mashup.export(output_path, format="mp3")
                logger.info(f"Exported mashup to {output_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to export mashup: {e}")
                return False
        return False
    
    def _save_state(self, action):
        """Save current state for undo with proper deep copy"""
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

    def undo(self):
        """Undo last action - only one step back"""
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.segments = copy.deepcopy(state['segments'])
            self._rebuild_preview_mashup()
            logger.info(f"Undid action: {state['action']}")
            return True, state['action']
        return False, "Nothing to undo"

    def redo(self):
        """Redo last undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.segments = copy.deepcopy(state['segments'])
            self._rebuild_preview_mashup()
            logger.info(f"Redid action: {state['action']}")
            return True, state['action']
        return False, "Nothing to redo"

    def get_total_duration(self):
        """Get total mashup duration"""
        if not self.segments:
            return 0
        return max([seg['mashup_end'] for seg in self.segments])
    
    def get_song_duration(self, song_id):
        """Get duration of a specific song"""
        if song_id in self.songs:
            return self.songs[song_id]['duration']
        return 0
    
    def validate_segment(self, song_id, start_time, end_time):
        """Validate segment parameters"""
        if song_id not in self.songs:
            return False, f"Song {song_id} not found"
        
        if start_time < 0:
            return False, "Start time cannot be negative"
        
        if end_time <= start_time:
            return False, "End time must be greater than start time"
        
        song_duration = self.songs[song_id]['duration'] / 1000
        if start_time >= song_duration:
            return False, f"Start time exceeds song duration ({song_duration:.1f}s)"
        
        if end_time > song_duration:
            return False, f"End time exceeds song duration ({song_duration:.1f}s)"
        
        return True, "Valid"
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
                logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
        self.temp_files = []
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

# Flask integration wrapper
class FlaskMashupEditor(MashupEditor):
    """Flask-specific wrapper for the mashup editor"""
    
    def __init__(self, mashup_file=None, metadata_file=None, audio_folder='downloaded_music'):
        """Initialize with Flask-specific parameters"""
        mashup_folder = os.path.dirname(mashup_file) if mashup_file else 'Final_Mashup'
        super().__init__(music_folder=audio_folder, mashup_folder=mashup_folder)
        
        self.mashup_file = mashup_file
        self.metadata_file = metadata_file
        self.modified = False
    
    def get_editor_data(self):
        """Get editor data in format expected by Flask templates"""
        segments_data = self.get_segments_data()
        songs_data = self.get_songs_data()
        
        return {
            'segments': segments_data,
            'original_songs': list(songs_data.keys()),
            'total_duration': self.get_total_duration() / 1000,  # Convert to seconds
            'session_id': str(uuid.uuid4())
        }
    
    def update_segment(self, segment_index, new_start, new_end, new_position, song_name=None):
        """Update segment with Flask-compatible interface"""
        if segment_index >= len(self.segments):
            return False
        
        segment = self.segments[segment_index]
        segment_id = segment['id']
        
        # Convert seconds to milliseconds if needed
        if new_start < 1000:  # Assume seconds if less than 1000
            new_start *= 1000
        if new_end < 1000:
            new_end *= 1000
        if new_position < 1000:
            new_position *= 1000
        
        # Update segment
        if song_name and song_name in self.songs:
            segment['song_id'] = song_name
        
        segment['song_start'] = new_start
        segment['song_end'] = new_end
        segment['mashup_start'] = new_position
        segment['mashup_end'] = new_position + (new_end - new_start)
        
        self.modified = True
        return True
    
    def rebuild_mashup(self):
        """Rebuild mashup for Flask interface"""
        try:
            # This would trigger a rebuild of the preview
            preview_file = self.generate_mashup_preview()
            self.modified = True
            return preview_file is not None
        except Exception as e:
            logger.error(f"Error rebuilding mashup: {e}")
            return False
    
    def export_edited_mashup(self, output_path):
        """Export edited mashup for Flask interface"""
        try:
            return self.export_mashup(output_path)
        except Exception as e:
            logger.error(f"Error exporting edited mashup: {e}")
            return False

# Test function
def test_mashup_editor():
    """Test the mashup editor functionality"""
    print("Testing Mashup Editor...")
    
    try:
        editor = MashupEditor()
        print(f"✅ Editor initialized with {len(editor.songs)} songs")
        
        songs_data = editor.get_songs_data()
        segments_data = editor.get_segments_data()
        
        print(f"✅ Songs data: {list(songs_data.keys())}")
        print(f"✅ Segments: {len(segments_data)}")
        
        # Test adding a segment if songs are available
        if songs_data:
            song_id = list(songs_data.keys())[0]
            segment_id = editor.add_segment(song_id, 0, 10)
            if segment_id:
                print(f"✅ Added test segment: {segment_id}")
                
                # Test undo
                success, message = editor.undo()
                print(f"✅ Undo: {message}")
        
        editor.cleanup()
        print("✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_mashup_editor()
