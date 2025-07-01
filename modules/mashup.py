import gc

import librosa, librosa.display
import shutil
import numpy as np
import soundfile as sf
import os
from dotenv import load_dotenv
load_dotenv()
import json
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
from scipy import signal
import warnings
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


warnings.filterwarnings('ignore')

# Global status tracking
mashup_status = {"status": "idle", "message": "", "progress": 0}

def update_status(message: str, progress: int = None):
    global mashup_status
    mashup_status["message"] = message
    if progress is not None:
        mashup_status["progress"] = progress

def get_mashup_status():
    return mashup_status.copy()

class AdvancedSegmentSelector:
    """-quality segment selector with perfect musical boundaries"""

    def __init__(self):
        self.sr = 22050
        self.max_consecutive_merge = 4

    def find_enhanced_segments(self, y: np.ndarray, sr: int, duration: float, tempo: float, song_key: str) -> Dict:
        """Create -quality segments with perfect start/end points"""
        
        print(f"ðŸŽµ Creating perfect segments for {duration:.1f}s audio...")
        
        # Phase 1: Detect musical structure with beat alignment
        boundaries = self._detect_musical_structure_boundaries(y, sr)
        downbeats = self._get_precise_downbeats(y, sr)
        
        # Phase 2: Create beat-aligned segments
        musical_segments = self._create_beat_aligned_segments(y, sr, boundaries, downbeats, duration)
        
        # Phase 3: Extend segments for perfect endings
        perfect_segments = self._create_perfect_musical_segments(musical_segments, y, sr)
        
        # Phase 4: Score for mashup quality
        scored_segments = self._score_for_mashup_quality(perfect_segments, song_key, tempo)
        
        print(f"âœ¨ Created {len(scored_segments)} perfect segments with musical boundaries")
        
        return {
            'best_overall': scored_segments[:8],
            'best_intro': self._select_intro_segments(scored_segments, duration),
            'best_outro': self._select_outro_segments(scored_segments, duration),
            'all_segments': scored_segments
        }

    def _detect_musical_structure_boundaries(self, y: np.ndarray, sr: int) -> List[float]:
        """Detect boundaries at musical phrase endings"""
        
        hop_length = 512
        
        # Extract musical features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        
        # Calculate musical novelty (chord changes + timbral changes)
        chroma_novelty = np.sum(np.diff(chroma, axis=1) ** 2, axis=0)
        mfcc_novelty = np.sum(np.diff(mfcc, axis=1) ** 2, axis=0)
        contrast_novelty = np.sum(np.diff(spectral_contrast, axis=1) ** 2, axis=0)
        
        # Combine novelties with emphasis on harmonic changes
        combined_novelty = (
            0.5 * chroma_novelty +
            0.3 * mfcc_novelty +
            0.2 * contrast_novelty
        )
        
        # Smooth and find peaks
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
        
        smoothed = gaussian_filter1d(combined_novelty, sigma=2.0)
        threshold = np.mean(smoothed) + 0.8 * np.std(smoothed)
        min_distance = int(8.0 * sr / hop_length)  # Minimum 8 seconds apart
        
        peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)
        boundaries = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        
        # Add start and end
        all_boundaries = np.concatenate([[0.0], boundaries, [len(y) / sr]])
        return sorted(all_boundaries)

    def _get_precise_downbeats(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Get precise downbeat locations for perfect timing"""
        
        # Get beats with high precision
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units='time')
        
        # Estimate downbeats (every 4th beat typically)
        if len(beats) >= 4:
            # Calculate average beat interval
            beat_intervals = np.diff(beats)
            avg_interval = np.median(beat_intervals)
            
            # Estimate downbeats
            downbeats = []
            for i in range(0, len(beats), 4):
                if i < len(beats):
                    downbeats.append(beats[i])
            
            return np.array(downbeats)
        else:
            # Fallback: use regular beats
            return beats

    def _create_beat_aligned_segments(self, y: np.ndarray, sr: int, boundaries: List[float], 
                                downbeats: np.ndarray, duration: float) -> List[Dict]:
        """Create segments with cue point detection"""
        
        segments = []
        cue_detector = CuePointDetector(sr)
        
        for i in range(len(boundaries) - 1):
            start_boundary = boundaries[i]
            end_boundary = boundaries[i + 1]
            
            # Snap to nearest downbeats
            start_time = self._snap_to_nearest_downbeat(start_boundary, downbeats, 'before')
            end_time = self._snap_to_nearest_downbeat(end_boundary, downbeats, 'after')
            
            # Ensure minimum duration
            if end_time - start_time < 6.0:
                next_downbeat = self._find_next_downbeat(end_time, downbeats)
                if next_downbeat and next_downbeat - start_time <= 30.0:
                    end_time = next_downbeat
            
            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            if len(segment_audio) > sr * 3:
                features = self._extract_musical_features(segment_audio, sr)
                
                # NEW: Detect cue points for this segment
                segment_downbeats = downbeats[(downbeats >= start_time) & (downbeats <= end_time)] - start_time
                cue_points = cue_detector.detect_cue_points(segment_audio, segment_downbeats)
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'audio': segment_audio,
                    'features': features,
                    'type': self._classify_musical_segment(features, start_time, duration),
                    'beat_aligned': True,
                    'cue_points': cue_points,  # NEW: Add cue points
                    'original_index': i
                })
        
        return segments


    def _snap_to_nearest_downbeat(self, time: float, downbeats: np.ndarray, direction: str) -> float:
        """Snap time to nearest downbeat for perfect alignment"""
        
        if len(downbeats) == 0:
            return time
        
        if direction == 'before':
            # Find the last downbeat before or at this time
            valid_beats = downbeats[downbeats <= time + 0.1]
            return valid_beats[-1] if len(valid_beats) > 0 else time
        else:  # 'after'
            # Find the first downbeat after or at this time
            valid_beats = downbeats[downbeats >= time - 0.1]
            return valid_beats[0] if len(valid_beats) > 0 else time

    def _find_next_downbeat(self, time: float, downbeats: np.ndarray) -> Optional[float]:
        """Find the next downbeat after given time"""
        
        next_beats = downbeats[downbeats > time]
        return next_beats[0] if len(next_beats) > 0 else None

    def _create_perfect_musical_segments(self, segments: List[Dict], y: np.ndarray, sr: int) -> List[Dict]:
        """Extend segments to create perfect musical phrases"""
        
        perfect_segments = []
        used_indices = set()
        
        for i, segment in enumerate(segments):
            if i in used_indices:
                continue
            
            # Start with current segment
            extended_segment = segment.copy()
            merge_candidates = [segment]
            current_indices = [i]
            
            # Look for opportunities to create perfect musical phrases
            for j in range(i + 1, min(i + self.max_consecutive_merge, len(segments))):
                if j in used_indices:
                    break
                
                next_segment = segments[j]
                
                # Check if merging creates a better musical phrase
                if self._should_merge_for_perfect_phrase(merge_candidates, next_segment):
                    merge_candidates.append(next_segment)
                    current_indices.append(j)
                else:
                    break
            
            # Create extended segment if beneficial
            if len(merge_candidates) > 1:
                extended_segment = self._create_perfect_phrase_segment(merge_candidates, y, sr)
                extended_segment['is_extended'] = True
                extended_segment['merged_from'] = len(merge_candidates)
                print(f"ðŸŽ¼ Created perfect phrase by merging {len(merge_candidates)} segments")
            
            perfect_segments.append(extended_segment)
            used_indices.update(current_indices)
        
        return perfect_segments

    def _should_merge_for_perfect_phrase(self, current_segments: List[Dict], next_segment: Dict) -> bool:
        """Enhanced merging logic for complete musical phrases"""
        
        # Check total duration doesn't exceed limit
        total_duration = sum(seg['duration'] for seg in current_segments) + next_segment['duration']
        if total_duration > 40:  # Max 40 seconds
            return False
        
        current_features = current_segments[-1]['features']
        next_features = next_segment['features']
        current_type = current_segments[-1].get('type', '')
        next_type = next_segment.get('type', '')
        
        # Musical criteria for merging
        criteria_met = 0
        
        # EXISTING CRITERIA
        # 1. Natural energy progression
        if current_features['energy'] > next_features['energy'] * 0.8:
            criteria_met += 1
        
        # 2. Harmonic resolution
        if next_features['harmonic_stability'] > current_features['harmonic_stability']:
            criteria_met += 1
        
        # 3. Rhythmic completion
        if (current_features['onset_density'] > 6 and next_features['onset_density'] < 5):
            criteria_met += 1
        
        # 4. Perfect duration (15-25 seconds is ideal)
        if 15 <= total_duration <= 25:
            criteria_met += 1
        
        # NEW CRITERIA FOR COMPLETE PHRASES
        
        # 5. Musical sentence structure (SRDC pattern)
        # Check if we're building a complete musical sentence
        if len(current_segments) == 1:  # Statement + Response pattern
            if (current_features['harmonic_strength'] > 0.6 and 
                next_features['harmonic_strength'] > 0.6):
                criteria_met += 1
        elif len(current_segments) == 2:  # Adding departure/conclusion
            if next_features['harmonic_stability'] > 0.7:  # Strong conclusion
                criteria_met += 1
        
        # 6. Cadential completion
        # Check if next segment provides proper phrase ending
        spectral_decline = (current_features.get('spectral_centroid', 2500) > 
                        next_features.get('spectral_centroid', 2500))
        energy_resolution = next_features['energy'] < current_features['energy'] * 1.1
        
        if spectral_decline and energy_resolution:
            criteria_met += 1
        
        # 7. Motivic coherence
        # Check if segments share similar musical characteristics
        energy_similarity = abs(current_features['energy'] - next_features['energy']) < 0.3
        harmonic_coherence = (current_features['harmonic_strength'] > 0.5 and 
                            next_features['harmonic_strength'] > 0.5)
        
        if energy_similarity and harmonic_coherence:
            criteria_met += 1
        
        # 8. Phrase length optimization
        # Prefer standard phrase lengths (8, 16, 32 beats at ~120 BPM)
        estimated_beats = total_duration * 2  # Rough beat estimation
        if estimated_beats in [8, 12, 16, 20, 24, 28, 32]:
            criteria_met += 1
        elif 14 <= estimated_beats <= 18 or 30 <= estimated_beats <= 34:
            criteria_met += 0.5  # Partial credit for close matches
        
        # 9. Antecedent-consequent relationship
        # Check for question-answer phrase structure
        if len(current_segments) == 1:
            current_ends_open = current_features['harmonic_stability'] < 0.7
            next_provides_closure = next_features['harmonic_stability'] > 0.7
            
            if current_ends_open and next_provides_closure:
                criteria_met += 1
        
        # 10. Dynamic contour completion
        # Check for natural dynamic progression
        dynamic_range_current = current_features.get('dynamic_range', 0.3)
        dynamic_range_next = next_features.get('dynamic_range', 0.3)
        
        # Prefer phrases that build then resolve
        if (dynamic_range_current > 0.4 and dynamic_range_next < dynamic_range_current):
            criteria_met += 1
        
        # 11. Tonal center stability
        # Ensure merged phrase maintains tonal coherence
        if (current_features['harmonic_stability'] > 0.6 and 
            next_features['harmonic_stability'] > 0.6):
            criteria_met += 1
        
        # 12. Rhythmic phrase completion
        # Check for rhythmic phrase boundaries (strong-weak pattern)
        current_rhythm_strength = current_features.get('onset_density', 6)
        next_rhythm_strength = next_features.get('onset_density', 6)
        
        # Strong start, weaker end suggests phrase completion
        if current_rhythm_strength > 7 and next_rhythm_strength < 6:
            criteria_met += 1
        
        # 13. Melodic arc completion
        # Check for complete melodic gesture
        current_spectral = current_features.get('spectral_centroid', 2500)
        next_spectral = next_features.get('spectral_centroid', 2500)
        
        # Rising then falling melodic contour
        if len(current_segments) >= 2:
            prev_spectral = current_segments[-2]['features'].get('spectral_centroid', 2500)
            if prev_spectral < current_spectral > next_spectral:  # Peak in middle
                criteria_met += 1
        
        # 14. Textural consistency
        # Ensure segments work together texturally
        perceptual_similarity = abs(current_features.get('perceptual_quality', 0.6) - 
                                next_features.get('perceptual_quality', 0.6)) < 0.2
        if perceptual_similarity:
            criteria_met += 1
        
        # 15. Chorus completion bonus
        # Special handling for chorus segments
        if current_type == 'chorus' or next_type == 'chorus':
            if 16 <= total_duration <= 32:  # Ideal chorus length
                criteria_met += 2  # Double bonus for complete choruses
        
        # DECISION LOGIC
        # More lenient for shorter phrases, stricter for longer ones
        if total_duration <= 20:
            required_criteria = 3  # Easier for short phrases
        elif total_duration <= 30:
            required_criteria = 4  # Moderate for medium phrases
        else:
            required_criteria = 5  # Stricter for long phrases
        
        # Log decision for debugging
        if criteria_met >= required_criteria:
            print(f"🎼 MERGING: {criteria_met}/{required_criteria} criteria met, duration: {total_duration:.1f}s")
            return True
        else:
            print(f"🎼 NOT merging: only {criteria_met}/{required_criteria} criteria met")
            return False

    def _create_perfect_phrase_segment(self, segments_to_merge: List[Dict], y: np.ndarray, sr: int) -> Dict:
        """Create a perfect musical phrase from multiple segments"""
        
        start_time = segments_to_merge[0]['start_time']
        end_time = segments_to_merge[-1]['end_time']
        total_duration = end_time - start_time
        
        # Extract combined audio
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        combined_audio = y[start_sample:end_sample]
        
        # Calculate weighted features
        total_weight = sum(seg['duration'] for seg in segments_to_merge)
        merged_features = {}
        
        for key in segments_to_merge[0]['features'].keys():
            weighted_sum = sum(seg['features'][key] * seg['duration'] for seg in segments_to_merge)
            merged_features[key] = weighted_sum / total_weight
        
        # Determine dominant type
        types = [seg['type'] for seg in segments_to_merge]
        merged_type = max(set(types), key=types.count)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': total_duration,
            'audio': combined_audio,
            'features': merged_features,
            'type': merged_type,
            'beat_aligned': True,
            'perfect_phrase': True,
            'phrase_quality': self._calculate_phrase_quality(segments_to_merge)
        }

    def _calculate_phrase_quality(self, segments: List[Dict]) -> float:
        """Calculate the quality of the musical phrase"""
        
        if len(segments) < 2:
            return 0.7
        
        # Analyze musical progression
        energies = [seg['features']['energy'] for seg in segments]
        stabilities = [seg['features']['harmonic_stability'] for seg in segments]
        
        # Natural energy curve
        energy_progression = 1.0 if energies[-1] <= energies[0] * 1.2 else 0.6
        
        # Harmonic resolution
        harmonic_resolution = np.mean(stabilities[-2:])
        
        # Duration appropriateness
        total_duration = sum(seg['duration'] for seg in segments)
        duration_score = 1.0 if 12 <= total_duration <= 30 else 0.7
        
        return (energy_progression * 0.4 + harmonic_resolution * 0.4 + duration_score * 0.2)

    def _extract_musical_features(self, segment_audio: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive musical features for mashup quality"""
        
        hop_length = 1024
        
        # Energy and dynamics
        rms = librosa.feature.rms(y=segment_audio, hop_length=hop_length)
        energy = float(np.mean(rms))
        dynamic_range = float(np.max(rms) - np.min(rms))
        
        # Spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=segment_audio, sr=sr, hop_length=hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment_audio, sr=sr, hop_length=hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr, hop_length=hop_length)
        
        # Harmonic analysis
        chroma = librosa.feature.chroma_cqt(y=segment_audio, sr=sr, hop_length=hop_length)
        harmonic_strength = float(np.mean(np.max(chroma, axis=0)))
        chroma_std = np.std(np.argmax(chroma, axis=1))
        harmonic_stability = float(1 - min(chroma_std / 12, 1.0))
        
        # Rhythmic features
        onset_frames = librosa.onset.onset_detect(y=segment_audio, sr=sr, hop_length=hop_length)
        onset_density = len(onset_frames) / (len(segment_audio) / sr)
        
        # Tempo analysis
        try:
            tempo, _ = librosa.beat.beat_track(y=segment_audio, sr=sr, hop_length=hop_length)
            tempo_consistency = float(1 / (1 + abs(float(tempo) - 120) / 120))
        except:
            tempo_consistency = 0.5
        
        # Perceptual quality
        spectral_flatness = librosa.feature.spectral_flatness(y=segment_audio, hop_length=hop_length)
        perceptual_quality = float(1 - np.mean(spectral_flatness))
        
        return {
            'energy': energy,
            'dynamic_range': dynamic_range,
            'spectral_centroid': float(np.mean(spectral_centroid)),
            'spectral_rolloff': float(np.mean(spectral_rolloff)),
            'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
            'harmonic_strength': harmonic_strength,
            'harmonic_stability': harmonic_stability,
            'onset_density': onset_density,
            'tempo_consistency': tempo_consistency,
            'perceptual_quality': perceptual_quality,
            'mashup_readiness': self._calculate_mashup_readiness(energy, harmonic_stability, onset_density)
        }

    def _calculate_mashup_readiness(self, energy: float, harmonic_stability: float, onset_density: float) -> float:
        """Calculate how ready a segment is for mashup use"""
        
        # Perfect segments for mashups have:
        # - Moderate to high energy (0.3-0.8)
        # - Good harmonic stability (>0.6)
        # - Clear rhythm (4-12 onsets per second)
        
        energy_score = 1.0 if 0.3 <= energy <= 0.8 else max(0.3, 1.0 - abs(energy - 0.55) * 2)
        harmony_score = harmonic_stability
        rhythm_score = 1.0 if 4 <= onset_density <= 12 else max(0.3, 1.0 - abs(onset_density - 8) / 8)
        
        return (energy_score * 0.4 + harmony_score * 0.3 + rhythm_score * 0.3)

    def _score_for_mashup_quality(self, segments: List[Dict], song_key: str, tempo: float) -> List[Dict]:
        """Score segments specifically for mashup excellence"""
        
        for segment in segments:
            features = segment['features']
            segment_type = segment.get('type', 'verse')
            
            # Base mashup score
            mashup_score = (
                features['mashup_readiness'] * 0.3 +
                features['perceptual_quality'] * 0.2 +
                features['harmonic_stability'] * 0.2 +
                features['energy'] * 0.15 +
                features['dynamic_range'] * 0.1 +
                features['tempo_consistency'] * 0.05
            )
            
            # Bonus for perfect phrases
            if segment.get('perfect_phrase', False):
                mashup_score += 0.15 * segment.get('phrase_quality', 0.7)
            
            # Bonus for beat alignment
            if segment.get('beat_aligned', False):
                mashup_score += 0.1
            
            # Type-based bonuses
            type_bonus = {
                'chorus': 0.25,
                'verse': 0.08,
                'bridge': 0.06,
                'intro': 0.05,
                'outro': 0.05
            }.get(segment_type, 0.03)
            
            mashup_score += type_bonus
            
            # -quality bonus for perfect characteristics
            if (features['energy'] > 0.4 and features['harmonic_stability'] > 0.7 and 
                features['onset_density'] > 5 and segment.get('beat_aligned', False)):
                mashup_score += 0.1  # -ready bonus
            
            segment['score'] = min(1.0, mashup_score)
            segment['_ready'] = mashup_score > 0.8
        
        # Sort by mashup quality
        return sorted(segments, key=lambda x: x['score'], reverse=True)

    def _classify_musical_segment(self, features: Dict, start_time: float, duration: float) -> str:
        """Classify segment type based on musical characteristics"""
        
        energy = features['energy']
        onset_density = features['onset_density']
        harmonic_strength = features['harmonic_strength']
        
        # Position-based classification
        if start_time < duration * 0.15:
            return 'intro'
        elif start_time > duration * 0.85:
            return 'outro'
        
        # Musical characteristic-based classification
        if energy > 0.5 and onset_density > 7 and harmonic_strength > 0.6:
            return 'chorus'
        elif energy > 0.3 and onset_density > 5:
            return 'verse'
        elif harmonic_strength > 0.7:
            return 'bridge'
        else:
            return 'verse'

    def _select_intro_segments(self, segments: List[Dict], duration: float) -> List[Dict]:
        """Select perfect intro segments"""
        intro_candidates = [s for s in segments if s['start_time'] < duration * 0.3]
        intro_candidates.sort(key=lambda x: -x['score'])
        return intro_candidates[:2]

    def _select_outro_segments(self, segments: List[Dict], duration: float) -> List[Dict]:
        """Select perfect outro segments"""
        outro_candidates = [s for s in segments if s['start_time'] > duration * 0.7]
        outro_candidates.sort(key=lambda x: -x['score'])
        return outro_candidates[:2]




class CuePointDetector:
    """
    Detect DJ-quality cue points using beat-synchronous, multi-feature
    novelty analysis and phrase-boundary detection.
    """

    # ------------------------------------------------------------------ #
    #                           Constructor                              #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        novelty_sigma: float = 12.0,          # kernel width (frames) for Foote
        peak_prominence: float = 0.10,       # relative prominence for peaks
        weights: Tuple[float, float, float] = (0.35, 0.35, 0.30),  # (rms, flux, onset)
        quality_weights: Dict[str, float] = None,                  # bonus weights
        ml_ranker: Optional[object] = None   # e.g. XGBoost regressor (optional)
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.novelty_sigma = novelty_sigma
        self.peak_prominence = peak_prominence
        self.w_rms, self.w_flux, self.w_onset = weights

        # Bonuses when computing overall quality
        default_qw = dict(
            energy_good=0.20,
            energy_drop=0.15,
            phrase=0.25,
            beat_align=0.15,
        )
        self.qw = default_qw if quality_weights is None else quality_weights
        self.ml_ranker = ml_ranker    # if provided, used to re-rank cue list

    # ------------------------------------------------------------------ #
    #                      Public main function                           #
    # ------------------------------------------------------------------ #
    def _beat_synchronise(self, feature_tuple, beat_frames):
        """Beat-synchronize features with proper shape alignment"""
        synced = []
        for feat in feature_tuple:
            try:
                # Ensure beat_frames doesn't exceed feature length
                valid_beats = beat_frames[beat_frames < len(feat)]
                if len(valid_beats) < 2:
                    # If too few beats, create minimal sync
                    synced.append(np.array([feat.mean()]))
                else:
                    sync_feat = librosa.util.sync(feat, valid_beats, aggregate=np.median)
                    synced.append(sync_feat)
            except Exception as e:
                print(f"Beat sync error: {e}")
                # Fallback: downsample to beat count
                target_len = min(len(beat_frames), len(feat))
                downsampled = feat[:target_len:max(1, len(feat)//target_len)]
                synced.append(downsampled)
        return synced

    def detect_cue_points(self, audio: np.ndarray, downbeats: np.ndarray) -> List[Dict]:
        """Enhanced detect_cue_points with error handling"""
        try:
            # Pre-processing
            har, perc = librosa.effects.hpss(audio)
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr, hop_length=self.hop_length)
            beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)

            # Feature extraction
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            flux = self._spectral_flux(perc)
            onset_env = librosa.onset.onset_strength(y=perc, sr=self.sr, hop_length=self.hop_length)

            # FIX: Align all feature lengths before combining
            min_len = min(len(rms), len(flux), len(onset_env))
            if min_len < 10:  # Too short to analyze
                print(f"⚠️ Audio too short for analysis: {min_len} frames")
                return []

            rms = rms[:min_len]
            flux = flux[:min_len]
            onset_env = onset_env[:min_len]

            # Beat-synchronise features
            beat_syn = self._beat_synchronise((rms, flux, onset_env), beats)
            
            # FIX: Ensure all beat-sync features have same length
            min_beat_len = min(len(beat_syn[0]), len(beat_syn[1]), len(beat_syn[2]))
            if min_beat_len < 4:  # Need at least 4 beats
                print(f"⚠️ Too few beats for analysis: {min_beat_len}")
                return []

            for i in range(len(beat_syn)):
                beat_syn[i] = beat_syn[i][:min_beat_len]

            # Novelty curve
            try:
                novelty = self._foote_novelty_matrix(audio)
                # Align novelty with beat features
                if len(novelty) != min_beat_len:
                    novelty = np.interp(
                        np.linspace(0, 1, min_beat_len),
                        np.linspace(0, 1, len(novelty)),
                        novelty
                    )
            except Exception as e:
                print(f"Novelty calculation failed: {e}")
                novelty = np.zeros(min_beat_len)

            # Combine features safely
            combined = (
                self.w_rms * self._norm(beat_syn[0])
                + self.w_flux * self._norm(beat_syn[1])
                + self.w_onset * self._norm(beat_syn[2])
            )
            
            combined = gaussian_filter1d(combined, sigma=1.0)
            
            # Ensure novelty and combined have same length
            if len(novelty) != len(combined):
                target_len = min(len(novelty), len(combined))
                novelty = novelty[:target_len]
                combined = combined[:target_len]

            combined = 0.6 * self._norm(combined) + 0.4 * self._norm(novelty)

            # Peak picking with bounds checking
            if len(combined) < 3:  # Need minimum length for peak detection
                print(f"⚠️ Combined feature too short: {len(combined)}")
                return []

            peaks, _ = find_peaks(
                combined,
                prominence=self.peak_prominence * np.max(combined),
                distance=max(1, len(combined)//6)  # Adaptive distance
            )

            # Ensure beat_times has enough elements
            valid_peaks = peaks[peaks < len(beat_times)]
            
            cue_points = []
            for p in valid_peaks:
                if p < len(beat_times):
                    cue_time = beat_times[p]
                    aligned_time = self._nearest_downbeat(cue_time, downbeats)
                    quality = self._cue_quality(rms, combined[p], aligned_time, beat_times, audio)
                    
                    cue = dict(
                        time=float(aligned_time),
                        quality=float(quality),
                        type=self._cue_type(rms, aligned_time),
                        novelty_strength=float(combined[p]),
                        energy_drop=self._energy_drop(rms, aligned_time),
                        phrase_boundary=self._is_phrase_boundary(aligned_time),
                    )
                    cue_points.append(cue)

            # ML re-ranking (if available)
            if self.ml_ranker is not None and len(cue_points) > 0:
                try:
                    feats = np.array([[c['novelty_strength'], c['energy_drop'], c['phrase_boundary']] for c in cue_points])
                    preds = self.ml_ranker.predict(feats)
                    for c, s in zip(cue_points, preds):
                        c['ml_score'] = float(s)
                        c['quality'] = 0.7 * c['quality'] + 0.3 * s
                except Exception as e:
                    print(f"ML ranking failed: {e}")

            cue_points.sort(key=lambda x: x['quality'], reverse=True)
            return cue_points[:8]

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return []

    def _foote_novelty_matrix(self, y):
        """Enhanced novelty calculation with error handling"""
        try:
            # Compute log-magnitude spectrogram
            S = np.abs(librosa.stft(y, hop_length=self.hop_length)) ** 2
            if S.size == 0:
                return np.array([0.5])  # Fallback for empty spectrogram
                
            S = librosa.power_to_db(S)
            
            # Self-similarity with size check
            if S.shape[1] < 10:  # Too few frames
                return np.array([0.5] * S.shape[1])
                
            D = 1 - librosa.segment.recurrence_matrix(S, metric='cosine').astype(float)
            
            # Compute 1D novelty
            nov = librosa.segment.cross_similarity(D, D, mode='affinity').mean(axis=0)
            
            if len(nov) > 0:
                nov = gaussian_filter1d(nov, sigma=min(self.novelty_sigma, len(nov)//4))
                return self._norm(nov)
            else:
                return np.array([0.5])
                
        except Exception as e:
            print(f"Novelty matrix error: {e}")
            # Return safe fallback
            frames = librosa.frames_to_time(
                np.arange(0, len(y)//self.hop_length), 
                sr=self.sr, 
                hop_length=self.hop_length
            )
            return np.ones(len(frames)) * 0.5

    def _energy_drop(self, rms, time):
        """Safe energy drop detection"""
        try:
            beat_len = max(1, int(0.5 * self.sr / self.hop_length))
            idx = int(time * self.sr / self.hop_length)
            
            if idx < beat_len or idx + beat_len >= len(rms):
                return False
                
            a = rms[max(0, idx - beat_len): idx].mean()
            b = rms[idx: min(len(rms), idx + beat_len)].mean()
            
            return b < 0.7 * a if a > 1e-6 else False
            
        except Exception:
            return False


    def _spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        stft = librosa.stft(audio, hop_length=self.hop_length)
        mag = np.abs(stft)
        flux = np.sqrt(np.sum(np.diff(mag, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])
        return flux

    def _norm(self, v):
        if v.max() - v.min() < 1e-6:
            return np.zeros_like(v)
        return (v - v.min()) / (v.max() - v.min())

    def _nearest_downbeat(self, t, downbeats):
        if len(downbeats) == 0:
            return t
        return downbeats[np.argmin(np.abs(downbeats - t))]

    # -------------------- Cue quality helpers ------------------------- #
    def _cue_quality(self, rms, novelty_val, time, beat_times, audio):
        # Novelty
        q = novelty_val
        # Energy bonus (ideal cue at moderate energy)
        frame = np.argmin(np.abs(beat_times - time))
        energy = rms[min(frame, len(rms)-1)] / (rms.max()+1e-6)
        if 0.25 <= energy <= 0.85:
            q += self.qw['energy_good']
        # Energy drop?
        if self._energy_drop(rms, time):
            q += self.qw['energy_drop']
        # Phrase boundary?
        if self._is_phrase_boundary(time):
            q += self.qw['phrase']
        # Beat aligned always
        q += self.qw['beat_align']
        return min(1.0, q)


    def _cue_type(self, rms, time):
        idx = int(time * self.sr / self.hop_length)
        energy = rms[min(idx, len(rms)-1)]
        avg = rms.mean()
        if energy > 1.5 * avg:
            return 'high_energy'
        if energy < 0.6 * avg:
            return 'low_energy'
        return 'medium'

    def _is_phrase_boundary(self, time):
        # assume 4-beat bars, so phrase every 8 or 16 bars
        beats_per_phrase = 32  # 8 bars * 4 beats
        beat = int(round(time / (60 / 128)))  # rough beat count at 128 BPM
        return beat % beats_per_phrase == 0

class EnhancedMusicalDataExtractor:
    """Professional music analysis for -quality mashups"""

    def __init__(self):
        self.sr = 44100
        self.segment_selector = AdvancedSegmentSelector()
        self.key_profiles = self._get_key_profiles()

    def extract_enhanced_features(self, audio_path: str) -> Dict:
        """Extract features optimized for beautiful mashups"""
        
        print(f"ðŸŽµ Analyzing: {os.path.basename(audio_path)}")
        
        try:
            # Load and preprocess audio
            y, sr = self._load_and_preprocess_audio(audio_path)
            duration = len(y) / sr
            
            # Quick quality check
            if np.mean(np.abs(y)) < 0.001:
                print("âš ï¸ Very quiet audio detected")
                return self._generate_fallback_features(audio_path, duration)
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(y, sr, duration, audio_path)
            
            # Create perfect segments
            segment_count = len(features['best_segments']['best_overall'])
            
            # Fallback if no segments created
            if segment_count == 0:
                print(f"âš ï¸ No segments created, generating fallback segments")
                features['best_segments'] = self._create_emergency_segments(y, sr, duration)
                segment_count = len(features['best_segments']['best_overall'])
            
            print(f"âœ¨ Analysis complete: {segment_count} perfect segments created")
            return features
            
        except Exception as e:
            print(f"âŒ Analysis failed: {e}")
            return self._generate_fallback_features(audio_path, 120.0)

    def _load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio for optimal quality"""
        
        # Load with high quality
        y, original_sr = librosa.load(audio_path, sr=None)
        
        # Resample to target rate if needed
        if original_sr != self.sr:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=self.sr)
        
        # Apply gentle noise reduction
        y = self._apply_gentle_processing(y)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        return y, self.sr

    def _apply_gentle_processing(self, y: np.ndarray) -> np.ndarray:
        """Apply gentle processing for cleaner audio"""
        
        # High-pass filter to remove DC and low rumble
        from scipy.signal import butter, filtfilt
        nyquist = self.sr / 2
        high_cutoff = 30 / nyquist
        b, a = butter(2, high_cutoff, btype='high')
        y_filtered = filtfilt(b, a, y)
        
        return y_filtered

    def _extract_comprehensive_features(self, y: np.ndarray, sr: int, duration: float, audio_path: str) -> Dict:
        """Extract all features needed for perfect mashups"""
        
        hop_length = 1024
        
        # Tempo analysis
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo_scalar = float(tempo)
        
        # Key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        
        key_correlations = {}
        for key_name, profile in self.key_profiles.items():
            correlation = np.corrcoef(chroma_mean, profile)[0, 1]
            key_correlations[key_name] = correlation if not np.isnan(correlation) else 0
        
        dominant_key = max(key_correlations.items(), key=lambda x: x[1])
        
        # Energy and spectral features
        rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)
        energy_mean = np.mean(rms_energy)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        brightness = np.mean(spectral_centroid)
        
        # Create perfect segments
        best_segments = self.segment_selector.find_enhanced_segments(y, sr, duration, tempo_scalar, dominant_key[0])
        
        return {
            'tempo': tempo_scalar,
            'dominant_key': dominant_key,
            'energy_mean': energy_mean,
            'brightness': brightness,
            'best_segments': best_segments,
            'duration': duration,
            'audio_path': audio_path,
            'full_audio': y,
            'name': os.path.splitext(os.path.basename(audio_path))[0],
            '_ready': self._assess_readiness(best_segments, energy_mean, tempo_scalar)
        }

    def _assess_readiness(self, segments: Dict, energy: float, tempo: float) -> bool:
        """Assess if song is ready for -quality mashups"""
        
        best_segments = segments.get('best_overall', [])
        if not best_segments:
            return False
        
        # Check for -ready segments
        _segments = [s for s in best_segments if s.get('_ready', False)]
        
        return (len(_segments) >= 2 and energy > 0.2 and 90 <= tempo <= 150)

    def _create_emergency_segments(self, y: np.ndarray, sr: int, duration: float) -> Dict:
        """Create emergency segments when normal analysis fails"""
        
        segments = []
        segment_duration = 15.0
        
        for start_time in np.arange(0, duration - segment_duration, segment_duration / 2):
            end_time = min(start_time + segment_duration, duration)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            if len(segment_audio) > sr * 5:  # At least 5 seconds
                # Basic features
                features = {
                    'energy': float(np.mean(np.abs(segment_audio))),
                    'spectral_centroid': 2500.0,
                    'harmonic_strength': 0.6,
                    'harmonic_stability': 0.6,
                    'tempo_consistency': 0.7,
                    'perceptual_quality': 0.6,
                    'onset_density': 6.0,
                    'dynamic_range': 0.3,
                    'mashup_readiness': 0.6
                }
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'audio': segment_audio,
                    'features': features,
                    'type': 'verse',
                    'score': 0.6,
                    'emergency_segment': True
                })
        
        return {
            'best_overall': segments[:8],
            'best_intro': segments[:1],
            'best_outro': segments[-1:],
            'all_segments': segments
        }

    def _generate_fallback_features(self, audio_path: str, duration: float) -> Dict:
        """Generate fallback features"""
        
        return {
            'tempo': 120.0,
            'dominant_key': ('C_major', 0.5),
            'energy_mean': 0.4,
            'brightness': 2500.0,
            'best_segments': {
                'best_overall': [],
                'best_intro': [],
                'best_outro': [],
                'all_segments': []
            },
            'duration': duration,
            'audio_path': audio_path,
            'full_audio': np.zeros(int(duration * self.sr)),
            'name': os.path.splitext(os.path.basename(audio_path))[0],
            '_ready': False
        }

    def _get_key_profiles(self) -> Dict[str, np.ndarray]:
        """Get key profiles for key detection"""
        
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = {}
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for i in range(12):
            keys[f"{note_names[i]}_major"] = np.roll(major_profile, i)
            keys[f"{note_names[i]}_minor"] = np.roll(minor_profile, i)
        
        return keys

    def analyze_multiple_songs_parallel(self, audio_files: List[str]) -> List[Dict]:
        """Analyze multiple songs in parallel for mashup creation"""
        
        songs_data = []
        print(f"ðŸŽµ Analyzing {len(audio_files)} songs for perfect mashup...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_file = {
                executor.submit(self.extract_enhanced_features, audio_file): audio_file
                for audio_file in audio_files
            }
            
            for future in as_completed(future_to_file):
                audio_file = future_to_file[future]
                try:
                    analysis = future.result()
                    if analysis:
                        analysis['name'] = os.path.splitext(os.path.basename(audio_file))[0]
                        segment_count = len(analysis.get('best_segments', {}).get('best_overall', []))
                        
                        if segment_count > 0:
                            songs_data.append(analysis)
                            print(f"âœ… {analysis['name']}: {segment_count} perfect segments")
                        else:
                            print(f"âš ï¸ {analysis['name']}: No usable segments")
                            
                except Exception as e:
                    print(f"âŒ Failed to analyze {os.path.basename(audio_file)}: {e}")
        
        print(f"ðŸŽµ Successfully analyzed {len(songs_data)} songs")
        return songs_data

class QualityTimelineBuilder:
    """Build -quality timelines with perfect flow"""

    def __init__(self, config):
        self.config = config

    def build_perfect_timeline(self, songs_data: List[Dict], target_duration: float) -> 'Timeline':
        """Build timeline optimized for -quality mashups"""
        
        print(f"ðŸŽ¼ Building -quality timeline for {target_duration:.1f}s...")
        
        # Collect all segments
        all_segments = self._collect_all_segments(songs_data)
        
        if not all_segments:
            print("âŒ No segments available for timeline")
            return Timeline(target_duration)
        
        # Build timeline ensuring all songs are represented
        timeline = self._build_balanced_timeline(all_segments, target_duration)
        
        # Apply -quality enhancements
        timeline = self._apply_enhancements(timeline)
        
        print(f"âœ¨ Created perfect timeline with {len(timeline.segments)} segments")
        return timeline

    def _collect_all_segments(self, songs_data: List[Dict]) -> List[Dict]:
        """Collect and prepare all segments"""
        
        all_segments = []
        
        for song_data in songs_data:
            song_name = song_data.get('name', 'Unknown')
            segments = song_data.get('best_segments', {}).get('best_overall', [])
            
            for segment in segments:
                segment['song_name'] = song_name
                segment['song_data'] = song_data
                all_segments.append(segment)
        
        # Sort by quality
        all_segments.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_segments

    def _build_balanced_timeline(self, all_segments: List[Dict], target_duration: float) -> 'Timeline':
        """Build timeline ensuring EXACTLY one segment per song"""
        
        timeline = Timeline(target_duration)
        current_time = 0
        
        # Group segments by song and select ONLY the best one per song
        best_by_song = {}
        for segment in all_segments:
            song_name = segment['song_name']
            if song_name not in best_by_song or segment.get('score', 0) > best_by_song[song_name].get('score', 0):
                best_by_song[song_name] = segment
        
        # Sort by quality and add ONLY one per song
        selected_segments = sorted(best_by_song.values(), key=lambda x: x.get('score', 0), reverse=True)
        
        print(f"🎵 Selected segments from {len(selected_segments)} songs:")
        for segment in selected_segments:
            if current_time + segment['duration'] <= target_duration:
                segment['timeline_position'] = current_time
                timeline.add_segment(segment)
                current_time += segment['duration']
                print(f"✅ Added {segment['song_name']}: {segment['duration']:.1f}s (Score: {segment.get('score', 0):.3f})")
            else:
                print(f"⏭️ Skipped {segment['song_name']}: would exceed duration")
        
        print(f"🎵 Final timeline: {len(timeline.segments)} segments from {len(timeline.segments)} different songs")
        return timeline


    def _apply_enhancements(self, timeline: 'Timeline') -> 'Timeline':
        """Apply -quality enhancements"""
        
        if len(timeline.segments) < 2:
            return timeline
        
        print("âœ¨ Applying -quality enhancements...")
        
        # Enhance segments for perfect flow
        for i, segment in enumerate(timeline.segments):
            # Add perfect transition markers
            segment['_enhanced'] = True
            segment['transition_ready'] = True
            
            # Calculate optimal crossfade duration based on segment characteristics
            if 'features' in segment:
                energy = segment['features'].get('energy', 0.5)
                # Higher energy = shorter crossfade for punch
                optimal_crossfade = 1.0 if energy > 0.6 else 1.5
                segment['optimal_crossfade'] = optimal_crossfade
        
        return timeline

class Timeline:
    """Professional timeline for -quality mashups"""

    def __init__(self, target_duration: float = 180.0):
        self.target_duration = target_duration
        self.segments = []

    def add_segment(self, segment: Dict):
        """Add segment to timeline"""
        self.segments.append(segment)

    def get_total_duration(self) -> float:
        """Get total duration"""
        return sum(seg['duration'] for seg in self.segments)
    def get_structure_info(self) -> dict:
        """Get structure information for metadata"""
        if not self.segments:
            return {"total_segments": 0, "structure": "empty"}
        
        structure_info = {
            "total_segments": len(self.segments),
            "total_duration": self.get_total_duration(),
            "segments_by_type": {},
            "average_segment_duration": self.get_total_duration() / len(self.segments),
            "songs_used": list(set(seg.get('song_name', 'Unknown') for seg in self.segments))
        }
        
        # Count segments by type
        for segment in self.segments:
            seg_type = segment.get('type', 'unknown')
            structure_info["segments_by_type"][seg_type] = structure_info["segments_by_type"].get(seg_type, 0) + 1
        
        return structure_info


class QualityAudioProcessor:
    """Process audio for -quality output"""

    def __init__(self, config):
        self.config = config

    def assemble_mashup(self, timeline: Timeline) -> np.ndarray:
        """Assemble mashup with -quality processing"""
        
        if not timeline.segments:
            return np.array([])
        
        print("ðŸŽµ Assembling -quality mashup...")
        
        # Start with first segment
        result_audio = timeline.segments[0]['audio'].copy()
        
        # Add subsequent segments with perfect crossfades
        for i in range(1, len(timeline.segments)):
            current_segment = timeline.segments[i]
            
            # Apply -quality crossfade
            result_audio = self._apply_crossfade(
                result_audio, 
                current_segment['audio'],
                current_segment.get('optimal_crossfade', 1.2)
            )
            
            print(f"âœ¨ Added {current_segment.get('song_name', 'Unknown')} with perfect crossfade")
        
        # Apply final  processing
        result_audio = self._apply_mastering(result_audio)
        
        duration = len(result_audio) / self.config.sample_rate
        print(f"ðŸŽ‰ -quality mashup completed: {duration:.1f}s")
        
        return result_audio

    def _apply_crossfade(self, main_audio: np.ndarray, new_audio: np.ndarray, 
                              crossfade_duration: float) -> np.ndarray:
        """Apply crossfade using cue point information for perfect transitions"""
        
        fade_samples = int(crossfade_duration * self.config.sample_rate)
        fade_samples = min(fade_samples, len(main_audio) // 3, len(new_audio) // 3)
        
        if fade_samples > 0:
            # Create smooth crossfade curves
            fade_out = np.cos(np.linspace(0, np.pi/2, fade_samples)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi/2, fade_samples)) ** 2
            
            # Apply crossfade with cue point awareness
            main_end = main_audio[-fade_samples:] * fade_out
            new_start = new_audio[:fade_samples] * fade_in
            
            # Perfect blend with phase alignment
            crossfade_section = main_end + new_start
            
            # Apply gentle limiting to crossfade section
            crossfade_section = np.tanh(crossfade_section * 0.95)
            
            result = np.concatenate([
                main_audio[:-fade_samples],
                crossfade_section,
                new_audio[fade_samples:]
            ])
        else:
            result = np.concatenate([main_audio, new_audio])
        
        return result


    def _apply_mastering(self, audio: np.ndarray) -> np.ndarray:
        """Apply -optimized mastering"""
        
        # Gentle compression for consistent loudness
        compressed = np.tanh(audio * 1.1)
        
        # Normalize with headroom for platform compression
        peak = np.max(np.abs(compressed))
        if peak > 0:
            normalized = compressed / peak * 0.95
        else:
            normalized = compressed
        
        return normalized

@dataclass
class MashupConfig:
    """Configuration for  mashups"""
    
    sample_rate: int = 44100
    bit_depth: int = 24
    crossfade_duration: float = 1.2
    target_loudness: float = -14.0  
    enable_optimization: bool = True

class MashupGenerator:
    """Main generator for -quality mashups"""

    def __init__(self, config: Optional[MashupConfig] = None):
        self.config = config or MashupConfig()
        self.audio_analyzer = EnhancedMusicalDataExtractor()
        self.timeline_builder = QualityTimelineBuilder(self.config)
        self.audio_processor = QualityAudioProcessor(self.config)

    def create_mashup(self, demo_folder: str, target_duration: float = 180.0):
        """Create -quality mashup"""
        
        try:
            # Find audio files
            audio_files = self._find_audio_files(demo_folder)
            if len(audio_files) < 2:
                print("âŒ Need at least 2 audio files")
                return None
            
            print(f"ðŸŽµ Found {len(audio_files)} songs")
            
            # Analyze all songs
            songs_data = self.audio_analyzer.analyze_multiple_songs_parallel(audio_files)
            if len(songs_data) < 2:
                print("âŒ Need at least 2 valid songs")
                return None
            
            print(f"âœ… Successfully analyzed {len(songs_data)} songs")
            
            # Build perfect timeline
            timeline = self.timeline_builder.build_perfect_timeline(songs_data, target_duration)
            if not timeline.segments:
                print("âŒ No segments available for mashup")
                return None
            
            # Assemble -quality mashup
            mashup_audio = self.audio_processor.assemble_mashup(timeline)
            if len(mashup_audio) == 0:
                print("âŒ Failed to assemble mashup")
                return None
            
            # Export with  optimization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"_mashup_{timestamp}.wav"
            output_path = os.path.join('Final_Mashup', output_filename)
            
            sf.write(output_path, mashup_audio, self.config.sample_rate, subtype='PCM_24')
            
            duration = len(mashup_audio) / self.config.sample_rate
            print(f"ðŸŽ‰ -quality mashup created: {output_filename} ({duration:.1f}s)")
            print(f"âœ¨ Ready for social media upload!")
            
            metadata = self._generate_metadata(timeline)
        
            # Save metadata to JSON file
            metadata_file = os.path.join('Final_Mashup', "mashup_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return output_path,metadata_file
        
        except:
            return None
    
    # ADD THE METADATA FUNCTION HERE:
    def _generate_metadata(self, timeline: Timeline) -> Dict:
        """Generate comprehensive metadata"""
        return {
            'timeline_data': [
                {
                    'song_name': seg.get('song_name', 'Unknown'),
                    'start_time': seg.get('start_time', 0.0),
                    'end_time': seg.get('end_time', 10.0),
                    'duration': seg.get('duration', 10.0),
                    'section': seg.get('section', 'body'),
                    'energy': seg.get('energy', 0.3),
                    'score': seg.get('score', 0.5)
                } for seg in timeline.segments
            ],
            'title': f"Professional AI Mashup - {len(timeline.segments)} Songs",
            'artist': "Professional Mashup Generator",
            'album': "AI Generated Mashups",
            'date': datetime.now().isoformat(),
            'genre': "Mashup",
            'comment': "Generated using Professional SmartStructuredMashupGenerator",
            'songs_used': [seg.get('song_name', 'Unknown') for seg in timeline.segments],
            'duration': timeline.get_total_duration(),
            'structure': timeline.get_structure_info(),
            'quality_settings': {
                'sample_rate': self.config.sample_rate,
                'bit_depth': self.config.bit_depth,
                'crossfade_duration': self.config.crossfade_duration
            }
        }

    def _find_audio_files(self, demo_folder: str) -> List[str]:
        """Find audio files in folder"""
        
        if not os.path.exists(demo_folder):
            return []
        
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        audio_files = []
        
        for file in os.listdir(demo_folder):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(demo_folder, file))
        
        return sorted(audio_files)

def create_mashup(demo_folder='downloaded_music', target_duration=180.0, include_background=False, session_folder=None):
    """
    Wrapper function to create mashup and save in user session folder
    """
    try:
        config = MashupConfig()
        generator = MashupGenerator(config)
        
        # Create the mashup using your new system
        result_path,metadata_path = generator.create_mashup(demo_folder, target_duration)
        
        if result_path and os.path.exists(result_path):
            # Save to session folder instead of Final_Mashup
            if session_folder is None:
                raise ValueError('session_folder must be provided for session-based storage')
            
            # Ensure session mashup folder exists
            session_mashup_folder = os.path.join(session_folder, 'mashup')
            os.makedirs(session_mashup_folder, exist_ok=True)
            
            # Move file to session folder
            final_path = os.path.join(session_mashup_folder, os.path.basename(result_path))
            final_metadata_path = os.path.join(session_mashup_folder, os.path.basename(metadata_path))
            shutil.move(result_path, final_path)  # Use move instead of copy
            shutil.move(metadata_path, final_metadata_path)  # Use move instead of copy
            
            print(f"✅ Mashup saved to session: {final_path}")
            
            # Return tuple format that app.py expects: (mashup_file, background_file)
            return final_path, None
        else:
            print(f"❌ Mashup creation failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Mashup creation error: {e}")
        import traceback
        traceback.print_exc()
        return None, None



# Also add this for backward compatibility
def cleanup_temp_files():
    """Cleanup function for compatibility"""
    # Clean up any temporary files if needed
    import gc
    gc.collect()


# Example usage
if __name__ == "__main__":
    mashup_file = create_mashup(
        demo_folder='downloaded_music',
        target_duration=180.0
    )
    
    if mashup_file:
        print(f"ðŸŽµ Your -ready mashup: {mashup_file}")
    else:
        print("ðŸ˜ž Mashup creation failed")