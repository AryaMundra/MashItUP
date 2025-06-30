class ProfessionalMashupEditor {
    constructor() {
        this.currentTime = 0;
        this.isPlaying = false;
        this.totalDuration = 0;
        this.zoomLevel = 1;
        this.selectedSegment = null;
        this.audioContext = null;
        this.audioBuffers = new Map();
        
        this.initializeEditor();
    }
    
    async initializeEditor() {
        this.setupAudioContext();
        this.bindEvents();
        this.setupTimeline();
        this.drawWaveforms();
        this.updateTimeRuler();
        
        // Get total duration from the page
        const durationText = document.getElementById('timeDisplay').textContent;
        this.totalDuration = parseFloat(durationText.split('/')[1].trim());
    }
    
    setupAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn('Web Audio API not supported');
        }
    }
    
    bindEvents() {
        // Transport controls
        document.getElementById('playBtn').addEventListener('click', () => this.togglePlayback());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pausePlayback());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopPlayback());
        document.getElementById('zoomInBtn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoomOutBtn').addEventListener('click', () => this.zoomOut());
        
        // Segment controls
        document.querySelectorAll('.action-btn.update').forEach(btn => {
            btn.addEventListener('click', (e) => this.updateSegment(e.target.dataset.segment));
        });
        
        document.querySelectorAll('.action-btn.preview').forEach(btn => {
            btn.addEventListener('click', (e) => this.previewSegment(e.target.dataset.segment));
        });
        
        document.querySelectorAll('.action-btn.delete').forEach(btn => {
            btn.addEventListener('click', (e) => this.deleteSegment(e.target.dataset.segment));
        });
        
        // Track controls
        document.querySelectorAll('.track-control-btn.mute').forEach(btn => {
            btn.addEventListener('click', (e) => this.toggleMute(e.target.dataset.track));
        });
        
        document.querySelectorAll('.track-control-btn.solo').forEach(btn => {
            btn.addEventListener('click', (e) => this.toggleSolo(e.target.dataset.track));
        });
        
        // Segment selection
        document.querySelectorAll('.segment-block').forEach(block => {
            block.addEventListener('click', (e) => this.selectSegment(e.target.dataset.segment));
        });
        
        // Export controls
        document.getElementById('rebuildBtn').addEventListener('click', () => this.rebuildMashup());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportMashup());
        document.getElementById('previewFullBtn').addEventListener('click', () => this.previewFullMashup());
        
        // Section change handlers
        document.querySelectorAll('.section-select').forEach(select => {
            select.addEventListener('change', (e) => this.updateSegmentSection(e.target.dataset.segment, e.target.value));
        });
        
        // Timeline click for seeking
        document.querySelector('.timeline-ruler').addEventListener('click', (e) => this.seekToPosition(e));
    }
    
    setupTimeline() {
        this.updateTimeRuler();
        this.positionSegments();
    }
    
    updateTimeRuler() {
        const rulerLabels = document.getElementById('rulerLabels');
        rulerLabels.innerHTML = '';
        
        const timeStep = Math.max(1, Math.floor(this.totalDuration / 20)); // Show ~20 labels max
        
        for (let t = 0; t <= this.totalDuration; t += timeStep) {
            const label = document.createElement('span');
            label.style.position = 'absolute';
            label.style.left = `${(t / this.totalDuration) * 100}%`;
            label.style.fontSize = '10px';
            label.style.color = '#999';
            label.textContent = this.formatTime(t);
            rulerLabels.appendChild(label);
        }
    }
    
    drawWaveforms() {
        document.querySelectorAll('.waveform-canvas').forEach(canvas => {
            this.drawWaveform(canvas);
        });
    }
    
    drawWaveform(canvas) {
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        
        // Set canvas size
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = rect.width;
        const height = rect.height;
        
        // Clear canvas
        ctx.fillStyle = '#2a2a2a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw mock waveform (in real implementation, you'd load actual audio data)
        this.drawMockWaveform(ctx, width, height);
    }
    
    drawMockWaveform(ctx, width, height) {
        ctx.strokeStyle = '#4ECDC4';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        const centerY = height / 2;
        const samples = width;
        
        for (let i = 0; i < samples; i++) {
            const x = i;
            // Generate mock waveform data
            const amplitude = Math.sin(i * 0.02) * Math.random() * 0.8;
            const y = centerY + amplitude * (height * 0.4);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Draw center line
        ctx.strokeStyle = '#555';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
    }
    
    positionSegments() {
        document.querySelectorAll('.segment-block').forEach(block => {
            const segmentIndex = block.dataset.segment;
            const segmentItem = document.querySelector(`[data-index="${segmentIndex}"]`);
            
            if (segmentItem) {
                const startTime = parseFloat(segmentItem.querySelector('.mashup-position').value);
                const duration = parseFloat(segmentItem.querySelector('.end-time').value) - 
                               parseFloat(segmentItem.querySelector('.start-time').value);
                
                const leftPercent = (startTime / this.totalDuration) * 100;
                const widthPercent = (duration / this.totalDuration) * 100;
                
                block.style.left = `${leftPercent}%`;
                block.style.width = `${widthPercent}%`;
            }
        });
    }
    
    selectSegment(segmentIndex) {
        // Remove previous selection
        document.querySelectorAll('.segment-block.selected').forEach(block => {
            block.classList.remove('selected');
        });
        
        document.querySelectorAll('.segment-item.selected').forEach(item => {
            item.classList.remove('selected');
        });
        
        // Add new selection
        const segmentBlock = document.querySelector(`[data-segment="${segmentIndex}"]`);
        const segmentItem = document.querySelector(`[data-index="${segmentIndex}"]`);
        
        if (segmentBlock && segmentItem) {
            segmentBlock.classList.add('selected');
            segmentItem.classList.add('selected');
            this.selectedSegment = segmentIndex;
            
            // Scroll to segment in editor panel
            segmentItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    
    async updateSegment(segmentIndex) {
        const segmentItem = document.querySelector(`[data-index="${segmentIndex}"]`);
        const songName = segmentItem.querySelector('.song-select').value;
        const section = segmentItem.querySelector('.section-select').value;
        const startTime = parseFloat(segmentItem.querySelector('.start-time').value);
        const endTime = parseFloat(segmentItem.querySelector('.end-time').value);
        const mashupPosition = parseFloat(segmentItem.querySelector('.mashup-position').value);
        
        if (endTime <= startTime) {
            this.showStatus('End time must be greater than start time', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/update-segment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    segment_index: parseInt(segmentIndex),
                    song_name: songName,
                    section: section,
                    start_time: startTime,
                    end_time: endTime,
                    mashup_position: mashupPosition
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.positionSegments();
                this.showStatus('Segment updated successfully', 'success');
                
                // Update segment block class for new section
                const segmentBlock = document.querySelector(`[data-segment="${segmentIndex}"]`);
                segmentBlock.className = `segment-block ${section}`;
                
                // Update section badge
                const sectionBadge = segmentItem.querySelector('.segment-section-badge');
                sectionBadge.textContent = section.charAt(0).toUpperCase() + section.slice(1);
                sectionBadge.className = `segment-section-badge ${section}`;
                
            } else {
                this.showStatus('Failed to update segment: ' + result.error, 'error');
            }
        } catch (error) {
            this.showStatus('Error updating segment: ' + error.message, 'error');
        }
    }
    
    updateSegmentSection(segmentIndex, newSection) {
        const segmentBlock = document.querySelector(`[data-segment="${segmentIndex}"]`);
        const segmentItem = document.querySelector(`[data-index="${segmentIndex}"]`);
        
        if (segmentBlock && segmentItem) {
            // Update visual styling
            segmentBlock.className = `segment-block ${newSection}`;
            if (segmentBlock.classList.contains('selected')) {
                segmentBlock.classList.add('selected');
            }
            
            // Update badge
            const sectionBadge = segmentItem.querySelector('.segment-section-badge');
            sectionBadge.textContent = newSection.charAt(0).toUpperCase() + newSection.slice(1);
            sectionBadge.className = `segment-section-badge ${newSection}`;
        }
    }
    
    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        const playBtn = document.getElementById('playBtn');
        
        if (this.isPlaying) {
            playBtn.innerHTML = '⏸ Pause';
            this.startPlayback();
        } else {
            playBtn.innerHTML = '▶ Play';
            this.pausePlayback();
        }
    }
    
    startPlayback() {
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
        }
        
        this.playbackInterval = setInterval(() => {
            this.currentTime += 0.1;
            if (this.currentTime >= this.totalDuration) {
                this.stopPlayback();
                return;
            }
            this.updatePlayhead();
            this.updateTimeDisplay();
        }, 100);
    }
    
    pausePlayback() {
        this.isPlaying = false;
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
        }
        document.getElementById('playBtn').innerHTML = '▶ Play';
    }
    
    stopPlayback() {
        this.isPlaying = false;
        this.currentTime = 0;
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
        }
        this.updatePlayhead();
        this.updateTimeDisplay();
        document.getElementById('playBtn').innerHTML = '▶ Play';
    }
    
    updatePlayhead() {
        const playhead = document.getElementById('playhead');
        const position = (this.currentTime / this.totalDuration) * 100;
        playhead.style.left = `${position}%`;
    }
    
    updateTimeDisplay() {
        const display = document.getElementById('timeDisplay');
        const currentFormatted = this.formatTime(this.currentTime);
        const totalFormatted = this.formatTime(this.totalDuration);
        display.textContent = `${currentFormatted} / ${totalFormatted}`;
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(3);
        return `${mins.toString().padStart(2, '0')}:${secs.padStart(6, '0')}`;
    }
    
    seekToPosition(event) {
        const ruler = event.currentTarget;
        const rect = ruler.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const position = clickX / rect.width;
        
        this.currentTime = position * this.totalDuration;
        this.updatePlayhead();
        this.updateTimeDisplay();
    }
    
    zoomIn() {
        this.zoomLevel = Math.min(this.zoomLevel * 1.5, 10);
        this.applyZoom();
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(this.zoomLevel / 1.5, 0.1);
        this.applyZoom();
    }
    
    applyZoom() {
        const timeline = document.querySelector('.multitrack-timeline');
        timeline.style.transform = `scaleX(${this.zoomLevel})`;
        timeline.style.transformOrigin = 'left center';
    }
    
    toggleMute(trackIndex) {
        const muteBtn = document.querySelector(`[data-track="${trackIndex}"].mute`);
        muteBtn.classList.toggle('active');
        
        if (muteBtn.classList.contains('active')) {
            muteBtn.style.background = '#d32f2f';
            muteBtn.textContent = 'M';
        } else {
            muteBtn.style.background = '#555';
            muteBtn.textContent = 'M';
        }
    }
    
    toggleSolo(trackIndex) {
        const soloBtn = document.querySelector(`[data-track="${trackIndex}"].solo`);
        soloBtn.classList.toggle('active');
        
        if (soloBtn.classList.contains('active')) {
            soloBtn.style.background = '#f57c00';
            soloBtn.textContent = 'S';
        } else {
            soloBtn.style.background = '#555';
            soloBtn.textContent = 'S';
        }
    }
    
    async previewSegment(segmentIndex) {
        this.showStatus(`Previewing segment ${parseInt(segmentIndex) + 1}`, 'info');
        
        // In a real implementation, you would load and play the specific segment
        const segmentItem = document.querySelector(`[data-index="${segmentIndex}"]`);
        const songName = segmentItem.querySelector('.song-select').value;
        const startTime = parseFloat(segmentItem.querySelector('.start-time').value);
        const endTime = parseFloat(segmentItem.querySelector('.end-time').value);
        
        console.log(`Preview: ${songName} from ${startTime}s to ${endTime}s`);
    }
    
    async deleteSegment(segmentIndex) {
        if (confirm('Are you sure you want to delete this segment?')) {
            try {
                const response = await fetch('/api/delete-segment', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        segment_index: parseInt(segmentIndex)
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Remove from UI
                    const segmentItem = document.querySelector(`[data-index="${segmentIndex}"]`);
                    const segmentBlock = document.querySelector(`[data-segment="${segmentIndex}"]`);
                    const audioTrack = document.querySelector(`[data-track="${segmentIndex}"]`);
                    
                    if (segmentItem) segmentItem.remove();
                    if (segmentBlock) segmentBlock.remove();
                    if (audioTrack) audioTrack.remove();
                    
                    this.showStatus('Segment deleted successfully', 'success');
                } else {
                    this.showStatus('Failed to delete segment: ' + result.error, 'error');
                }
            } catch (error) {
                this.showStatus('Error deleting segment: ' + error.message, 'error');
            }
        }
    }
    
    async rebuildMashup() {
        this.showStatus('Rebuilding mashup...', 'info');
        
        try {
            const response = await fetch('/api/rebuild-mashup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showStatus('Mashup rebuilt successfully', 'success');
                this.positionSegments();
                this.drawWaveforms();
            } else {
                this.showStatus('Failed to rebuild mashup: ' + result.error, 'error');
            }
        } catch (error) {
            this.showStatus('Error rebuilding mashup: ' + error.message, 'error');
        }
    }
    
    async exportMashup() {
        this.showStatus('Exporting edited mashup...', 'info');
        
        try {
            const response = await fetch('/api/export-edited-mashup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showStatus('Mashup exported successfully!', 'success');
                
                // Create download link
                const downloadLink = document.createElement('a');
                downloadLink.href = result.download_url;
                downloadLink.download = result.filename;
                downloadLink.textContent = '📥 Download Edited Mashup';
                downloadLink.className = 'export-btn';
                downloadLink.style.marginTop = '1rem';
                
                const statusDiv = document.getElementById('exportStatus');
                statusDiv.appendChild(downloadLink);
            } else {
                this.showStatus('Failed to export mashup: ' + result.error, 'error');
            }
        } catch (error) {
            this.showStatus('Error exporting mashup: ' + error.message, 'error');
        }
    }
    
    previewFullMashup() {
        this.showStatus('Loading full mashup preview...', 'info');
        
        // In a real implementation, you would load the current mashup state
        const audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = '/preview/current_mashup.wav'; // This would be the current state
        audioPlayer.play().catch(e => {
            this.showStatus('Could not play preview: ' + e.message, 'error');
        });
    }
    
    showStatus(message, type) {
        const statusDiv = document.getElementById('exportStatus');
        statusDiv.className = `export-status ${type}`;
        statusDiv.style.display = 'block';
        statusDiv.innerHTML = `<div>${message}</div>`;
        
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
}


// Undo/Redo System
class UndoRedoManager {
    constructor() {
        this.undoStack = [];
        this.redoStack = [];
        this.maxStackSize = 50;
    }
    
    pushAction(action) {
        this.undoStack.push(action);
        this.redoStack = []; // Clear redo stack when new action is performed
        
        // Limit stack size
        if (this.undoStack.length > this.maxStackSize) {
            this.undoStack.shift();
        }
        
        this.updateButtons();
    }
    
    undo() {
        if (this.undoStack.length === 0) return false;
        
        const action = this.undoStack.pop();
        this.redoStack.push(action);
        
        // Execute undo
        if (action.undo) {
            action.undo();
        }
        
        this.updateButtons();
        this.showUndoToast(action.description);
        return true;
    }
    
    redo() {
        if (this.redoStack.length === 0) return false;
        
        const action = this.redoStack.pop();
        this.undoStack.push(action);
        
        // Execute redo
        if (action.redo) {
            action.redo();
        }
        
        this.updateButtons();
        return true;
    }
    
    updateButtons() {
        const undoBtn = document.getElementById('undoBtn');
        const redoBtn = document.getElementById('redoBtn');
        
        if (undoBtn) {
            undoBtn.disabled = this.undoStack.length === 0;
            undoBtn.textContent = this.undoStack.length > 0 
                ? `↶ Undo (${this.undoStack.length})` 
                : '↶ Undo';
        }
        
        if (redoBtn) {
            redoBtn.disabled = this.redoStack.length === 0;
            redoBtn.textContent = this.redoStack.length > 0 
                ? `↷ Redo (${this.redoStack.length})` 
                : '↷ Redo';
        }
    }
    
    showUndoToast(description) {
        const existingToast = document.querySelector('.undo-toast');
        if (existingToast) {
            existingToast.remove();
        }
        
        const toast = document.createElement('div');
        toast.className = 'undo-toast';
        toast.innerHTML = `
            <span class="undo-toast-message">${description} undone</span>
            <button class="undo-toast-btn" onclick="undoManager.redo()">Redo</button>
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 100);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// Global undo manager
let undoManager = new UndoRedoManager();

// Used segments tracking
let usedSegments = new Map(); // songName -> array of used segments

// Add to initialization
function initializeEditor() {
    // ... existing initialization code ...
    
    // Initialize undo/redo
    initializeUndoRedo();
    
    // Initialize used segments display
    initializeUsedSegments();
    
    // ... rest of existing code ...
}

function initializeUndoRedo() {
    document.getElementById('undoBtn')?.addEventListener('click', () => {
        undoManager.undo();
    });
    
    document.getElementById('redoBtn')?.addEventListener('click', () => {
        undoManager.redo();
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            undoManager.undo();
        } else if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
            e.preventDefault();
            undoManager.redo();
        }
    });
}

function initializeUsedSegments() {
    // Load initial used segments from existing timeline segments
    loadUsedSegmentsFromTimeline();
    
    // Update displays
    updateAllUsedSegmentsDisplay();
}

function loadUsedSegmentsFromTimeline() {
    usedSegments.clear();
    
    document.querySelectorAll('.timeline-segment').forEach(segment => {
        const songName = segment.dataset.song;
        const startTime = parseFloat(segment.dataset.start);
        const endTime = parseFloat(segment.dataset.end);
        const position = parseFloat(segment.dataset.position);
        const section = segment.classList[1];
        
        if (!usedSegments.has(songName)) {
            usedSegments.set(songName, []);
        }
        
        usedSegments.get(songName).push({
            startTime,
            endTime,
            position,
            section,
            segmentIndex: segment.dataset.segment
        });
    });
}

function updateUsedSegmentsForSong(songName) {
    const segments = usedSegments.get(songName) || [];
    const audio = songAudios.get(songName);
    
    if (!audio) return;
    
    const duration = audio.duration || 100;
    
    // Update waveform overlay
    updateWaveformOverlay(songName, segments, duration);
    
    // Update range track markers
    updateRangeTrackMarkers(songName, segments, duration);
    
    // Update info display
    updateUsedSegmentsInfo(songName, segments);
}

function updateWaveformOverlay(songName, segments, duration) {
    const overlay = document.querySelector(`[data-song="${songName}"].used-segments-overlay`);
    if (!overlay) return;
    
    overlay.innerHTML = '';
    
    segments.forEach((segment, index) => {
        const startPercent = (segment.startTime / duration) * 100;
        const widthPercent = ((segment.endTime - segment.startTime) / duration) * 100;
        
        const indicator = document.createElement('div');
        indicator.className = 'used-segment-indicator';
        indicator.style.left = `${startPercent}%`;
        indicator.style.width = `${widthPercent}%`;
        indicator.title = `${segment.section}: ${formatTimeSimple(segment.startTime)} - ${formatTimeSimple(segment.endTime)}`;
        
        overlay.appendChild(indicator);
    });
}

function updateRangeTrackMarkers(songName, segments, duration) {
    const track = document.querySelector(`[data-song="${songName}"].used-segments-track`);
    if (!track) return;
    
    track.innerHTML = '';
    
    segments.forEach((segment, index) => {
        const startPercent = (segment.startTime / duration) * 100;
        const widthPercent = ((segment.endTime - segment.startTime) / duration) * 100;
        
        const marker = document.createElement('div');
        marker.className = 'used-segment-marker';
        marker.style.left = `${startPercent}%`;
        marker.style.width = `${widthPercent}%`;
        
        track.appendChild(marker);
    });
}

function updateUsedSegmentsInfo(songName, segments) {
    const countElement = document.querySelector(`[data-song="${songName}"].used-segments-count`);
    const listElement = document.querySelector(`[data-song="${songName}"].used-segments-list`);
    
    if (countElement) {
        countElement.textContent = `${segments.length} segment${segments.length !== 1 ? 's' : ''}`;
    }
    
    if (listElement) {
        listElement.innerHTML = '';
        
        segments.forEach((segment, index) => {
            const item = document.createElement('div');
            item.className = 'used-segment-item';
            item.innerHTML = `
                <span class="used-segment-time">${formatTimeSimple(segment.startTime)} - ${formatTimeSimple(segment.endTime)}</span>
                <span class="used-segment-section">${segment.section}</span>
                <span class="used-segment-position">@${formatTimeSimple(segment.position)}</span>
            `;
            
            listElement.appendChild(item);
        });
    }
}

function updateAllUsedSegmentsDisplay() {
    usedSegments.forEach((segments, songName) => {
        updateUsedSegmentsForSong(songName);
    });
}

// Enhanced segment management with undo support
async function addNewSegmentWithUndo(songName) {
    const selector = rangeSelectors.get(songName);
    const audio = songAudios.get(songName);
    
    if (!selector || !audio) return;
    
    const duration = audio.duration || 100;
    const startTime = selector.startPos * duration;
    const endTime = selector.endPos * duration;
    const section = document.querySelector(`[data-song="${songName}"].section-select`).value;
    
    showStatus('Adding new segment...', 'info');
    
    try {
        const response = await fetch('/api/add-segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                song_name: songName,
                start_time: startTime,
                end_time: endTime,
                section: section
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            const segmentData = result.segment_data;
            
            // Add to timeline
            addSegmentToTimeline(segmentData);
            
            // Update used segments
            if (!usedSegments.has(songName)) {
                usedSegments.set(songName, []);
            }
            usedSegments.get(songName).push({
                startTime,
                endTime,
                position: segmentData.position,
                section,
                segmentIndex: document.querySelectorAll('.timeline-segment').length - 1
            });
            
            updateUsedSegmentsForSong(songName);
            
            // Add undo action
            undoManager.pushAction({
                description: `Add segment from ${songName}`,
                undo: () => {
                    removeSegmentFromTimeline(segmentData);
                    removeFromUsedSegments(songName, startTime, endTime);
                    updateUsedSegmentsForSong(songName);
                },
                redo: () => {
                    addSegmentToTimeline(segmentData);
                    addToUsedSegments(songName, startTime, endTime, segmentData.position, section);
                    updateUsedSegmentsForSong(songName);
                }
            });
            
            showStatus('Segment added successfully', 'success');
            updateTimelineInfo();
        } else {
            showStatus('Failed to add segment: ' + result.error, 'error');
        }
    } catch (error) {
        showStatus('Error adding segment: ' + error.message, 'error');
    }
}

async function replaceSegmentWithUndo(songName) {
    if (!selectedSegment) {
        showStatus('Please select a segment to replace', 'error');
        return;
    }
    
    const selector = rangeSelectors.get(songName);
    const audio = songAudios.get(songName);
    
    if (!selector || !audio) return;
    
    const duration = audio.duration || 100;
    const startTime = selector.startPos * duration;
    const endTime = selector.endPos * duration;
    const section = document.querySelector(`[data-song="${songName}"].section-select`).value;
    
    // Store original data for undo
    const originalData = {
        songName: selectedSegment.dataset.song,
        startTime: parseFloat(selectedSegment.dataset.start),
        endTime: parseFloat(selectedSegment.dataset.end),
        section: selectedSegment.classList[1],
        position: parseFloat(selectedSegment.dataset.position)
    };
    
    showStatus('Replacing segment...', 'info');
    
    try {
        const segmentIndex = selectedSegment.dataset.segment;
        const response = await fetch('/api/update-segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                segment_index: parseInt(segmentIndex),
                song_name: songName,
                section: section,
                start_time: startTime,
                end_time: endTime,
                mashup_position: parseFloat(selectedSegment.dataset.position)
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update visual
            updateSegmentVisual(selectedSegment, songName, section, startTime, endTime);
            
            // Update used segments
            removeFromUsedSegments(originalData.songName, originalData.startTime, originalData.endTime);
            addToUsedSegments(songName, startTime, endTime, originalData.position, section);
            
            updateUsedSegmentsForSong(originalData.songName);
            updateUsedSegmentsForSong(songName);
            
            // Add undo action
            undoManager.pushAction({
                description: `Replace segment with ${songName}`,
                undo: () => {
                    updateSegmentVisual(selectedSegment, originalData.songName, originalData.section, originalData.startTime, originalData.endTime);
                    removeFromUsedSegments(songName, startTime, endTime);
                    addToUsedSegments(originalData.songName, originalData.startTime, originalData.endTime, originalData.position, originalData.section);
                    updateUsedSegmentsForSong(songName);
                    updateUsedSegmentsForSong(originalData.songName);
                },
                redo: () => {
                    updateSegmentVisual(selectedSegment, songName, section, startTime, endTime);
                    removeFromUsedSegments(originalData.songName, originalData.startTime, originalData.endTime);
                    addToUsedSegments(songName, startTime, endTime, originalData.position, section);
                    updateUsedSegmentsForSong(originalData.songName);
                    updateUsedSegmentsForSong(songName);
                }
            });
            
            showStatus('Segment replaced successfully', 'success');
        } else {
            showStatus('Failed to replace segment: ' + result.error, 'error');
        }
    } catch (error) {
        showStatus('Error replacing segment: ' + error.message, 'error');
    }
}

// Helper functions for used segments management
function addToUsedSegments(songName, startTime, endTime, position, section) {
    if (!usedSegments.has(songName)) {
        usedSegments.set(songName, []);
    }
    
    usedSegments.get(songName).push({
        startTime,
        endTime,
        position,
        section,
        segmentIndex: document.querySelectorAll('.timeline-segment').length - 1
    });
}

function removeFromUsedSegments(songName, startTime, endTime) {
    if (!usedSegments.has(songName)) return;
    
    const segments = usedSegments.get(songName);
    const index = segments.findIndex(seg => 
        Math.abs(seg.startTime - startTime) < 0.1 && 
        Math.abs(seg.endTime - endTime) < 0.1
    );
    
    if (index !== -1) {
        segments.splice(index, 1);
    }
}

function removeSegmentFromTimeline(segmentData) {
    const segments = document.querySelectorAll('.timeline-segment');
    segments.forEach(segment => {
        if (segment.dataset.song === segmentData.song_name &&
            Math.abs(parseFloat(segment.dataset.start) - segmentData.start_time) < 0.1 &&
            Math.abs(parseFloat(segment.dataset.end) - segmentData.end_time) < 0.1) {
            segment.remove();
        }
    });
}

// Update existing functions to use undo versions
function replaceSegment(songName) {
    replaceSegmentWithUndo(songName);
}

function addNewSegment(songName) {
    addNewSegmentWithUndo(songName);
}

// Update song audio loading to show used segments
document.querySelectorAll('.song-audio').forEach(audio => {
    const songName = audio.dataset.song;
    
    audio.addEventListener('loadedmetadata', function() {
        initializeWaveform(songName);
        updateSongTimeLabels(songName);
        updateUsedSegmentsForSong(songName); // Add this line
    });
});

// Initialize editor when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Professional Mashup Editor...');
    window.editor = new ProfessionalMashupEditor();
});
