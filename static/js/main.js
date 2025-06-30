class MashupApp {
    constructor() {
        this.selectedSongs = [];
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const searchBtn = document.getElementById('searchBtn');
        const searchInput = document.getElementById('searchInput');
        const createMashupBtn = document.getElementById('createMashup');

        if (searchBtn) {
            searchBtn.addEventListener('click', () => {
                console.log('Search button clicked!'); // Debug log
                this.searchSongs();
            });
        } else {
            console.error('Search button not found!');
        }

        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    console.log('Enter key pressed!'); // Debug log
                    this.searchSongs();
                }
            });
        }

        if (createMashupBtn) {
            createMashupBtn.addEventListener('click', () => this.createMashup());
        }
    }

    async searchSongs() {
        const query = document.getElementById('searchInput').value.trim();
        console.log('Search query:', query); // Debug log
        
        if (!query) {
            alert('Please enter a search term');
            return;
        }

        const resultsContainer = document.getElementById('searchResults');
        resultsContainer.innerHTML = '<div class="loading">Searching...</div>';

        try {
            console.log('Sending fetch request...'); // Debug log
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            console.log('Response received:', response.status); // Debug log

            const data = await response.json();
            console.log('Response data:', data); // Debug log
            
            if (data.error) {
                resultsContainer.innerHTML = `<div class="error">${data.error}</div>`;
                return;
            }

            this.displaySearchResults(data.results);
        } catch (error) {
            console.error('Search error:', error); // Debug log
            resultsContainer.innerHTML = '<div class="error">Search failed. Please try again.</div>';
        }
    }

    displaySearchResults(results) {
        const container = document.getElementById('searchResults');
        
        if (!results || results.length === 0) {
            container.innerHTML = '<div class="no-results">No songs found.</div>';
            return;
        }

        container.innerHTML = results.map(song => `
            <div class="song-card" data-song='${JSON.stringify(song)}'>
                <div class="song-info">
                    <h3>${song.name}</h3>
                    <p>${song.artist}</p>
                    <small>${song.album}</small>
                </div>
                <button class="select-btn" onclick="app.toggleSong(this)">
                    Select
                </button>
            </div>
        `).join('');
    }

    toggleSong(button) {
        const songCard = button.closest('.song-card');
        const songData = JSON.parse(songCard.dataset.song);
        
        const existingIndex = this.selectedSongs.findIndex(s => s.id === songData.id);
        
        if (existingIndex > -1) {
            this.selectedSongs.splice(existingIndex, 1);
            button.textContent = 'Select';
            button.classList.remove('selected');
        } else {
            this.selectedSongs.push(songData);
            button.textContent = 'Selected';
            button.classList.add('selected');
        }
        
        this.updateSelectedList();
    }

    updateSelectedList() {
        const countElement = document.getElementById('selectedCount');
        const listElement = document.getElementById('selectedList');
        const createButton = document.getElementById('createMashup');
        
        if (countElement) countElement.textContent = this.selectedSongs.length;
        
        if (listElement) {
            listElement.innerHTML = this.selectedSongs.map(song => `
                <div class="selected-item">
                    <span>${song.name} - ${song.artist}</span>
                    <button onclick="app.removeSong('${song.id}')" class="remove-btn">×</button>
                </div>
            `).join('');
        }
        
        if (createButton) {
            createButton.disabled = this.selectedSongs.length === 0;
        }
    }

    removeSong(songId) {
        this.selectedSongs = this.selectedSongs.filter(s => s.id !== songId);
        this.updateSelectedList();
        
        // Update UI in search results
        const songCards = document.querySelectorAll('.song-card');
        songCards.forEach(card => {
            const songData = JSON.parse(card.dataset.song);
            if (songData.id === songId) {
                const button = card.querySelector('.select-btn');
                button.textContent = 'Select';
                button.classList.remove('selected');
            }
        });
    }

    async createMashup() {
        if (this.selectedSongs.length === 0) return;
        
        const modal = document.getElementById('processingModal');
        if (modal) modal.style.display = 'block';
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ songs: this.selectedSongs })
            });
            
            const data = await response.json();
            
            if (data.error) {
                alert('Error: ' + data.error);
                if (modal) modal.style.display = 'none';
                return;
            }
            
            // Poll for status updates
            this.pollStatus();
            
        } catch (error) {
            alert('Failed to start processing');
            if (modal) modal.style.display = 'none';
        }
    }

    async pollStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            const messageElement = document.getElementById('processingMessage');
            if (messageElement) messageElement.textContent = data.message;
            
            if (data.status === 'completed') {
                setTimeout(() => {
                    window.location.href = '/results';
                }, 2000);
            } else if (data.status === 'error') {
                alert('Error: ' + data.message);
                const modal = document.getElementById('processingModal');
                if (modal) modal.style.display = 'none';
            } else {
                // Continue polling
                setTimeout(() => this.pollStatus(), 2000);
            }
        } catch (error) {
            console.error('Status polling failed:', error);
            setTimeout(() => this.pollStatus(), 5000);
        }
    }
}

// Initialize the app when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing MashupApp...');
    window.app = new MashupApp();
});
