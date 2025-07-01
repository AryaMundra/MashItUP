# MashItUP 
MashItUP is a web-based music mashup editor that lets you create, edit, and export custom audio mashups from your favorite songs. With timeline editing, segment selection, seamless transitions, and audio previews. 

# Features
Create custom music mashups from your favorite songs

Just search for the song required

Arrange and combine segments to craft unique mashups

Preview ,edit and export your finished mashup as an audio file

All in your browser, no extra software required


#  Getting Started
1. Clone the repository
bash
git clone https://github.com/yourusername/mashitup.git
cd mashitup
2. Install dependencies
Requires Python 3.8+

bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# Set up environment variables
Create a .env file in the project root:


FLASK_SECRET_KEY=your-secret-key
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
Get Spotify credentials from Spotify Developer Dashboard.

5. Run the app

python app.py



# API Endpoints
/api/editor/add-segment

/api/editor/add-segment-at-position

/api/editor/add-segment-with-shift-option

/api/editor/delete-segment

/api/editor/delete-segment-with-timeline-adjustment

/api/editor/replace-segment

/api/editor/undo

/api/editor/redo

/api/editor/generate-mashup-preview

/api/editor/export-mashup

/api/editor/timeline-positions

/api/editor/used-segments/<song_id>

/preview-original/<song_id>

/api/session-keepalive

