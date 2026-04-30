import os
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

# Import your modular Meeting Assistant
from assistant import MeetingAssistant

# Load the hidden variables from your .env file
load_dotenv() 

app = Flask(__name__)
# Enable CORS so Next.js (port 3000) can communicate with Flask (port 5000)
CORS(app) 

# Fetch token securely
HF_TOKEN = os.getenv("HF_TOKEN")

# Global reference to hold the active meeting session
current_assistant = None

@app.route('/api/start', methods=['POST'])
def start_meeting():
    global current_assistant
    
    # Get the attendee names sent from the Next.js frontend
    data = request.json or {}
    attendees_list = data.get('attendees', [])
    
    # Delete the old minutes file before starting a new meeting
    if os.path.exists("minutes.txt"):
        os.remove("minutes.txt")
    
    # If a meeting isn't already running, start a new one
    if current_assistant is None or not current_assistant.is_recording:
        # Initialize a fresh assistant with the provided names
        current_assistant = MeetingAssistant(hf_token=HF_TOKEN, expected_attendees=attendees_list)
        current_assistant.start()
        return jsonify({"status": "Recording started", "is_recording": True}), 200
        
    return jsonify({"status": "Already recording", "is_recording": True}), 200

@app.route('/api/stop', methods=['POST'])
def stop_meeting():
    global current_assistant
    
    if current_assistant and current_assistant.is_recording:
        # Run the stop/summarize method in a background thread
        # This prevents the frontend button from freezing while LM Studio generates text
        threading.Thread(target=current_assistant.stop).start()
        return jsonify({"status": "Stopping and summarizing...", "is_recording": False}), 200
        
    return jsonify({"status": "Not currently recording", "is_recording": False}), 200

@app.route('/api/clear', methods=['POST'])
def clear_meeting():
    global current_assistant
    
    # Only allow clearing if we aren't actively recording
    if current_assistant and not current_assistant.is_recording:
        current_assistant.full_transcript = [] # Wipe the transcript memory
        
    if os.path.exists("minutes.txt"):
        os.remove("minutes.txt") # Wipe the file
        
    return jsonify({"status": "Cleared"}), 200

@app.route('/api/transcript', methods=['GET'])
def get_transcript():
    global current_assistant
    
    # Send the live array of transcribed sentences to the frontend
    if current_assistant:
        return jsonify({
            "is_recording": current_assistant.is_recording,
            "transcript": current_assistant.full_transcript
        }), 200
        
    return jsonify({"is_recording": False, "transcript": []}), 200

@app.route('/api/minutes', methods=['GET'])
def get_minutes():
    # Check if the LM Studio summarizer has finished writing the file
    if os.path.exists("minutes.txt"):
        with open("minutes.txt", "r") as f:
            content = f.read()
        return jsonify({"minutes": content}), 200
        
    return jsonify({"minutes": None}), 200

if __name__ == '__main__':
    # Run the Flask server on port 5000
    app.run(debug=True, port=5000, use_reloader=False)