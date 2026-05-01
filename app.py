import os
import threading
import io
import numpy as np
import librosa
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from pydub import AudioSegment

# Import your modular Meeting Assistant
from assistant import MeetingAssistant

# Load the hidden variables from your .env file
load_dotenv() 

app = Flask(__name__)
# Enable CORS so Next.js (port 3000) can communicate with Flask (port 5000)
CORS(app) 

# Fetch token securely
HF_TOKEN = os.getenv("HF_TOKEN")

# Global references
current_assistant = None
enrolled_profiles = {} # Stores the voice prints before the meeting starts

print("Loading Global Pyannote Model for Voice Profiling...")
pyannote_model = Model.from_pretrained("pyannote/embedding", token=HF_TOKEN)
global_inference = Inference(pyannote_model, window="whole")


@app.route('/api/enroll_voice', methods=['POST'])
def enroll_voice():
    """Receives a 5-second WebM audio clip, extracts the voice print, and saves it."""
    global enrolled_profiles

    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Missing audio or name"}), 400

    audio_file = request.files['audio']
    person_name = request.form['name']

    try:
        # Convert the browser's WebM file into a 16kHz numpy array for the AI
        audio_data = io.BytesIO(audio_file.read())
        audio_segment = AudioSegment.from_file(audio_data, format="webm")
        
        # Export to raw wav in memory
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Load with librosa at exactly 16kHz
        audio_np, sr = librosa.load(wav_io, sr=16000)
        
        # Extract the mathematical voice print
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        embedding = global_inference({"waveform": tensor, "sample_rate": 16000})
        
        # Save the profile in our global dictionary
        enrolled_profiles[person_name] = embedding.flatten()
        print(f"Successfully profiled voice for: {person_name}")

        return jsonify({"status": "Profile saved!"}), 200
    except Exception as e:
        print(f"Error profiling voice: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/start', methods=['POST'])
def start_meeting():
    global current_assistant, enrolled_profiles
    
    # Delete the old minutes file before starting a new meeting
    if os.path.exists("minutes.txt"):
        os.remove("minutes.txt")
    
    # If a meeting isn't already running, start a new one
    if current_assistant is None or not current_assistant.is_recording:
        # Inject our pre-recorded voice profiles directly into the assistant
        current_assistant = MeetingAssistant(hf_token=HF_TOKEN, enrolled_profiles=enrolled_profiles)
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
    global current_assistant, enrolled_profiles
    
    # Only allow clearing if we aren't actively recording
    if current_assistant and not current_assistant.is_recording:
        current_assistant.full_transcript = [] # Wipe the transcript memory
        
    if os.path.exists("minutes.txt"):
        os.remove("minutes.txt") # Wipe the file

    # Optional: Clear enrolled profiles if you want a totally fresh slate
    enrolled_profiles.clear()
        
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