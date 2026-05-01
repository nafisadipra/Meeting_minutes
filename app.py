import os
import io
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from pydub import AudioSegment

from assistant import MeetingAssistant
from profiler import VoiceProfiler

load_dotenv()
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
current_assistant = None

profiler = VoiceProfiler()
enrolled_profiles = profiler.load_all_profiles()


def enroll_voice():
    global enrolled_profiles
    if 'audio' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Missing audio or name"}), 400

    audio_file = request.files['audio']
    person_name = request.form['name']

    try:
        audio_data = io.BytesIO(audio_file.read())
        audio_segment = AudioSegment.from_file(audio_data, format="webm")

        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # Let profiler.py do the heavy lifting
        embedding = profiler.enroll_from_audio(wav_io, person_name)
        
        # Update our active dictionary
        enrolled_profiles[person_name] = embedding
        
        return jsonify({"status": "Profile saved!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_meeting():
    global current_assistant, enrolled_profiles

    data = request.json or {}
    attendees_list = data.get('attendees', [])

    if os.path.exists("minutes.txt"):
        os.remove("minutes.txt")

    if current_assistant is None or not current_assistant.is_recording:
        current_assistant = MeetingAssistant(
            hf_token=HF_TOKEN,
            enrolled_profiles=enrolled_profiles,
            expected_attendees=attendees_list,
        )
        current_assistant.start()
        return jsonify({"status": "Recording started", "is_recording": True}), 200

    return jsonify({"status": "Already recording", "is_recording": True}), 200

@app.route('/api/stop', methods=['POST'])
def stop_meeting():
    global current_assistant, enrolled_profiles

    if current_assistant and current_assistant.is_recording:
        discovered_centroids = current_assistant.stop()

        new_profiles = {}
        updated_profiles = {}

        for name, vector in discovered_centroids.items():
            vector_list = np.asarray(vector).flatten().tolist()
            if name not in enrolled_profiles:
                new_profiles[name] = vector_list
            else:
                updated_profiles[name] = vector_list

        return jsonify({
            "status": "Stopping and summarizing...",
            "is_recording": False,
            "new_profiles_discovered": new_profiles,
            "updated_profiles_available": updated_profiles,
        }), 200

    return jsonify({"status": "Not recording", "is_recording": False}), 200

@app.route('/api/save_discovered_profiles', methods=['POST'])
def save_discovered_profiles():
    """Saves voices discovered during the meeting permanently to disk."""
    global enrolled_profiles, profiler
    data = request.json or {}
    profiles_to_save = data.get('profiles', {})

    for name, vector_list in profiles_to_save.items():
        embedding = np.array(vector_list)
        enrolled_profiles[name] = embedding
        
        # Save to disk using profiler so it persists after reboot
        file_path = os.path.join(profiler.profiles_dir, f"{name}.npy")
        np.save(file_path, embedding)
        print(f"Permanently saved discovered profile for: {name}")

    return jsonify({"status": "Profiles saved successfully!"}), 200

@app.route('/api/clear', methods=['POST'])
def clear_meeting():
    global current_assistant, enrolled_profiles
    if current_assistant and not current_assistant.is_recording:
        current_assistant.full_transcript = []
    if os.path.exists("minutes.txt"):
        os.remove("minutes.txt")
    return jsonify({"status": "Cleared"}), 200

@app.route('/api/transcript', methods=['GET'])
def get_transcript():
    global current_assistant
    if current_assistant:
        return jsonify({
            "is_recording": current_assistant.is_recording,
            "transcript": current_assistant.full_transcript,
        }), 200
    return jsonify({"is_recording": False, "transcript": []}), 200

@app.route('/api/minutes', methods=['GET'])
def get_minutes():
    if os.path.exists("minutes.txt"):
        with open("minutes.txt", "r") as f:
            content = f.read()
        return jsonify({"minutes": content}), 200
    return jsonify({"minutes": None}), 200

@app.route('/api/list_profiles', methods=['GET'])
def list_profiles():
    global enrolled_profiles
    return jsonify({
        "enrolled_profiles": list(enrolled_profiles.keys()),
        "count": len(enrolled_profiles),
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)