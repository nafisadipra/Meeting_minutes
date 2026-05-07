import os
import io
import threading
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from dotenv import load_dotenv
from pydub import AudioSegment

from assistant import MeetingAssistant
from profiler import VoiceProfiler

load_dotenv()
app = Flask(__name__)
CORS(app)

_state_lock = threading.Lock()
current_assistant: MeetingAssistant | None = None
profiler = VoiceProfiler()
enrolled_profiles = profiler.load_all_profiles()


@app.route("/api/enroll", methods=["POST"])
def enroll_voice() -> tuple[Response, int]:
    global enrolled_profiles
    if "audio" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing audio or name"}), 400

    audio_file = request.files["audio"]
    person_name = request.form["name"].strip()
    if not person_name:
        return jsonify({"error": "Name cannot be empty"}), 400

    try:
        audio_data = io.BytesIO(audio_file.read())
        audio_segment = AudioSegment.from_file(audio_data, format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        embedding = profiler.enroll_from_audio(wav_io, person_name)
        with _state_lock:
            enrolled_profiles[person_name] = embedding
        return jsonify({"status": "Profile saved!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def start_meeting() -> tuple[Response, int]:
    global current_assistant, enrolled_profiles
    data = request.json or {}
    attendees_list = data.get("attendees", [])

    minutes_path = os.path.join(
        os.getenv("DATA_ROOT", "./data"), os.getenv("MINUTES_FILE", "minutes.txt")
    )
    if os.path.exists(minutes_path):
        os.remove(minutes_path)

    with _state_lock:
        if current_assistant is None or not current_assistant.is_recording:
            current_assistant = MeetingAssistant(
                enrolled_profiles=enrolled_profiles,
                expected_attendees=attendees_list,
            )
            current_assistant.start()
            return jsonify({"status": "Recording started", "is_recording": True}), 200
    return jsonify({"status": "Already recording", "is_recording": True}), 200


@app.route("/api/stop", methods=["POST"])
def stop_meeting() -> tuple[Response, int]:
    global current_assistant, enrolled_profiles
    with _state_lock:
        assistant = current_assistant

    if assistant and assistant.is_recording:
        try:
            discovered_centroids = assistant.stop()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        new_profiles: dict = {}
        updated_profiles: dict = {}
        with _state_lock:
            for name, vector in discovered_centroids.items():
                vec_list = np.asarray(vector).flatten().tolist()
                if name not in enrolled_profiles:
                    new_profiles[name] = vec_list
                else:
                    updated_profiles[name] = vec_list

        return (
            jsonify(
                {
                    "status": "Stopping and summarizing...",
                    "is_recording": False,
                    "new_profiles_discovered": new_profiles,
                    "updated_profiles_available": updated_profiles,
                }
            ),
            200,
        )
    return jsonify({"status": "Not recording", "is_recording": False}), 200


@app.route("/api/save_discovered_profiles", methods=["POST"])
def save_discovered_profiles() -> tuple[Response, int]:
    global enrolled_profiles
    data = request.json or {}
    profiles_to_save = data.get("profiles", {})
    for name, vector_list in profiles_to_save.items():
        embedding = np.array(vector_list)
        file_path = os.path.join(profiler.profiles_dir, f"{name}.npy")
        np.save(file_path, embedding)
        with _state_lock:
            enrolled_profiles[name] = embedding
        print(f"Saved discovered profile: {name}")
    return jsonify({"status": "Profiles saved successfully!"}), 200


@app.route("/api/clear", methods=["POST"])
def clear_meeting() -> tuple[Response, int]:
    global current_assistant
    with _state_lock:
        if current_assistant and not current_assistant.is_recording:
            current_assistant.full_transcript = []
    minutes_path = os.path.join(
        os.getenv("DATA_ROOT", "./data"), os.getenv("MINUTES_FILE", "minutes.txt")
    )
    if os.path.exists(minutes_path):
        os.remove(minutes_path)
    return jsonify({"status": "Cleared"}), 200


@app.route("/api/transcript", methods=["GET"])
def get_transcript() -> tuple[Response, int]:
    with _state_lock:
        assistant = current_assistant
    if assistant:
        return (
            jsonify(
                {
                    "is_recording": assistant.is_recording,
                    "transcript": assistant.full_transcript,
                }
            ),
            200,
        )
    return jsonify({"is_recording": False, "transcript": []}), 200


@app.route("/api/minutes", methods=["GET"])
def get_minutes() -> tuple[Response, int]:
    minutes_path = os.path.join(
        os.getenv("DATA_ROOT", "./data"), os.getenv("MINUTES_FILE", "minutes.txt")
    )
    if os.path.exists(minutes_path):
        with open(minutes_path, "r") as f:
            content = f.read()
        return jsonify({"minutes": content}), 200
    return jsonify({"minutes": None}), 200


@app.route("/api/list_profiles", methods=["GET"])
def list_profiles() -> tuple[Response, int]:
    with _state_lock:
        names = list(enrolled_profiles.keys())
    return jsonify({"enrolled_profiles": names, "count": len(names)}), 200


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug, port=port, use_reloader=False)
