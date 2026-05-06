#!/usr/bin/env python3
"""
Simple interactive CLI for Meeting Assistant.
Runs offline, uses GPU if available, requires Ollama for summaries.
"""

import os
import sys
import time
import subprocess
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Global state
current_assistant = None
last_discovered = {}  # {name: embedding} from last meeting


# ------------------------------------------------------------
# Helper functions (same as before)
# ------------------------------------------------------------
def get_profiler():
    from profiler import VoiceProfiler

    return VoiceProfiler()


def get_assistant(enrolled, attendees=None):
    from assistant import MeetingAssistant

    return MeetingAssistant(enrolled_profiles=enrolled, expected_attendees=attendees)


def get_minutes_path():
    return os.path.join(
        os.getenv("DATA_ROOT", "./data"), os.getenv("MINUTES_FILE", "minutes.txt")
    )


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# ------------------------------------------------------------
# Command implementations
# ------------------------------------------------------------
def download_models():
    print("Downloading models (internet required)...")
    subprocess.run([sys.executable, "download_models.py"], check=False)
    input("\nPress Enter to continue...")


def enroll_voice():
    import speech_recognition as sr
    import io

    name = input("Enter name for the new speaker: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    print(f"\nEnrolling '{name}'. Speak for 5 seconds...")
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("No speech detected. Enrollment failed.")
            return

    wav_io = io.BytesIO(audio.get_wav_data())
    profiler = get_profiler()
    profiler.enroll_from_audio(wav_io, name)
    print(f"Enrolled '{name}' successfully.")
    input("\nPress Enter to continue...")


def start_meeting():
    global current_assistant, last_discovered
    if current_assistant and current_assistant.is_recording:
        print("Already recording. Stop the current meeting first.")
        input("\nPress Enter to continue...")
        return

    # Ask for expected attendees (like frontend's input field)
    attendees_input = input("Expected attendees (comma separated, optional): ").strip()
    attendees = (
        [a.strip() for a in attendees_input.split(",") if a.strip()]
        if attendees_input
        else None
    )

    profiler = get_profiler()
    enrolled = profiler.load_all_profiles()

    print("\nStarting meeting. Live transcript will appear below.")
    print("Press Ctrl+C to stop the meeting.\n")
    current_assistant = get_assistant(enrolled, attendees)
    current_assistant.start()
    current_assistant._printed_lines = 0

    try:
        while current_assistant.is_recording:
            new_lines = (
                len(current_assistant.full_transcript)
                - current_assistant._printed_lines
            )
            if new_lines > 0:
                for line in current_assistant.full_transcript[-new_lines:]:
                    print(line)
                current_assistant._printed_lines = len(
                    current_assistant.full_transcript
                )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n\nStopping meeting...")
        centroids = current_assistant.stop()
        current_assistant = None

        # Discover new speakers
        new_speakers = {
            name: vec for name, vec in centroids.items() if name not in enrolled
        }
        if new_speakers:
            print("\nNew speakers discovered during this meeting:")
            for name in new_speakers:
                print(f"  - {name}")
            ans = input("Save them permanently? (y/n): ").strip().lower()
            if ans == "y":
                for name, vec in new_speakers.items():
                    path = os.path.join(profiler.profiles_dir, f"{name}.npy")
                    np.save(path, vec)
                    print(f"Saved {name}")
                last_discovered = {}
            else:
                last_discovered = new_speakers
        else:
            print("No new speakers discovered.")

        minutes_path = get_minutes_path()
        if os.path.exists(minutes_path):
            print(f"\nMeeting minutes saved to: {minutes_path}")
        else:
            print(
                "\nMinutes may still be generating (Ollama running). Use option 6 to check later."
            )
        input("\nPress Enter to continue...")


def show_status():
    if current_assistant and current_assistant.is_recording:
        print("Recording in progress.")
    else:
        print("Not recording.")
    input("\nPress Enter to continue...")


def show_transcript():
    if current_assistant and current_assistant.full_transcript:
        print("\n--- Transcript ---")
        for line in current_assistant.full_transcript:
            print(line)
    else:
        print("No transcript available.")
    input("\nPress Enter to continue...")


def show_minutes():
    minutes_path = get_minutes_path()
    if os.path.exists(minutes_path):
        print("\n--- Meeting Minutes ---")
        with open(minutes_path, "r") as f:
            print(f.read())
    else:
        print("No minutes file found. Run a meeting first.")
    input("\nPress Enter to continue...")


def list_profiles():
    profiler = get_profiler()
    profiles = profiler.load_all_profiles()
    if profiles:
        print("\nEnrolled profiles:", ", ".join(profiles.keys()))
    else:
        print("No enrolled profiles.")
    input("\nPress Enter to continue...")


def clear_session():
    global current_assistant, last_discovered
    if current_assistant:
        current_assistant.full_transcript = []
    minutes_path = get_minutes_path()
    if os.path.exists(minutes_path):
        os.remove(minutes_path)
    last_discovered = {}
    print("Cleared transcript and minutes.")
    input("\nPress Enter to continue...")


# ------------------------------------------------------------
# Main menu loop
# ------------------------------------------------------------
def main():
    while True:
        clear_screen()
        print("\n" + "=" * 50)
        print("         MEETING ASSISTANT CLI")
        print("=" * 50)
        print("1. Download models (run once, needs internet)")
        print("2. Enroll a new voice")
        print("3. Start a meeting (live transcript)")
        print("4. Show recording status")
        print("5. Show current transcript")
        print("6. Show generated minutes")
        print("7. List enrolled profiles")
        print("8. Clear transcript & minutes")
        print("9. Exit")
        print("-" * 50)

        choice = input("Choose an option (1-9): ").strip()

        if choice == "1":
            download_models()
        elif choice == "2":
            enroll_voice()
        elif choice == "3":
            start_meeting()
        elif choice == "4":
            show_status()
        elif choice == "5":
            show_transcript()
        elif choice == "6":
            show_minutes()
        elif choice == "7":
            list_profiles()
        elif choice == "8":
            clear_session()
        elif choice == "9":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option. Press Enter to try again.")
            input()


if __name__ == "__main__":
    main()
