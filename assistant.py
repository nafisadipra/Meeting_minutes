import threading
import queue
import numpy as np
import speech_recognition as sr
import torch
from pyannote.audio import Model, Inference
from faster_whisper import WhisperModel

# Import your custom modules
from speaker_bank import SmartSpeakerBank
from summarizer import LocalLLMSummarizer

class MeetingAssistant:
    def __init__(self, hf_token, expected_attendees=None): 
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.last_speaker = "Unknown"
        
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8 # Slightly longer pause so Whisper can grab full sentences
        
        self.full_transcript = []
        
        self.bank = SmartSpeakerBank(attendees=expected_attendees)
        self.summarizer = LocalLLMSummarizer()
        
        print("Loading Whisper Speech-to-Text Model...")
        # "base.en" is incredibly fast and highly accurate for English
        self.whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
        
        print("Loading Pyannote Embedding Model...")
        model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        self.inference = Inference(model, window="whole")

    def _extract_embedding(self, audio_np, sample_rate):
        tensor = torch.from_numpy(audio_np).unsqueeze(0) 
        return self.inference({"waveform": tensor, "sample_rate": sample_rate})

    def _audio_listener_thread(self):
        # Forcing 16kHz prevents audio stretching/distortion for the AI models
        with sr.Microphone(sample_rate=16000) as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening... (Press Ctrl+C to stop)")
            
            while self.is_recording:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    self.audio_queue.put(audio) 
                except sr.WaitTimeoutError:
                    continue

    def _processor_thread(self):
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_window = self.audio_queue.get(timeout=1)
                
                audio_np = np.frombuffer(audio_window.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                
                # 1. Transcribe the audio using Local Whisper first
                segments, _ = self.whisper.transcribe(audio_np, beam_size=5)
                text = "".join([segment.text for segment in segments]).strip()
                
                if not text:
                    continue 

                # 2. Smart Speaker Routing
                # If the audio is less than 2 seconds (32000 samples) AND we know who was talking,
                # assume they are just continuing their thought (e.g., saying "Yeah", "So", "You").
                if len(audio_np) < 32000 and self.last_speaker != "Unknown":
                    speaker_label = self.last_speaker
                else:
                    # For longer, complex clips, run the heavy Pyannote voice matching
                    vector = self._extract_embedding(audio_np, audio_window.sample_rate)
                    speaker_label = self.bank.process_segment(vector)
                    
                    # Remember this person for the next short clip
                    self.last_speaker = speaker_label 
                
                log_entry = f"[{speaker_label}]: {text}"
                print(log_entry) 
                self.full_transcript.append(log_entry)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Audio Processing Error]: {e}")
                continue

    def start(self):
        self.is_recording = True
        self.listener = threading.Thread(target=self._audio_listener_thread)
        self.processor = threading.Thread(target=self._processor_thread)
        self.listener.start()
        self.processor.start()

    def stop(self):
        print("\nStopping recording... Processing remaining audio in queue.")
        self.is_recording = False
        self.listener.join()
        self.processor.join()
        
        self.summarizer.generate_minutes(self.full_transcript)