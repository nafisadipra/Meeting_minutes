import threading
import queue
import numpy as np
import speech_recognition as sr
import torch
from pyannote.audio import Model, Inference

# Import your custom modules
from speaker_bank import SmartSpeakerBank
from summarizer import LocalLLMSummarizer

class MeetingAssistant:
    def __init__(self, hf_token, expected_attendees=None): 
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recognizer = sr.Recognizer()
        self.full_transcript = []
        
        # Pass the attendees down to the bank
        self.bank = SmartSpeakerBank(attendees=expected_attendees)
        self.summarizer = LocalLLMSummarizer()
        
        print("Loading Pyannote Embedding Model...")
        model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        self.inference = Inference(model, window="whole")

    def _extract_embedding(self, audio_data):
        audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_np).unsqueeze(0) 
        return self.inference({"waveform": tensor, "sample_rate": audio_data.sample_rate})

    def _audio_listener_thread(self):
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening... (Press Ctrl+C to stop)")
            
            while self.is_recording:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    self.audio_queue.put(audio) 
                except sr.WaitTimeoutError:
                    continue

    def _processor_thread(self):
        while self.is_recording or not self.audio_queue.empty():
            try:
                audio_window = self.audio_queue.get(timeout=1)
                
                try:
                    text = self.recognizer.recognize_google(audio_window)
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"API Error: {e}")
                    continue

                vector = self._extract_embedding(audio_window)
                speaker_label = self.bank.process_segment(vector)
                
                log_entry = f"[{speaker_label}]: {text}"
                print(log_entry) 
                self.full_transcript.append(log_entry)
                
            except queue.Empty:
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
        
        # Delegate summarization to the modular class
        self.summarizer.generate_minutes(self.full_transcript)