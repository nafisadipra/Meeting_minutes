# profiler.py
import os
import io
import torch
import numpy as np
import librosa
from speechbrain.inference.speaker import EncoderClassifier

class VoiceProfiler:
    def __init__(self, profiles_dir="voice_profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        print("Loading SpeechBrain ECAPA-TDNN Profiler Model...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": "cpu"}
        )

    def _extract_embedding(self, audio_np):
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(tensor)
        return embeddings.squeeze().cpu().numpy()

    def enroll_from_audio(self, wav_io, name):
        """Processes an audio stream, extracts the voice print, and saves it to disk."""
        # ECAPA-TDNN expects 16kHz audio
        audio_np, sr = librosa.load(wav_io, sr=16000)
        
        embedding = self._extract_embedding(audio_np)
        
        # Save permanently to disk
        file_path = os.path.join(self.profiles_dir, f"{name}.npy")
        np.save(file_path, embedding)
        print(f"Successfully profiled and saved voice for: {name}")
        
        return embedding

    def load_all_profiles(self):
        """Loads all saved .npy profiles from disk into a dictionary."""
        profiles = {}
        if not os.path.exists(self.profiles_dir):
            return profiles
            
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith(".npy"):
                name = filename.replace(".npy", "")
                filepath = os.path.join(self.profiles_dir, filename)
                profiles[name] = np.load(filepath)
                
        print(f"Loaded {len(profiles)} pre-enrolled profiles from disk.")
        return profiles