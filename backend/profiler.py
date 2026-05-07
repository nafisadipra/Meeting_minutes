import os
import numpy as np
import torch
import librosa
from speechbrain.inference.speaker import EncoderClassifier


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # SpeechBrain's EncoderClassifier does not implement device_type for MPS;
    # CPU is the correct target on Apple Silicon for this model.
    return "cpu"


class VoiceProfiler:
    def __init__(self):
        models_root = os.getenv("MODELS_ROOT", "./models")
        model_path = os.getenv(
            "SPEECHBRAIN_MODEL_PATH",
            f"{models_root}/speechbrain/spkrec-ecapa-voxceleb",
        )
        data_root = os.getenv("DATA_ROOT", "./data")
        profiles_sub = os.getenv("VOICE_PROFILES_DIR", "voice_profiles")
        self.profiles_dir = os.path.join(data_root, profiles_sub)
        os.makedirs(self.profiles_dir, exist_ok=True)

        device = _resolve_device()
        print(f"Loading SpeechBrain from {model_path} on {device}")
        self.encoder = EncoderClassifier.from_hparams(
            source=model_path, savedir=model_path, run_opts={"device": device}
        )

    def _extract_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        with torch.no_grad():
            emb = self.encoder.encode_batch(tensor)
        return emb.squeeze().cpu().numpy()

    def enroll_from_audio(self, wav_io, name: str) -> np.ndarray:
        audio_np, _ = librosa.load(wav_io, sr=16000)
        embedding = self._extract_embedding(audio_np)
        path = os.path.join(self.profiles_dir, f"{name}.npy")
        np.save(path, embedding)
        print(f"Saved {name}.npy")
        return embedding

    def load_all_profiles(self) -> dict[str, np.ndarray]:
        profiles: dict[str, np.ndarray] = {}
        if not os.path.exists(self.profiles_dir):
            return profiles
        for fname in os.listdir(self.profiles_dir):
            if fname.endswith(".npy"):
                name = fname[:-4]
                profiles[name] = np.load(os.path.join(self.profiles_dir, fname))
        print(f"Loaded {len(profiles)} profiles")
        return profiles
