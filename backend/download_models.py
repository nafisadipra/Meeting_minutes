#!/usr/bin/env python3
"""Download models to local folders – run once with internet.
Outputs:
  - SpeechBrain: ./models/speechbrain/spkrec-ecapa-voxceleb/
  - Whisper:     ./models/whisper/base.en/  (contains model.bin, config.json, etc.)
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "./models"))
MODELS_ROOT.mkdir(exist_ok=True, parents=True)

# ---------- 1. SpeechBrain ----------
print("Downloading SpeechBrain model...")
from speechbrain.inference.speaker import EncoderClassifier

speechbrain_path = MODELS_ROOT / "speechbrain" / "spkrec-ecapa-voxceleb"
EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(speechbrain_path),
)
print(f"SpeechBrain saved to {speechbrain_path}")

# ---------- 2. Whisper (faster-whisper) ----------
print("Downloading Whisper model...")
temp_root = MODELS_ROOT / "_whisper_cache"
temp_root.mkdir(exist_ok=True)

from faster_whisper import WhisperModel

# Force download into temp_root
model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
    download_root=str(temp_root),
)
# Transcribe dummy audio to ensure download is complete
import numpy as np

dummy = np.zeros(16000, dtype=np.float32)
_ = model.transcribe(dummy, language="en")

# Find the actual snapshot folder (contains model.bin, config.json, etc.)
cache_dir = temp_root / "models--Systran--faster-whisper-base.en"
if not cache_dir.exists():
    # Fallback for different naming
    possible = list(temp_root.glob("models--*--faster-whisper-base.en"))
    if possible:
        cache_dir = possible[0]
    else:
        raise FileNotFoundError("Could not locate downloaded whisper cache folder")

snapshots_dir = cache_dir / "snapshots"
if not snapshots_dir.exists():
    raise FileNotFoundError(f"Snapshots folder not found in {cache_dir}")

snapshot_dirs = list(snapshots_dir.iterdir())
if not snapshot_dirs:
    raise FileNotFoundError("No snapshot found in whisper cache")
snapshot = snapshot_dirs[0]

# Destination: MODELS_ROOT / whisper / base.en
dest = MODELS_ROOT / "whisper" / "base.en"
if dest.exists():
    print(f"Destination {dest} already exists, removing...")
    shutil.rmtree(dest)
dest.mkdir(parents=True)

# Copy all files from snapshot to destination
print(f"Copying Whisper model from {snapshot} to {dest}")
for item in snapshot.iterdir():
    if item.is_file():
        shutil.copy2(item, dest / item.name)
    elif item.is_dir():
        shutil.copytree(item, dest / item.name, dirs_exist_ok=True)

# Clean up temporary cache (optional)
shutil.rmtree(temp_root)

print(f"Whisper model ready at {dest}")
print("\nAll models downloaded. You can now disconnect from the internet.")
