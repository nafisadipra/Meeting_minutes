# 🎙️ Vocalis - Smart Local Meeting Assistant

**Local-first, privacy-centric meeting transcription, speaker diarization, and summarization platform built with Next.js, Flask, and Local LLMs.**

-----

## Features

  - **Real-Time Speaker Diarization** - Live voice profiling and continuous centroid tracking (SpeechBrain ECAPA-TDNN)
  - **True Local-First Privacy** - No cloud APIs. Audio, embeddings, and transcripts stay strictly on your device
  - **Live Voice Enrollment** - 5-second audio capture to dynamically register attendee voice prints
  - **Dynamic Voice Discovery** - Automatically detects new speakers during sessions and prompts to save them permanently
  - **Automated Meeting Minutes** - Generates structured markdown summaries via your own local LLM
  - **High-Accuracy Transcription** - Powered by `faster-whisper` for robust, CPU-friendly speech-to-text
  - **Modern Stack** - Next.js 16, React 19, Tailwind CSS v4, and Framer Motion
  - **Zero Cloud Dependency** - Fully functional offline once models are downloaded

-----

## Quick Start

### 1. Backend (Python/Flask)

```bash
# Navigate to the backend directory
cd backend

# Install Python dependencies
pip install Flask flask-cors python-dotenv pydub numpy scipy SpeechRecognition torch faster-whisper speechbrain requests librosa

# Set your Hugging Face token (required for SpeechBrain)
export HF_TOKEN="your_huggingface_token"

# Start the Flask API server
python app.py
```

### 2. Frontend (Next.js)

```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build
```

*Note: For meeting summarization, ensure you have a local LLM server (e.g., LM Studio, text-generation-webui) running and exposing an OpenAI-compatible API on `http://localhost:1234/v1/chat/completions`.*

-----

## 📋 Requirements

  - **Node.js** 18+
  - **npm** 9+
  - **Python** 3.9+
  - **FFmpeg** (Required for `pydub` audio processing)
  - **Local LLM Server** (Optional, but required for automated meeting minutes)
  - **Hugging Face Account** (To authenticate SpeechBrain model downloads)

-----

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────┐
│                  Next.js Frontend                   │
│         (React 19 + Tailwind v4 + Framer Motion)    │
└──────────────────┬──────────────────────────────────┘
                   │ REST API / JSON Bridge
┌──────────────────┴──────────────────────────────────┐
│                 Flask Backend API                   │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Faster      │  │ SpeechBrain  │  │ Local LLM  │  │
│  │ Whisper     │  │ (ECAPA-TDNN) │  │ Summarizer │  │
│  └─────────────┘  └──────────────┘  └────────────┘  │
│  ┌─────────────────────────────────────────────────┐│
│  │      Smart Speaker Bank (Centroid Tracking)     ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

-----

## 🔐 Privacy & Local AI Guarantees

### What Leaves Your Device?

  - ❌ Your audio recordings
  - ❌ Your meeting transcripts
  - ❌ Your generated minutes
  - ❌ Your speaker voice profiles

### What You Control

  - ✅ **Voice Embeddings** - Stored locally as `.npy` files in the `voice_profiles` directory.
  - ✅ **Transcripts & Minutes** - Generated in memory and written to `minutes.txt`.
  - ✅ **The Models** - You choose which Whisper and LLM models run on your hardware.

### Zero Knowledge

We have **zero knowledge** of your meetings. Because the entire processing pipeline—from the Whisper transcription to the SpeechBrain diarization and the LLM summarization—runs natively on your hardware, there are no external servers to compromise.

-----

## 🛠️ Technology Stack

### Frontend
  - **Next.js 16.2.4** - React framework
  - **React 19** - UI Library
  - **Tailwind CSS v4** - PostCSS styling
  - **Framer Motion** - Fluid animations and layout transitions
  - **React Markdown** - Rendering AI-generated `.md` minutes

### Backend (Python)
  - **Flask** - Lightweight API server
  - **Faster-Whisper** - Optimized CTranslate2 implementation of OpenAI's Whisper
  - **SpeechBrain** - State-of-the-art ECAPA-TDNN speaker embedding extraction
  - **PyDub / Librosa** - Audio stream resampling and manipulation
  - **SciPy / NumPy** - Cosine distance calculations for speaker clustering

-----

## 📝 Roadmap

### v1.0 (Current)
  - [x] Faster-Whisper transcription integration
  - [x] Live voice enrollment (5-second capture)
  - [x] Real-time diarization and Smart Speaker Bank
  - [x] Unrecognized speaker discovery & permanent saving
  - [x] Local LLM map-reduce summarization

### v1.1 (Planned)
  - [ ] Real-time audio waveform visualization
  - [ ] GPU acceleration toggle in the UI
  - [ ] Edit and rename dynamically discovered speakers mid-meeting
  - [ ] Export transcripts to PDF/Docx

### v2.0 (Future)
  - [ ] Multi-microphone support
  - [ ] Integration with calendar APIs (CalDAV) for automatic attendee fetching
  - [ ] Local vector database for semantic search across past meeting minutes

-----

## 🤝 Contributing

We welcome contributions! If you have ideas for improving diarization accuracy or frontend performance, please submit a pull request. 

For major architectural changes, please open an issue first to discuss what you would like to change.

-----

## 📜 License

MIT License - see the LICENSE file for details.

-----

## ⚠️ Disclaimer

While Vocalis is designed to gracefully handle speaker overlapping and ambient noise using continuity windows and dynamic energy thresholding, transcript accuracy and diarization depend heavily on microphone quality and room acoustics. 

For the highest accuracy, we recommend using a dedicated, omnidirectional boundary microphone placed centrally among attendees.
