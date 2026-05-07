import threading
import queue
import time
import os
import numpy as np
import speech_recognition as sr
import torch
from scipy.spatial.distance import cosine as cos_dist
from faster_whisper import WhisperModel
from profiler import VoiceProfiler
from speaker_bank import SmartSpeakerBank
from summarizer import OllamaSummarizer


class MeetingAssistant:
    MIN_RELIABLE_EMBEDDING_SAMPLES = 48000
    MIN_USABLE_EMBEDDING_SAMPLES = 24000
    SHORT_AUDIO_INHERIT_SAMPLES = 16000
    CONTINUITY_WINDOW_SECONDS = 5.0
    STRONG_CONTINUITY_WINDOW_SECONDS = 3.0
    TRANSITION_DISTANCE_JUMP = 0.12

    def __init__(self, enrolled_profiles=None, expected_attendees=None):
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_recording = False
        self.last_speaker = "Unknown"
        self.last_speaker_time = 0.0
        self.last_confidence = 0.0
        self.speaker_distance_history: list = []

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.2
        self.recognizer.non_speaking_duration = 0.6

        self._transcript: list[str] = []
        self._transcript_lock = threading.Lock()

        self.bank = SmartSpeakerBank(
            anchor_profiles=enrolled_profiles,
            expected_attendees=expected_attendees,
            use_anchors_as_hints_only=True,
        )

        # CTranslate2 (faster-whisper) does not support MPS/Metal; CPU with int8
        # gives the best throughput on Apple Silicon via ARM NEON.
        whisper_path = os.getenv("WHISPER_MODEL_PATH", "./models/whisper/base.en")
        device = os.getenv(
            "WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        compute_env = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
        if compute_env == "auto":
            compute = "float16" if device == "cuda" else "int8"
        else:
            compute = compute_env

        if not os.path.isdir(whisper_path):
            raise FileNotFoundError(
                f"Whisper model not found at {whisper_path}. "
                "Run 'python download_models.py' first."
            )

        # cpu_threads: tune for Apple Silicon performance + efficiency core split.
        cpu_threads = min(os.cpu_count() or 4, 8)
        print(f"Loading Whisper from {whisper_path} on {device} ({compute})")
        self.whisper = WhisperModel(
            whisper_path,
            device=device,
            compute_type=compute,
            local_files_only=True,
            cpu_threads=cpu_threads,
        )

        self.profiler = VoiceProfiler()
        self.summarizer = OllamaSummarizer()

    @property
    def full_transcript(self) -> list[str]:
        with self._transcript_lock:
            return list(self._transcript)

    @full_transcript.setter
    def full_transcript(self, value: list[str]) -> None:
        with self._transcript_lock:
            self._transcript = value

    def _extract_embedding(self, audio_np: np.ndarray) -> np.ndarray:
        return self.profiler._extract_embedding(audio_np)

    def _audio_listener_thread(self) -> None:
        with sr.Microphone(sample_rate=16000) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("Listening...")
            while self.is_recording:
                try:
                    audio = self.recognizer.listen(
                        source, timeout=1, phrase_time_limit=15
                    )
                    self.audio_queue.put((time.time(), audio))
                except sr.WaitTimeoutError:
                    continue

    def _is_continuity_likely(self, current_time: float, strong: bool = False) -> bool:
        if self.last_speaker == "Unknown":
            return False
        elapsed = current_time - self.last_speaker_time
        window = (
            self.STRONG_CONTINUITY_WINDOW_SECONDS
            if strong
            else self.CONTINUITY_WINDOW_SECONDS
        )
        return elapsed <= window

    def _is_transition_detected(self, vector: np.ndarray, current_time: float) -> bool:
        if (
            self.last_speaker == "Unknown"
            or self.last_speaker not in self.bank.centroids
        ):
            return False
        current_dist = cos_dist(
            vector.flatten(), self.bank.centroids[self.last_speaker]
        )
        recent = [
            d
            for (t, spk, d) in self.speaker_distance_history
            if spk == self.last_speaker
            and (current_time - t) <= self.CONTINUITY_WINDOW_SECONDS
        ]
        if len(recent) < 2:
            return current_dist > 0.55
        avg_dist = float(np.mean(recent))
        return (current_dist - avg_dist) > self.TRANSITION_DISTANCE_JUMP

    def _record_distance(
        self, speaker: str, vector: np.ndarray, current_time: float
    ) -> None:
        if speaker not in self.bank.centroids:
            return
        dist = cos_dist(vector.flatten(), self.bank.centroids[speaker])
        self.speaker_distance_history.append((current_time, speaker, dist))
        self.speaker_distance_history = self.speaker_distance_history[-30:]

    def _decide_speaker(
        self, vector, n_samples: int, current_time: float
    ) -> tuple[str, float]:
        if vector is None:
            if self._is_continuity_likely(current_time):
                return self.last_speaker, 0.0
            return "Unknown", 0.0

        reliable = n_samples >= self.MIN_RELIABLE_EMBEDDING_SAMPLES
        usable = n_samples >= self.MIN_USABLE_EMBEDDING_SAMPLES
        transition = self._is_transition_detected(vector, current_time)
        prefer = None
        if self._is_continuity_likely(current_time) and not transition:
            prefer = self.last_speaker

        if not reliable:
            if (
                self._is_continuity_likely(current_time, strong=True)
                and not transition
            ):
                proposed, conf = self.bank.score_against_known(
                    vector, self.last_speaker
                )
                # Stay with current speaker only when the voice actually matches.
                # A low confidence means a different voice — fall through to detection.
                if proposed == self.last_speaker and conf > 0.35:
                    if conf > 0.4:
                        self.bank.update_centroid(
                            self.last_speaker, vector, n_samples=n_samples
                        )
                    return self.last_speaker, conf

            # Usable-length segments (≥1.5s) can introduce a new speaker.
            # Very short clips (<1.5s) stay assigned to the nearest known centroid.
            label, conf = self.bank.process_segment(
                vector,
                allow_new_speaker=usable,
                prefer_label=prefer,
                n_samples=n_samples,
            )
            return label, conf

        label, conf = self.bank.process_segment(
            vector, allow_new_speaker=True, prefer_label=prefer, n_samples=n_samples
        )
        return label, conf

    def _append_to_transcript(self, speaker: str, text: str) -> None:
        with self._transcript_lock:
            if self._transcript and self._transcript[-1].startswith(f"[{speaker}]:"):
                self._transcript[-1] += " " + text
            else:
                self._transcript.append(f"[{speaker}]: {text}")

    def _processor_thread(self) -> None:
        while self.is_recording or not self.audio_queue.empty():
            try:
                current_time, audio_window = self.audio_queue.get(timeout=1)
                audio_np = (
                    np.frombuffer(audio_window.get_raw_data(), dtype=np.int16).astype(
                        np.float32
                    )
                    / 32768.0
                )

                segments, _ = self.whisper.transcribe(
                    audio_np,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = "".join(seg.text for seg in segments).strip()
                if not text:
                    continue

                n_samples = len(audio_np)
                vector = None
                if n_samples >= self.SHORT_AUDIO_INHERIT_SAMPLES:
                    try:
                        vector = self._extract_embedding(audio_np)
                    except Exception as e:
                        print(f"Embedding error: {e}")

                speaker, confidence = self._decide_speaker(
                    vector, n_samples, current_time
                )
                if vector is not None:
                    self._record_distance(speaker, vector, current_time)

                self.last_speaker = speaker
                self.last_speaker_time = current_time
                self.last_confidence = confidence

                self._append_to_transcript(speaker, text)
                print(f"[{speaker} | conf={confidence:.2f}]: {text}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Process error: {e}")

    def start(self) -> None:
        self.is_recording = True
        self.listener = threading.Thread(
            target=self._audio_listener_thread, daemon=True
        )
        self.processor = threading.Thread(
            target=self._processor_thread, daemon=True
        )
        self.listener.start()
        self.processor.start()

    def stop(self) -> dict:
        print("Stopping recording, flushing queue...")
        self.is_recording = False
        self.listener.join()
        self.processor.join()

        output_path = os.path.join(
            os.getenv("DATA_ROOT", "./data"), os.getenv("MINUTES_FILE", "minutes.txt")
        )
        snapshot = self.full_transcript
        threading.Thread(
            target=self.summarizer.generate_minutes,
            args=(snapshot, output_path),
            daemon=True,
        ).start()
        return self.bank.get_finalized_centroids()
