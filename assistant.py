# assistant.py  -- FINAL CLEAN VERSION
import threading
import queue
import time
import numpy as np
import speech_recognition as sr
import torch
from scipy.spatial.distance import cosine as cos_dist
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier

from speaker_bank import SmartSpeakerBank
from summarizer import LocalLLMSummarizer


class MeetingAssistant:

    MIN_RELIABLE_EMBEDDING_SAMPLES = 48000
    MIN_USABLE_EMBEDDING_SAMPLES = 24000
    SHORT_AUDIO_INHERIT_SAMPLES = 16000

    CONTINUITY_WINDOW_SECONDS = 5.0
    STRONG_CONTINUITY_WINDOW_SECONDS = 3.0
    TRANSITION_DISTANCE_JUMP = 0.12

    def __init__(self, hf_token, enrolled_profiles=None, expected_attendees=None):
        self.audio_queue = queue.Queue()
        self.is_recording = False

        self.last_speaker = "Unknown"
        self.last_speaker_time = 0.0
        self.last_confidence = 0.0

        # Rolling history of (timestamp, speaker, distance) for transition detection
        self.speaker_distance_history = []

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.2
        self.recognizer.non_speaking_duration = 0.6

        self.full_transcript = []

        self.bank = SmartSpeakerBank(
            anchor_profiles=enrolled_profiles,
            expected_attendees=expected_attendees,
            use_anchors_as_hints_only=True,
        )
        self.summarizer = LocalLLMSummarizer()

        print("Loading Whisper Speech-to-Text Model...")
        self.whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

        print("Loading SpeechBrain ECAPA-TDNN Embedding Model...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"}
        )

    def _extract_embedding(self, audio_np):
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(tensor)
        return embeddings.squeeze().cpu().numpy()

    def _audio_listener_thread(self):
        with sr.Microphone(sample_rate=16000) as source:
            print("Adjusting for ambient noise...")
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

    def _is_continuity_likely(self, current_time, strong=False):
        if self.last_speaker == "Unknown":
            return False
        elapsed = current_time - self.last_speaker_time
        window = (self.STRONG_CONTINUITY_WINDOW_SECONDS if strong
                  else self.CONTINUITY_WINDOW_SECONDS)
        return elapsed <= window

    def _is_transition_detected(self, vector, current_time):
        """
        Compare this embedding's distance to the last speaker's centroid
        against the rolling average distance for that speaker. A significant
        jump indicates a speaker change and suppresses continuity preference.
        """
        if self.last_speaker == "Unknown":
            return False
        if self.last_speaker not in self.bank.centroids:
            return False

        current_dist = cos_dist(
            np.asarray(vector).flatten(),
            self.bank.centroids[self.last_speaker]
        )

        recent = [
            d for (t, spk, d) in self.speaker_distance_history
            if spk == self.last_speaker
            and (current_time - t) <= self.CONTINUITY_WINDOW_SECONDS
        ]

        if len(recent) < 2:
            # Insufficient history: only flag obvious mismatches
            return current_dist > 0.55

        avg_dist = np.mean(recent)
        jump = current_dist - avg_dist
        return jump > self.TRANSITION_DISTANCE_JUMP

    def _record_distance(self, speaker, vector, current_time):
        if speaker not in self.bank.centroids:
            return
        dist = cos_dist(
            np.asarray(vector).flatten(),
            self.bank.centroids[speaker]
        )
        self.speaker_distance_history.append((current_time, speaker, dist))
        self.speaker_distance_history = self.speaker_distance_history[-30:]

    def _decide_speaker(self, vector, n_samples, current_time):
        """
        Decide speaker from a pre-extracted embedding vector (or None if
        the audio was too short to embed).
        """
        # Very short audio: always inherit if continuity is likely
        if vector is None:
            if self._is_continuity_likely(current_time):
                return self.last_speaker, 0.0
            return "Unknown", 0.0

        is_reliable = n_samples >= self.MIN_RELIABLE_EMBEDDING_SAMPLES
        transition_detected = self._is_transition_detected(vector, current_time)

        # Continuity preference is suppressed at speaker transitions
        prefer_label = None
        if self._is_continuity_likely(current_time) and not transition_detected:
            prefer_label = self.last_speaker

        if not is_reliable:
            if (self._is_continuity_likely(current_time, strong=True)
                    and not transition_detected):
                proposed_label, confidence = self.bank.score_against_known(
                    vector, self.last_speaker
                )
                if proposed_label == self.last_speaker or confidence < 0.6:
                    if confidence > 0.4:
                        self.bank.update_centroid(
                            self.last_speaker, vector, n_samples=n_samples
                        )
                    return self.last_speaker, confidence

            label, confidence = self.bank.process_segment(
                vector, allow_new_speaker=False, prefer_label=prefer_label,
                n_samples=n_samples,
            )
            return label, confidence

        label, confidence = self.bank.process_segment(
            vector, allow_new_speaker=True, prefer_label=prefer_label,
            n_samples=n_samples,
        )
        return label, confidence

    def _processor_thread(self):
        while self.is_recording or not self.audio_queue.empty():
            try:
                current_time, audio_window = self.audio_queue.get(timeout=1)
                audio_np = (
                    np.frombuffer(audio_window.get_raw_data(), dtype=np.int16)
                    .astype(np.float32)
                    / 32768.0
                )

                segments, _ = self.whisper.transcribe(audio_np, beam_size=5)
                text = "".join([seg.text for seg in segments]).strip()

                if not text:
                    continue

                n_samples = len(audio_np)

                # Extract embedding once. Skip for very short audio.
                vector = None
                if n_samples >= self.SHORT_AUDIO_INHERIT_SAMPLES:
                    try:
                        vector = self._extract_embedding(audio_np)
                    except Exception as e:
                        print(f"[Embedding Error]: {e}")

                speaker_label, confidence = self._decide_speaker(
                    vector, n_samples, current_time
                )

                # Record distance AFTER decision so the centroid is updated
                # before we measure distance against it next time.
                if vector is not None:
                    self._record_distance(speaker_label, vector, current_time)

                self.last_speaker = speaker_label
                self.last_speaker_time = current_time
                self.last_confidence = confidence

                if (
                    self.full_transcript
                    and self.full_transcript[-1].startswith(f"[{speaker_label}]:")
                ):
                    self.full_transcript[-1] = (
                        self.full_transcript[-1].rstrip() + " " + text
                    )
                else:
                    self.full_transcript.append(f"[{speaker_label}]: {text}")

                print(f"[{speaker_label} | conf={confidence:.2f}]: {text}")

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

        threading.Thread(
            target=self.summarizer.generate_minutes,
            args=(self.full_transcript,),
        ).start()

        return self.bank.get_finalized_centroids()