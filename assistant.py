import threading
import queue
import time
import numpy as np
import speech_recognition as sr
import torch
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier # Add this

from speaker_bank import SmartSpeakerBank
from summarizer import LocalLLMSummarizer


class MeetingAssistant:
    """
    Meeting assistant with live enrollment.
    
    The previous architecture relied on pre-meeting voice enrollment, but
    pyannote embeddings are not stable enough across different recording
    conditions for cross-condition matching to work reliably. This version
    treats pre-meeting enrollment as a hint only and builds robust speaker
    profiles from actual meeting audio.
    
    Strategy:
    1. First N seconds of each speaker's meeting audio are accumulated
       into a "live profile" before any matching decisions are made.
    2. Matching only begins after we have at least one live profile.
    3. Pre-enrolled profiles are used as a tiebreaker for naming the
       discovered clusters, not for direct distance matching.
    """

    MIN_RELIABLE_EMBEDDING_SAMPLES = 48000
    MIN_USABLE_EMBEDDING_SAMPLES = 24000
    SHORT_AUDIO_INHERIT_SAMPLES = 16000

    CONTINUITY_WINDOW_SECONDS = 5.0
    STRONG_CONTINUITY_WINDOW_SECONDS = 3.0

    # How much audio to accumulate per speaker before locking in their profile
    LIVE_ENROLLMENT_TARGET_SAMPLES = 80000  # 5 seconds total across utterances

    def __init__(self, hf_token, enrolled_profiles=None, expected_attendees=None):
        self.audio_queue = queue.Queue()
        self.is_recording = False

        self.last_speaker = "Unknown"
        self.last_speaker_time = 0.0
        self.last_confidence = 0.0
        self.recent_decisions = []

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
        # Replace Pyannote with SpeechBrain's ECAPA-TDNN
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device":"cpu"} # change to "cuda" if you have a GPU
        )
    def _extract_embedding(self, audio_np, sample_rate):
        # SpeechBrain expects a tensor of shape (batch, time)
        tensor = torch.from_numpy(audio_np).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.encoder.encode_batch(tensor)
        return embeddings.squeeze().cpu().numpy()

    def _audio_listener_thread(self):
        with sr.Microphone(sample_rate=16000) as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("Listening... (Press Ctrl+C to stop)")

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

    def _decide_speaker(self, audio_np, sample_rate, current_time):
        n_samples = len(audio_np)

        if n_samples < self.SHORT_AUDIO_INHERIT_SAMPLES:
            if self._is_continuity_likely(current_time):
                return self.last_speaker, 0.0, False

        try:
            vector = self._extract_embedding(audio_np, sample_rate)
        except Exception as e:
            print(f"[Embedding Error]: {e}")
            if self._is_continuity_likely(current_time):
                return self.last_speaker, 0.0, False
            return "Unknown", 0.0, False

        is_reliable = n_samples >= self.MIN_RELIABLE_EMBEDDING_SAMPLES

        prefer_label = None
        if self._is_continuity_likely(current_time):
            prefer_label = self.last_speaker

        if not is_reliable:
            if self._is_continuity_likely(current_time, strong=True):
                proposed_label, confidence = self.bank.score_against_known(
                    vector, self.last_speaker
                )
                if proposed_label == self.last_speaker or confidence < 0.6:
                    if confidence > 0.4:
                        self.bank.update_centroid(
                            self.last_speaker, vector, n_samples=n_samples
                        )
                    return self.last_speaker, confidence, True

            label, confidence = self.bank.process_segment(
                vector, allow_new_speaker=False, prefer_label=prefer_label,
                n_samples=n_samples,
            )
            return label, confidence, True

        label, confidence = self.bank.process_segment(
            vector, allow_new_speaker=True, prefer_label=prefer_label,
            n_samples=n_samples,
        )
        return label, confidence, True

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
                text = "".join([segment.text for segment in segments]).strip()

                if not text:
                    continue

                speaker_label, confidence, was_embedded = self._decide_speaker(
                    audio_np, audio_window.sample_rate, current_time
                )

                self.last_speaker = speaker_label
                self.last_speaker_time = current_time
                self.last_confidence = confidence
                self.recent_decisions.append(
                    (current_time, speaker_label, confidence)
                )
                self.recent_decisions = self.recent_decisions[-10:]

                if (
                    self.full_transcript
                    and self.full_transcript[-1].startswith(f"[{speaker_label}]:")
                ):
                    self.full_transcript[-1] = (
                        self.full_transcript[-1].rstrip() + " " + text
                    )
                else:
                    log_entry = f"[{speaker_label}]: {text}"
                    self.full_transcript.append(log_entry)

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