

import os
import tempfile
import time

import numpy as np
import soundfile as sf
from kokoro import KPipeline


class TTSController:

    def __init__(self, mini):
        self.mini = mini
        self.pending_cleanup = []

        print("[TTS] Loading Kokoro...")
        self.pipeline = KPipeline(lang_code="b")

        self.voice = "bf_emma"
        self.speed = 1.0
        self.sample_rate = 24000

        print("[TTS] Ready.")

    def prepare_speech(self, text: str):
        """Generate speech and save it as a WAV file."""

        print(f"[TTS] Generating: {text}")

        generator = self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed
        )

        chunks = [
            audio for gs, ps, audio in generator
        ]

        if not chunks:
            raise RuntimeError("No audio generated.")

        audio = np.concatenate(chunks)

        duration = len(audio) / self.sample_rate

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            wav_path = f.name

        sf.write(wav_path, audio, self.sample_rate)

        print(f"[TTS] Audio prepared ({duration:.2f}s).")

        return wav_path, duration

    def play_prepared(self, wav_path: str, duration: float):
        """Play previously generated audio."""

        print("[TTS] Playing...")

        self.mini.media.play_sound(wav_path)

        # Wait for expected playback duration.
        time.sleep(duration + 0.5)

        print("[TTS] Playback finished.")

    # @staticmethod
    # def cleanup(wav_path: str):
    #     """Remove temporary audio file."""

    #     if os.path.exists(wav_path):
    #         os.remove(wav_path)

    # @staticmethod
    # def cleanup(wav_path: str):
    #     if os.path.exists(wav_path):
    #         try:
    #             os.remove(wav_path)
    #             print("[TTS] Temporary audio file removed.")
    #         except PermissionError:
    #             print("[TTS] File is still in use. Skipping cleanup.")

    # @staticmethod
    # def cleanup(wav_path: str):
    #     if not os.path.exists(wav_path):
    #         return

    #     max_retries = 10

    #     for attempt in range(max_retries):
    #         try:
    #             os.remove(wav_path)
    #             print("[TTS] Temporary audio file removed.")
    #             return

    #         except PermissionError:
    #             if attempt < max_retries - 1:
    #                 time.sleep(0.5)
    #             else:
    #                 print(
    #                     f"[TTS] Warning: Cannot remove {wav_path}. "
    #                     "File may still be in use."
    #                 )

    def defer_cleanup(self, wav_path: str):
        """Save the path for cleanup after Reachy Mini disconnects."""
        self.pending_cleanup.append(wav_path)


    def cleanup_pending(self):
        """Remove all temporary WAV files after the media connection closes."""
        for wav_path in self.pending_cleanup:
            if not os.path.exists(wav_path):
                continue

            for attempt in range(10):
                try:
                    os.remove(wav_path)
                    print(f"[TTS] Removed: {wav_path}")
                    break

                except PermissionError:
                    if attempt < 9:
                        time.sleep(0.5)
                    else:
                        print(f"[TTS] Could not remove: {wav_path}")

        self.pending_cleanup.clear()