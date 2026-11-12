"""Reachy's voice: text-to-speech for robot_controller.py.

    voice = ReachyVoice()                 # Kokoro (natural, offline); falls back to macOS `say`
    voice.say("Hi! Nice to meet you.")    # returns at once; sentence 1 plays while 2 is made
    voice.speaking                        # True from say() until the last word is played
    voice.stop()                          # cut it off, e.g. when someone interrupts
    voice.wait()                          # block until it has finished

To start a movement exactly when the sound starts:
    voice.say(text, on_start=lambda: play_move(...))

Try it:  python reachy_voice.py "Hello, I'm Reachy."
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")


class KokoroTTS:
    """Kokoro-82M (Apache 2.0) through kokoro-onnx. About 0.4 s per short sentence on an M4."""

    def __init__(self, model: str, voices: str, voice: str, speed: float):
        from kokoro_onnx import Kokoro

        self._tts = Kokoro(model, voices)
        self.voice, self.speed = voice, speed

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        audio, sr = self._tts.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return audio.astype(np.float32), sr


class SayTTS:
    """macOS `say`, rendered to a WAV file so playback can be cut off."""

    def __init__(self, voices: tuple):
        if not shutil.which("say"):
            raise RuntimeError("macOS `say` not found")
        installed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        self.voice = next((v for v in voices if v in installed), None)

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            cmd = ["say", "-o", f.name, "--file-format=WAVE", "--data-format=LEI16@22050"]
            if self.voice:
                cmd += ["-v", self.voice]
            subprocess.run(cmd + [text], check=True)
            with wave.open(f.name) as w:
                frames, sr = w.readframes(w.getnframes()), w.getframerate()
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, sr


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Split into sentences, joining very short ones ("Oh!") to the next so it doesn't sound choppy."""
    out: list[str] = []
    carry = ""
    for s in SENTENCE_END_RE.split(text.strip()):
        s = f"{carry} {s}".strip() if carry else s.strip()
        if not s:
            continue
        if len(s.split()) < min_words:
            carry = s
        else:
            out.append(s)
            carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            out.append(carry)
    return out


class ReachyVoice:
    def __init__(self, voice: str = "auto",
                 kokoro_model: str = os.path.join(HERE, "models", "kokoro-v1.0.int8.onnx"),
                 kokoro_voices: str = os.path.join(HERE, "models", "voices-v1.0.bin"),
                 kokoro_voice: str = "af_heart", speed: float = 1.0,
                 say_voices: tuple = ("Ava (Premium)", "Zoe (Premium)", "Ava (Enhanced)", "Samantha"),
                 output_device=None):
        """voice: "auto" (Kokoro, else `say`), "kokoro", "say" or "none" (print only)."""
        self.output_device = output_device
        self.tts = self._make_tts(voice, kokoro_model, kokoro_voices, kokoro_voice, speed, say_voices)
        self._lock = threading.Lock()
        self._gen = 0                    # bumped by stop(); older queued sentences are dropped
        self._pending = 0                # sentences said but not finished playing
        self._idle = threading.Event()
        self._idle.set()
        self._text_q: queue.Queue = queue.Queue()
        self._audio_q: queue.Queue = queue.Queue()
        if self.tts is not None:
            threading.Thread(target=self._synth_loop, daemon=True).start()
            threading.Thread(target=self._play_loop, daemon=True).start()

    @staticmethod
    def _make_tts(voice, kokoro_model, kokoro_voices, kokoro_voice, speed, say_voices):
        choices = {"auto": ("kokoro", "say"), "kokoro": ("kokoro",), "say": ("say",), "none": ()}[voice]
        for name in choices:
            try:
                tts = (KokoroTTS(kokoro_model, kokoro_voices, kokoro_voice, speed) if name == "kokoro"
                       else SayTTS(say_voices))
                print(f"[voice] {name}" + (f" ({tts.voice})" if name == "say" else f" ({kokoro_voice})"))
                return tts
            except Exception as e:   # noqa: BLE001 - missing package, model files or command
                print(f"[voice] {name} unavailable: {e}")
        print("[voice] none: replies are printed only")
        return None

    @property
    def speaking(self) -> bool:
        return self._pending > 0

    def say(self, text: str, on_start: Optional[Callable[[], None]] = None, interrupt: bool = False) -> None:
        """Speak `text` without waiting. If Reachy is already talking it is queued after,
        unless interrupt=True. on_start() is called when the first sound plays."""
        sentences = split_sentences(text or "")
        if not sentences:
            return
        if self.tts is None:
            print(f"[reachy says] {text}")
            if on_start:
                on_start()
            return
        with self._lock:
            if interrupt:
                self._stop_locked()
            gen = self._gen
            self._pending += len(sentences)
            self._idle.clear()
            for i, s in enumerate(sentences):
                self._text_q.put((gen, s, on_start if i == 0 else None))

    def stop(self) -> None:
        """Stop talking now and drop anything still queued."""
        with self._lock:
            self._stop_locked()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until Reachy has finished talking. False if the timeout ran out first."""
        return self._idle.wait(timeout)

    def _stop_locked(self) -> None:
        self._gen += 1
        self._pending = 0
        self._idle.set()
        for q in (self._text_q, self._audio_q):
            while not q.empty():
                q.get_nowait()

    def _finished_one(self, gen: int) -> None:
        with self._lock:
            if gen == self._gen and self._pending > 0:
                self._pending -= 1
                if self._pending == 0:
                    self._idle.set()

    def _synth_loop(self) -> None:
        while True:
            gen, text, on_start = self._text_q.get()
            if gen != self._gen:
                continue
            try:
                audio, sr = self.tts.synth(text)
            except Exception as e:   # noqa: BLE001 - one bad sentence shouldn't kill the voice
                print(f"[voice] couldn't say {text!r}: {e}")
                self._finished_one(gen)
                continue
            if gen == self._gen:
                self._audio_q.put((gen, audio, sr, on_start))

    def _play_loop(self) -> None:
        import sounddevice as sd

        while True:
            gen, audio, sr, on_start = self._audio_q.get()
            if gen != self._gen:
                continue
            if on_start is not None:
                try:
                    on_start()
                except Exception as e:   # noqa: BLE001 - the controller's callback
                    print(f"[voice] on_start failed: {e}")
            block = sr // 25
            try:
                with sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                     device=self.output_device) as out:
                    for i in range(0, len(audio), block):
                        if gen != self._gen:
                            break
                        out.write(audio[i:i + block].reshape(-1, 1))
            except Exception as e:   # noqa: BLE001 - e.g. no output device
                print(f"[voice] playback failed: {e}")
            self._finished_one(gen)


if __name__ == "__main__":
    v = ReachyVoice()
    v.say(" ".join(sys.argv[1:]) or "Hi! I'm Reachy. Nice to meet you both.")
    v.wait()
