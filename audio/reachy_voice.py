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
import time
import wave
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")


class KokoroTTS:
    """Kokoro-82M (Apache 2.0) through kokoro-onnx.

    The int8 model takes about 0.6 s per second of speech on an M-series CPU, so a sentence
    has to be made before it can play. 4 threads measured fastest (more threads were slower).
    """

    THREADS = 4

    def __init__(self, model: str, voices: str, voice: str, speed: float):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.THREADS
        session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        self._tts = Kokoro.from_session(session, voices)
        self.model_name = os.path.basename(model)
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


CLAUSE_RE = re.compile(r"(?<=[,;:—–])\s+")


def split_first_clause(sentence: str, min_words: int = 2, min_rest_words: int = 3,
                       max_rest_ratio: float = 3.0) -> list[str]:
    """"Sunscreen, a hat, and sandals are the basics." -> ["Sunscreen, a hat,", "and sandals..."].

    Making speech takes time in proportion to its length, so the first sound comes sooner if
    the first piece is short. Splits at the first comma (or ; : —) that leaves both pieces
    long enough to sound natural, and the rest at most max_rest_ratio times longer than the
    first piece: the rest is made while the first piece plays, so a tiny first piece before a
    long rest would leave a silent gap. Otherwise returns the sentence unchanged.
    """
    for m in CLAUSE_RE.finditer(sentence):
        head, rest = sentence[:m.start()], sentence[m.end():]
        head_words, rest_words = len(head.split()), len(rest.split())
        if (head_words >= min_words and rest_words >= min_rest_words
                and rest_words <= max_rest_ratio * head_words):
            return [head, rest]
    return [sentence]


def split_long_sentence(sentence: str, max_words: int = 12, min_words: int = 3) -> list[str]:
    """Cut a long sentence at its commas into pieces of about max_words.

    Each piece is made while the one before it plays. One long sentence (20-30 words takes
    about 2.5 s to make, longer on a hot laptop) would otherwise mean silence while it's made.
    Pieces shorter than min_words are kept with their neighbour; a long stretch with no comma
    stays whole.
    """
    clauses = [c for c in CLAUSE_RE.split(sentence) if c.strip()]
    pieces: list[str] = []
    for clause in clauses:
        if pieces and (len(pieces[-1].split()) + len(clause.split()) <= max_words
                       or len(clause.split()) < min_words or len(pieces[-1].split()) < min_words):
            pieces[-1] = f"{pieces[-1]} {clause}"
        else:
            pieces.append(clause)
    return pieces or [sentence]


MODELS_DIR = os.path.join(HERE, "models")
# Fastest first. On an M-series Mac the full-precision model makes speech about 2x faster than
# int8 (0.38 s vs 0.75 s for a short clause); int8 is smaller (92 MB vs 326 MB).
KOKORO_MODELS = ("kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx")


def default_kokoro_model() -> str:
    for name in KOKORO_MODELS:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return path
    return os.path.join(MODELS_DIR, KOKORO_MODELS[-1])   # missing: the error names this file


class ReachyVoice:
    def __init__(self, voice: str = "auto",
                 kokoro_model: Optional[str] = None,
                 kokoro_voices: str = os.path.join(HERE, "models", "voices-v1.0.bin"),
                 kokoro_voice: str = "af_heart", speed: float = 1.0,
                 say_voices: tuple = ("Ava (Premium)", "Zoe (Premium)", "Ava (Enhanced)", "Samantha"),
                 output_device=None):
        """voice: "auto" (Kokoro, else `say`), "kokoro", "say" or "none" (print only)."""
        self.output_device = output_device
        self.tts = self._make_tts(voice, kokoro_model or default_kokoro_model(), kokoro_voices,
                                  kokoro_voice, speed, say_voices)
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
                print(f"[voice] {name}" + (f" ({tts.voice})" if name == "say"
                                           else f" ({kokoro_voice}, {tts.model_name})"))
                return tts
            except Exception as e:   # noqa: BLE001 - missing package, model files or command
                print(f"[voice] {name} unavailable: {e}")
        print("[voice] none: replies are printed only")
        return None

    @property
    def speaking(self) -> bool:
        return self._pending > 0

    def say(self, text: str, on_start: Optional[Callable[[], None]] = None, interrupt: bool = False,
            quick_start: bool = False) -> None:
        """Speak `text` without waiting. If Reachy is already talking it is queued after,
        unless interrupt=True. on_start() is called when the first sound plays.

        quick_start: split the first sentence at its first comma, so its first part is made
        and starts playing sooner (use it for the first sentence of a reply)."""
        sentences = split_sentences(text or "")
        if not sentences:
            return
        if quick_start:
            sentences = split_first_clause(sentences[0]) + sentences[1:]
        sentences = [piece for s in sentences for piece in split_long_sentence(s)]
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
                self._text_q.put((gen, s, on_start if i == 0 else None, time.perf_counter()))

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

    WARM_UP = ("Hi there, nice to meet you.", "Oh, that sounds lovely! What do you two think about it?")

    def _warm_up(self) -> None:
        """The first real sentences are slow (0.84 s vs 0.48 s later), so make a couple at
        startup, before anyone talks to Reachy."""
        for text in self.WARM_UP:
            try:
                self.tts.synth(text)
            except Exception as e:   # noqa: BLE001 - reported again on the first real sentence
                print(f"[voice] warm-up failed: {e}")
                return

    def _synth_loop(self) -> None:
        self._warm_up()
        while True:
            gen, text, on_start, queued_t = self._text_q.get()
            if gen != self._gen:
                continue
            try:
                t0 = time.perf_counter()
                audio, sr = self.tts.synth(text)
                made_s = time.perf_counter() - t0
            except Exception as e:   # noqa: BLE001 - one bad sentence shouldn't kill the voice
                print(f"[voice] couldn't say {text!r}: {e}")
                self._finished_one(gen)
                continue
            if gen == self._gen:
                self._audio_q.put((gen, audio, sr, on_start, text, queued_t, made_s))

    def _play_loop(self) -> None:
        import sounddevice as sd

        out = None          # kept open between sentences: reopening adds a gap each time
        out_sr = None
        while True:
            waiting_since = time.perf_counter()
            gen, audio, sr, on_start, text, queued_t, made_s = self._audio_q.get()
            if gen != self._gen:
                continue
            self._report_gap(gen, on_start, text, queued_t, made_s, waiting_since)
            if on_start is not None:
                try:
                    on_start()
                except Exception as e:   # noqa: BLE001 - the controller's callback
                    print(f"[voice] on_start failed: {e}")
            block = sr // 25
            try:
                if out is None or out_sr != sr:
                    if out is not None:
                        out.close()
                    out = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                          device=self.output_device)
                    out.start()
                    out_sr = sr
                for i in range(0, len(audio), block):
                    if gen != self._gen:
                        break
                    out.write(audio[i:i + block].reshape(-1, 1))
            except Exception as e:   # noqa: BLE001 - e.g. no output device
                print(f"[voice] playback failed: {e}")
                if out is not None:
                    out.close()
                out = None
            self._last_piece = (gen, time.perf_counter())
            self._finished_one(gen)

    GAP_REPORT_S = 0.3

    def _report_gap(self, gen, on_start, text, queued_t, made_s, waiting_since) -> None:
        """Print when Reachy goes quiet mid-reply for longer than GAP_REPORT_S, and why."""
        last = getattr(self, "_last_piece", None)
        if on_start is not None or last is None or last[0] != gen:
            return                                   # first piece of a reply: no gap to report
        gap = time.perf_counter() - max(last[1], waiting_since)
        if gap < self.GAP_REPORT_S or queued_t - last[1] > 3.0:
            return                                   # short, or a separate say() much later
        if queued_t > last[1]:
            why = "the reply's next sentence wasn't written yet"
        else:
            why = f"making it took {made_s:.2f}s"
        print(f"[voice] {gap:.2f}s silent gap before {text[:40]!r}: {why}")


if __name__ == "__main__":
    v = ReachyVoice()
    v.say(" ".join(sys.argv[1:]) or "Hi! I'm Reachy. Nice to meet you both.")
    v.wait()
