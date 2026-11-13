"""Speaker / NLP model for Reachy Mini (Sara). Listens, writes down what was said, and writes
Reachy's reply with an emotion when the controller asks for one.

It never moves the robot and never plays sound: robot_controller.py (Pinky) does both, using
reachy_voice.py for the voice.

update(), called about 10 times a second:

    {"utterance": None or {                   # set once, when someone finishes a sentence
         "id": 12,
         "t": 1793801234.56,                  # time.time() when they started speaking
         "text": "Reachy, should we keep the water or the mirror?",
         "intent": "question",                # "question" or "statement"
         "said_name": True,                   # they said "Reachy"
         "answering_robot": False,            # first reply to a question Reachy just asked
         "interrupting": False},              # they started while Reachy was talking
     "interrupting": False}                   # True right away while someone talks over Reachy

respond(), called when the controller decides Reachy should answer (takes about 1 s):

    {"text": "Water, definitely. We can't last long without it.",
     "emotion": "helpful",                    # always one of the emotions passed in
     "reply_to": 12,                          # the utterance it answers
     "source": "llm"}                         # "template" if the LLM was unavailable

Usage in the controller:

    voice = ReachyVoice()
    nlp = SpeakerNLPModel(Config(), mini=mini, mic=RobotMic(mini), emotions=EMOTION_MOVES, voice=voice)
    nlp.start()
    while running:
        out = nlp.update()
        if out["interrupting"]:
            voice.stop()
        utt = out["utterance"]
        if utt and talking_to_reachy:          # the controller's decision
            reply = nlp.respond(utt["id"])
            voice.say(reply["text"])
            play_move(EMOTION_MOVES[reply["emotion"]])
    nlp.stop()

Try it on a laptop (no robot):
    python speaker_nlp_model.py --typed            # type what people say
    python speaker_nlp_model.py --speak            # use the mic, and speak replies aloud
    python speaker_nlp_model.py --watch            # show the conversation log live
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")    # put ANTHROPIC_API_KEY=... here; it's never shared
SIDE_LABEL = {"left": "Person Left", "right": "Person Right", None: "Someone"}


def load_env_file(path: str = ENV_FILE) -> list[str]:
    """Read KEY=value lines from a .env file into the environment.

    A variable that is already set (e.g. exported in ~/.zshrc) is kept. Empty values are
    skipped. Returns the names that were loaded, never their values.
    """
    loaded = []
    if not os.path.exists(path):
        return loaded
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.removeprefix("export ").split("=", 1)
            name, value = name.strip(), value.strip().strip("\"'")
            if name and value and not os.environ.get(name):
                os.environ[name] = value
                loaded.append(name)
    return loaded

# The emotions Reachy can reply with. The hints help the LLM choose; the controller maps each
# name to a movement. Pass your own list or dict as `emotions=` to replace them.
EMOTION_HINTS = {
    "welcoming": "saying hello or goodbye, making someone feel welcome",
    "thoughtful": "considering a question, weighing options",
    "surprised": "reacting to something unexpected",
    "helpful": "giving a suggestion or useful information",
    "anxious": "worried about a risk or a bad idea",
    "uncertain": "not sure, mixed feelings, can't decide",
    "enthusiastic": "excited, really likes an idea",
    "neutral": "a plain remark, none of the others fit",
}
DEFAULT_EMOTIONS = tuple(EMOTION_HINTS)


@dataclass
class Config:
    sample_rate: int = 16000             # whisper and the robot microphones both use 16 kHz

    # Sound direction (0 rad = robot's left, pi/2 = front, pi = robot's right). It only labels
    # who said what (Person Left / Right) in the conversation. Calibrate both in the lab.
    left_max_rad: float = 1.25
    right_min_rad: float = 1.90
    swap_left_right: bool = False        # True if the team means "left in the camera image"

    # Speech detection. "silero" is a small speech model that ignores coughs, claps and door
    # noise; "energy" only uses loudness.
    vad: str = "silero"
    silero_threshold: float = 0.5        # speech probability per 32 ms frame
    vad_rms: float = 0.012               # loudness threshold for "energy" (and silero's floor)
    silence_end_s: float = 0.4           # pause that ends a sentence
    long_silence_s: float = 1.0          # pause allowed instead, when the words so far look
                                         # unfinished ("It's just", "because", 1-2 words)
    verdict_grace_s: float = 0.2         # extra wait if that check isn't back after silence_end_s
    early_silence_s: float = 0.15        # pause after which transcription starts early, in case
                                         # the sentence is over (reused if it is, dropped if not)
    preroll_s: float = 0.25              # audio kept from just before speech starts, so the
                                         # first word ("Reachy, ...") isn't clipped
    min_utterance_s: float = 0.35
    max_utterance_s: float = 15.0
    unfinished_wait_s: float = 0.8       # "...is nice because" -> wait this long for the rest

    # Speech-to-text.
    stt_model: str = "base.en"           # "tiny.en" is faster, "small.en" more accurate
    stt_device: str = "auto"             # "auto": Apple GPU (about 2x faster), else CPU; or "cpu"

    # While Reachy talks.
    echo_guard_s: float = 0.5            # ignore the mic this long after Reachy stops talking
    allow_interruptions: bool = True     # False: ignore the mic completely while Reachy talks
    interrupt_rms: float = 0.06          # louder than Reachy's own voice at the mic
    interrupt_s: float = 0.3             # ...for this long counts as an interruption

    # Replies.
    use_llm: bool = True
    llm_backend: str = "claude"          # "claude" (needs an API key) or "ollama" (free, local)
    llm_model: str = "claude-haiku-5-5"
    llm_effort: str = "low"              # keeps replies quick
    llm_timeout_s: float = 8.0           # then a pre-written reply is used
    ollama_model: str = "qwen3:4b-instruct"
    ollama_host: str = "http://localhost:11434"
    history_lines: int = 12
    task_items: tuple = ("destination", "duration", "budget", "transport", "accommodation",
                         "activities", "packing")   # trip-planning topics; edit to match the study
    free_chat: bool = False              # True: no task, just chat (for demos)
    max_reply_sentences: int = 2
    followup_window_s: float = 8.0       # after Reachy asks a question, the next sentence started
                                         # within this time is marked answering_robot (0 = off)

    # Logs (both appended to across runs). None turns one off.
    log_csv: Optional[str] = os.path.join(HERE, "logs", "speaker_nlp.csv")           # 10 rows/s
    conversation_csv: Optional[str] = os.path.join(HERE, "logs", "conversation.csv")  # per line


# ---------------------------------------------------------------------------
# Sound direction
# ---------------------------------------------------------------------------

def doa_to_side(angle: Optional[float], cfg: Config) -> Optional[str]:
    if angle is None:
        return None
    if angle <= cfg.left_max_rad:
        side = "left"
    elif angle >= cfg.right_min_rad:
        side = "right"
    else:
        return None
    if cfg.swap_left_right:
        side = "right" if side == "left" else "left"
    return side


class RobotDoA:
    """The robot's microphone array (returns None in the simulator)."""

    def __init__(self, mini):
        self.mini = mini

    def read(self, voiced: bool) -> Optional[tuple[float, bool]]:
        return self.mini.media.get_DoA()


class MockDoA:
    """Laptop stand-in: the side is set from the keyboard."""

    ANGLES = {"left": 0.4, "front": math.pi / 2, "right": math.pi - 0.4}

    def __init__(self):
        self.angle = self.ANGLES["front"]

    def set_side(self, side: str) -> None:
        self.angle = self.ANGLES[side]

    def read(self, voiced: bool) -> Optional[tuple[float, bool]]:
        return self.angle, voiced


# ---------------------------------------------------------------------------
# Microphones
# ---------------------------------------------------------------------------

class LaptopMic:
    def __init__(self, cfg: Config):
        import sounddevice as sd

        self._q: queue.Queue = queue.Queue()
        self._stream = sd.InputStream(samplerate=cfg.sample_rate, channels=1, dtype="float32",
                                      blocksize=cfg.sample_rate // 20, callback=self._callback)

    def _callback(self, data, frames, t, status) -> None:
        self._q.put(data[:, 0].copy())

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def read(self) -> Optional[np.ndarray]:
        chunks = []
        while not self._q.empty():
            chunks.append(self._q.get_nowait())
        return np.concatenate(chunks) if chunks else None


class RobotMic:
    """The robot's microphones, through the controller's ReachyMini connection (audio only)."""

    def __init__(self, mini):
        self.mini = mini

    def start(self) -> None:
        self.mini.media.start_recording()

    def stop(self) -> None:
        self.mini.media.stop_recording()

    def read(self) -> Optional[np.ndarray]:
        chunks = []
        while (sample := self.mini.media.get_audio_sample()) is not None:
            chunks.append(sample.mean(axis=1))   # stereo -> mono
        return np.concatenate(chunks).astype(np.float32) if chunks else None


def rms(audio: Optional[np.ndarray]) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


# ---------------------------------------------------------------------------
# Speech detection and speech-to-text
# ---------------------------------------------------------------------------

class SileroVAD:
    """Streaming Silero VAD (the model bundled with faster-whisper), one 32 ms frame at a time."""

    FRAME, CONTEXT = 512, 64             # samples at 16 kHz

    def __init__(self, cfg: Config):
        from faster_whisper.vad import get_vad_model

        self.cfg = cfg
        self._session = get_vad_model().session
        self.prob = 0.0
        self._state = [np.zeros((1, 1, 128), dtype=np.float32) for _ in range(2)]
        self._context = np.zeros(self.CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def is_speech(self, chunk: np.ndarray) -> bool:
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32)])
        probs = []
        while len(self._pending) >= self.FRAME:
            frame, self._pending = self._pending[:self.FRAME], self._pending[self.FRAME:]
            x = np.concatenate([self._context, frame])[None, :]
            out, h, c = self._session.run(None, {"input": x, "h": self._state[0], "c": self._state[1]})
            self._state, self._context = [h, c], frame[-self.CONTEXT:]
            probs.append(float(np.ravel(out)[0]))
        if probs:
            self.prob = max(probs)
        return self.prob >= self.cfg.silero_threshold and rms(chunk) > self.cfg.vad_rms / 3


class EnergyVAD:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prob = 0.0

    def is_speech(self, chunk: np.ndarray) -> bool:
        return rms(chunk) > self.cfg.vad_rms


def make_vad(cfg: Config):
    if cfg.vad == "silero":
        try:
            return SileroVAD(cfg)
        except Exception as e:   # noqa: BLE001 - missing onnxruntime or model file
            print(f"[nlp] Silero VAD unavailable, using loudness ({e})")
    return EnergyVAD(cfg)


class UtteranceSegmenter:
    """Cuts the microphone stream into sentences, using the speech detector and a pause.

    feed() returns the sentence's audio once the pause reaches silence_end_s. Earlier, at
    early_silence_s, the audio so far is offered through pop_early() so it can be transcribed
    while we wait to be sure the sentence is over. final_matches_early then says whether the
    final sentence is the same audio (nobody spoke again), so that early transcript can be used.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._preroll: collections.deque = collections.deque()
        self.reset()
        self._last_voice_t = 0.0
        self._start_t = 0.0
        self.last_start_t = 0.0          # when the sentence just returned began
        self.last_end_t = 0.0            # ...and when its last voiced audio was heard
        self.final_matches_early = False
        self.last_waited = False         # the sentence just returned was given the longer pause
        self._early_seq = 0

    def reset(self) -> None:
        self._buf: list[np.ndarray] = []
        self._voiced_s = 0.0
        self.in_speech = False
        self._early_taken = False
        self._early: Optional[np.ndarray] = None
        self._verdict: Optional[bool] = None   # early words look finished? None: not known yet

    @property
    def early_seq(self) -> int:
        """Which early snapshot pop_early() last gave out (to match its transcript later)."""
        return self._early_seq

    def set_early_verdict(self, seq: int, finished: bool) -> None:
        """The early transcript of snapshot `seq` looks finished or not. Ignored if the person
        has spoken again since (that snapshot is stale)."""
        if self._early_taken and seq == self._early_seq:
            self._verdict = finished

    def _pause_needed(self) -> float:
        if not self._early_taken:
            return self.cfg.silence_end_s
        if self._verdict is False:                       # "It's just", "because": wait for more
            return self.cfg.long_silence_s
        if self._verdict is None:                        # check not back yet: a little longer
            return self.cfg.silence_end_s + self.cfg.verdict_grace_s
        return self.cfg.silence_end_s

    def pop_early(self) -> Optional[np.ndarray]:
        early, self._early = self._early, None
        return early

    def feed(self, chunk: np.ndarray, now: float, voiced: Optional[bool] = None) -> Optional[np.ndarray]:
        loud = rms(chunk) > self.cfg.vad_rms if voiced is None else voiced
        if loud:
            self._last_voice_t = now
            if not self.in_speech:
                self._start_t = now
                self._buf = list(self._preroll)      # the moment just before speech started
                self._preroll.clear()
            self.in_speech = True
            self._early_taken = False                # speaking again: the early audio is stale
            self._verdict = None
        if not self.in_speech:
            self._keep_preroll(chunk)
            return None
        self._buf.append(chunk)
        if loud:
            self._voiced_s += len(chunk) / self.cfg.sample_rate
        total_s = sum(len(b) for b in self._buf) / self.cfg.sample_rate
        silence_s = now - self._last_voice_t
        if silence_s >= self._pause_needed() or total_s >= self.cfg.max_utterance_s:
            audio, voiced_s = np.concatenate(self._buf), self._voiced_s
            self.last_start_t = self._start_t
            self.last_end_t = self._last_voice_t
            self.final_matches_early = self._early_taken
            self.last_waited = self._verdict is False
            self.reset()
            if voiced_s >= self.cfg.min_utterance_s:
                return audio
            return None
        if (not self._early_taken and silence_s >= self.cfg.early_silence_s
                and self._voiced_s >= self.cfg.min_utterance_s):
            self._early_taken = True
            self._early_seq += 1
            self._early = np.concatenate(self._buf)
        return None

    def _keep_preroll(self, chunk: np.ndarray) -> None:
        self._preroll.append(chunk)
        max_len = int(self.cfg.preroll_s * self.cfg.sample_rate)
        while len(self._preroll) > 1 and sum(len(c) for c in self._preroll) - len(self._preroll[0]) >= max_len:
            self._preroll.popleft()


class Transcriber:
    """faster-whisper in a background thread, so update() never waits for it.

    submit(job_id, audio) queues audio; results gets (job_id, text) when it's done.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._in: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()

    def start(self) -> None:
        from faster_whisper import WhisperModel   # fail here, not silently inside the thread

        threading.Thread(target=self._loop, args=(WhisperModel,), daemon=True).start()

    def submit(self, job_id: int, audio: np.ndarray) -> None:
        self._in.put((job_id, audio))

    def _devices(self) -> list[tuple[str, str]]:
        if self.cfg.stt_device == "auto":
            import platform
            on_apple_gpu = sys.platform == "darwin" and platform.machine() == "arm64"
            return ([("mps", "float16")] if on_apple_gpu else []) + [("cpu", "int8")]
        return [(self.cfg.stt_device, "float16" if self.cfg.stt_device != "cpu" else "int8")]

    def _load(self, WhisperModel, devices):
        # The first run downloads the model (about 150 MB for base.en).
        for i, (device, compute_type) in enumerate(devices):
            try:
                model = WhisperModel(self.cfg.stt_model, device=device, compute_type=compute_type)
                print(f"[nlp] speech-to-text ready ({self.cfg.stt_model} on {device})")
                return model, devices[i + 1:]
            except Exception as e:   # noqa: BLE001 - e.g. no Apple GPU support in this build
                print(f"[nlp] speech-to-text on {device} unavailable ({e})")
        raise RuntimeError("speech-to-text could not start on any device")

    def _loop(self, WhisperModel) -> None:
        model, fallbacks = self._load(WhisperModel, self._devices())
        while True:
            job_id, audio = self._in.get()
            try:
                text = self._transcribe(model, audio)
            except Exception as e:   # noqa: BLE001 - the experimental Apple GPU backend failed
                if not fallbacks:
                    raise
                print(f"[nlp] speech-to-text failed ({e}); switching to {fallbacks[0][0]}")
                model, fallbacks = self._load(WhisperModel, fallbacks)
                text = self._transcribe(model, audio)
            self.results.put((job_id, text))

    def _transcribe(self, model, audio: np.ndarray) -> str:
        # No vad_filter: Silero already cut this sentence, so a second pass only costs time.
        segments, _ = model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False, without_timestamps=True,
            condition_on_previous_text=False,
            hotwords="Reachy " + " ".join(self.cfg.task_items),   # nudges "Richie" -> "Reachy"
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------------------
# What was said
# ---------------------------------------------------------------------------

# Whisper often mishears the robot's name; these spellings are rewritten to "Reachy".
# "Rachel" is a real name: drop it from this list if a participant is called Rachel.
MISHEARD_NAME_RE = re.compile(
    r"\b(reachie|reechy|reachi|reeky|reiki|riki|reaky|richie|ritchie|richy|ricci|rachel)\b", re.I)
NAME_RE = re.compile(r"\b(reachy|reach me|robot)\b", re.I)
QUESTION_START_RE = re.compile(
    r"^((hey|hi|so|and|but|ok(ay)?|reachy)\W+)*"
    r"(what|why|how|which|who|where|when|should|could|would|do|does|did|is|are|can|shall|will)\b",
    re.I)
# Ends on a word nobody finishes a sentence with: they paused mid-thought.
UNFINISHED_RE = re.compile(
    r"\b(because|and|but|so|or|then|like|um|uh|the|a|an|to|of|with|for|if|that|my|your|is|are)"
    r"\W*$", re.I)


def looks_unfinished(text: str) -> bool:
    """Words that probably aren't a whole sentence yet: "It's just", "because the", "first".

    A question, or a short call like "Hi Reachy", counts as finished.
    """
    clean = MISHEARD_NAME_RE.sub("Reachy", text.strip())
    if not clean or clean.endswith("?"):
        return False
    words = re.findall(r"[A-Za-z']+", clean)
    if NAME_RE.search(clean) and len(words) <= 3:      # "Hi Reachy", "Reachy!"
        return False
    return bool(UNFINISHED_RE.search(clean)) or len(words) < 3


def classify_utterance(text: str) -> dict:
    clean = MISHEARD_NAME_RE.sub("Reachy", text.strip())
    is_question = clean.rstrip().endswith("?") or bool(QUESTION_START_RE.match(clean))
    return {"text": clean, "intent": "question" if is_question else "statement",
            "said_name": bool(NAME_RE.search(clean))}


# ---------------------------------------------------------------------------
# Replies: text + emotion
# ---------------------------------------------------------------------------

TASK_CONTEXT = """\
Today they may plan a trip together, agreeing on: {items}. They do the deciding; you are a \
warm, curious third voice they sometimes ask for an opinion or an idea.
Always answer about whatever was just said. Never bring up the trip or its topics yourself, \
not even in a greeting: wait until they mention it. If they talk about anything else (food, \
films, their day), stay on that topic. You can share general impressions of places, but never \
state prices, dates, opening times or travel rules as fact; you can't check them."""
FREE_CHAT_CONTEXT = "You are just chatting with them about whatever they like."

SYSTEM_PROMPT = """\
You are Reachy, a small, friendly desk robot. Two people sit in front of you: Person Left and \
Person Right. {context}

Reply with two fields:
- "text": what you say. It is spoken aloud by a text-to-speech voice, so write the way people talk.
- "emotion": how you feel saying it. Your body moves to match, so pick the one that fits best:
{emotions}

Rules for "text":
- One or two short sentences, under 30 words in total.
- Casual and natural: contractions, plain words, no lists, no markdown, no emoji.
- "Person Left" and "Person Right" are only labels for where people sit. Never say them; you \
don't know their names, so just talk to them as "you" (or "you two").
- Your name is Reachy. The transcript comes from speech recognition, so odd words may be \
mishearings; go with the most likely meaning rather than commenting on the words.
- First react to what was just said, then give your view in a few words.
- Sometimes hand the turn back with a short question to them.
- Vary how you start; do not open every reply the same way.
- Never claim to be human. Never take over the decision.
- You can only talk. You cannot look anything up (weather, news, time), set reminders or use \
apps. If asked, say so lightly in a few words. Never offer to do something you can't do."""

TAG_RE = re.compile(r"\[\s*@?[a-zA-Z_]+\s*\]")    # stray "[nod]"-style tags are dropped
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")   # the voice can't say these
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")
PERSON_LABEL_RE = re.compile(r"\bperson (left|right)\b", re.I)

# Models with a server-side fallback when Claude declines a request. Haiku has none, and a
# `fallbacks` parameter on a Haiku request would do nothing or be rejected.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


def clean_reply(text: str, max_sentences: int = 2) -> str:
    """Make the reply safe to speak: no tags, emoji, speaker labels or seat labels."""
    text = re.sub(r"\s+", " ", EMOJI_RE.sub("", TAG_RE.sub("", text or ""))).strip()
    text = re.sub(r"^(reachy|robot)\s*:\s*", "", text, flags=re.I).strip("\"'“” ")
    stripped = re.sub(r"^person (left|right)\b[,:]?\s*", "", text, flags=re.I)
    if stripped != text:
        text = stripped[:1].upper() + stripped[1:]                          # "Person Right, what..."
    text = re.sub(r",?\s*\bperson (left|right)\b(?=\s*[.!?…]?$)", "", text, flags=re.I)   # "..., Person Left?"
    text = PERSON_LABEL_RE.sub("you", text)                                # "I agree with Person Left"
    sentences = [s for s in SENTENCE_END_RE.split(text) if s.strip()]
    text = " ".join(sentences[:max_sentences]).strip()
    return text if re.search(r"[A-Za-z]", text) else ""



class ReplyStreamParser:
    """Pulls the emotion and whole sentences out of a {"emotion": ..., "text": ...} JSON reply
    while it is still streaming in, so Reachy can start talking at the first full sentence.

    Very short sentences ("Oh!") are joined to the next one so the voice doesn't sound choppy.
    """

    EMOTION_RE = re.compile(r'"emotion"\s*:\s*"((?:[^"\\]|\\.)*)"')
    TEXT_START_RE = re.compile(r'"text"\s*:\s*"')
    ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "", "f": "", "n": " ", "r": " ", "t": " "}

    def __init__(self, min_words: int = 3):
        self.min_words = min_words
        self.raw = ""
        self.emotion: Optional[str] = None
        self._used = 0            # characters of the decoded text already handed out
        self._carry = ""          # a short sentence waiting to be joined to the next

    def feed(self, delta: str) -> tuple[Optional[str], list[str]]:
        """Add streamed text. Returns (the emotion if it just became known, new full sentences)."""
        self.raw += delta
        new_emotion = None
        if self.emotion is None:
            m = self.EMOTION_RE.search(self.raw)
            if m:
                self.emotion = new_emotion = json.loads(f'"{m.group(1)}"')
        text, closed = self._text_so_far()
        return new_emotion, self._take_sentences(text, final=closed)

    def finish(self) -> list[str]:
        """Whatever is left once the stream has ended."""
        text, _ = self._text_so_far()
        return self._take_sentences(text, final=True)

    def _text_so_far(self) -> tuple[str, bool]:
        """Decode the "text" string as far as it has arrived. closed: its end quote arrived."""
        m = self.TEXT_START_RE.search(self.raw)
        if not m:
            return "", False
        out, i, s = [], m.end(), self.raw
        while i < len(s):
            c = s[i]
            if c == '"':
                return "".join(out), True
            if c == "\\":
                if i + 1 >= len(s):
                    break                               # the escape hasn't fully arrived
                e = s[i + 1]
                if e == "u":
                    if i + 6 > len(s):
                        break
                    try:
                        out.append(chr(int(s[i + 2:i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                out.append(self.ESCAPES.get(e, e))
                i += 2
                continue
            out.append(c)
            i += 1
        return "".join(out), False

    def _take_sentences(self, text: str, final: bool) -> list[str]:
        pending = text[self._used:]
        pieces = SENTENCE_END_RE.split(pending)
        if final:
            done, rest = pieces, ""
        else:
            done, rest = pieces[:-1], pieces[-1]        # the last piece may still be growing
        self._used = len(text) - len(rest)
        out = []
        for piece in done:
            piece = f"{self._carry} {piece.strip()}".strip() if self._carry else piece.strip()
            self._carry = ""
            if not piece:
                continue
            if len(piece.split()) < self.min_words and not final:
                self._carry = piece
            else:
                out.append(piece)
        if final and self._carry:
            out.append(self._carry)
            self._carry = ""
        return out


class Emotions:
    """The emotion names the controller uses, matched case-insensitively."""

    def __init__(self, emotions=None):
        if emotions is None:
            emotions = DEFAULT_EMOTIONS
        names = list(emotions.keys() if isinstance(emotions, dict) else emotions)
        if not names:
            raise ValueError("emotions must not be empty")
        self.names = names
        self._by_lower = {n.lower(): n for n in names}
        self.default = self._by_lower.get("neutral", names[0])

    def match(self, name: Optional[str]) -> str:
        return self._by_lower.get(str(name or "").strip().lower(), self.default)

    def prompt_lines(self) -> str:
        return "\n".join(f"  - {n}" + (f": {EMOTION_HINTS[n.lower()]}" if n.lower() in EMOTION_HINTS else "")
                         for n in self.names)


class TemplateReplies:
    """Backup so Reachy still answers when the LLM is slow, offline or unavailable."""

    GREETING = ("Hi! Nice to meet you both.", "welcoming")
    GENERIC = (
        ("Hmm, tough one. What do you two think matters most?", "thoughtful"),
        ("Ooh, good question. I'm honestly not sure. What's your gut say?", "uncertain"),
        ("I like where this is going. Tell me more?", "enthusiastic"),
    )
    ITEM = {   # first keyword found in what was just said picks the line
        "budget": ("Agree on the budget early, I'd say. It makes every other choice easier.", "helpful"),
        "hotel": ("A hotel's comfy, but a central one costs more. Is location worth it to you?", "thoughtful"),
        "hostel": ("A hostel's cheap and social. Do you two mind sharing a room with strangers?", "thoughtful"),
        "train": ("I love the train idea. The journey becomes part of the trip, doesn't it?", "enthusiastic"),
        "flight": ("Flying saves time, sure. Is the trip short enough that time matters most?", "thoughtful"),
        "fly": ("Flying saves time, sure. Is the trip short enough that time matters most?", "thoughtful"),
        "beach": ("A beach sounds lovely. Would you get bored, or is lazing about the point?", "enthusiastic"),
        "pack": ("Pack light, I'd say. What's the one thing you'd never leave behind?", "helpful"),
        "how long": ("A week feels about right to me. Long enough to relax, short enough to afford.", "helpful"),
    }
    GREETING_RE = re.compile(r"^\W*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I)

    def reply(self, last_text: str) -> tuple[str, str]:
        low = last_text.lower()
        for item, line in self.ITEM.items():
            if item in low:
                return line
        if not last_text or self.GREETING_RE.match(last_text):
            return self.GREETING
        return random.choice(self.GENERIC)


class ReplyEngine:
    """Writes Reachy's reply and picks its emotion: Claude, Ollama, or a pre-written line."""

    def __init__(self, cfg: Config, emotions: Emotions):
        self.cfg = cfg
        self.emotions = emotions
        self.templates = TemplateReplies()
        self.client = None
        context = (FREE_CHAT_CONTEXT if cfg.free_chat
                   else TASK_CONTEXT.format(items=", ".join(cfg.task_items)))
        self.system = SYSTEM_PROMPT.format(context=context, emotions=emotions.prompt_lines())
        # The reply must be exactly this shape, and the emotion one of the names. The emotion
        # comes first so that, when streaming, the movement is known before the first sentence.
        self.schema = {
            "type": "object",
            "properties": {"emotion": {"type": "string", "enum": list(emotions.names)},
                           "text": {"type": "string"}},
            "required": ["emotion", "text"],
            "additionalProperties": False,
        }
        if not cfg.use_llm:
            return
        try:
            if cfg.llm_backend == "ollama":
                import ollama

                self._ollama = ollama
                self.client = ollama.Client(host=cfg.ollama_host, timeout=cfg.llm_timeout_s)
                threading.Thread(target=self._warm_up_ollama, daemon=True).start()
            else:
                import anthropic

                from_file = "ANTHROPIC_API_KEY" in load_env_file()
                self._anthropic = anthropic
                self.client = anthropic.Anthropic(timeout=cfg.llm_timeout_s, max_retries=1)
                print(f"[nlp] replies: {cfg.llm_model} via Claude")
                if from_file:
                    print("[nlp] API key read from .env")
                elif os.environ.get("ANTHROPIC_API_KEY"):
                    print("[nlp] API key read from the environment (an exported key is used instead of .env)")
                elif not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
                    print("[nlp] no API key: paste it into .env (ANTHROPIC_API_KEY=...). Unless you ran "
                          "`ant auth login`, replies will be pre-written lines (or use --llm ollama)")
        except Exception as e:   # noqa: BLE001 - missing package or credentials
            print(f"[nlp] LLM unavailable, using pre-written replies ({e})")

    def _warm_up_ollama(self) -> None:
        """Load the model into memory now, so the first reply isn't slow."""
        try:
            self.client.generate(model=self.cfg.ollama_model, prompt="", keep_alive="30m")
            print(f"[nlp] replies: {self.cfg.ollama_model} via Ollama")
        except Exception as e:   # noqa: BLE001 - reported again, with a fallback, on the first reply
            print(f"[nlp] Ollama not ready ({e}). Is it running? brew services start ollama")

    def _prompt(self, history: list[tuple[str, str]]) -> str:
        transcript = "\n".join(f"{who}: {text}" for who, text in history[-self.cfg.history_lines:])
        last = history[-1][0] if history else "Someone"
        return f"Conversation so far:\n{transcript}\n\n{last} just spoke to you. Reply now as Reachy."

    def reply(self, history: list[tuple[str, str]], on_sentence: Optional[Callable[[str], None]] = None,
              on_emotion: Optional[Callable[[str], None]] = None) -> tuple[str, str, str]:
        """Returns (text, emotion, source).

        With on_sentence, each sentence is handed over as soon as it is ready (Claude streams,
        so Reachy can start talking before the reply is finished), and on_emotion(emotion) is
        called once, before the first sentence whenever possible.
        """
        if on_sentence is not None and self.client is not None and history \
                and self.cfg.llm_backend != "ollama":
            return self._reply_streaming(history, on_sentence, on_emotion)
        text, emotion, source = self._reply_whole(history)
        if on_emotion is not None:
            on_emotion(emotion)
        if on_sentence is not None:
            on_sentence(text)
        return text, emotion, source

    def _reply_whole(self, history: list[tuple[str, str]]) -> tuple[str, str, str]:
        if self.client is not None and history:
            ask = self._ask_ollama if self.cfg.llm_backend == "ollama" else self._ask_claude
            try:
                data = ask(self._prompt(history))
                text = clean_reply(data.get("text", ""), self.cfg.max_reply_sentences)
                if text:
                    return text, self.emotions.match(data.get("emotion")), "llm"
                print("[nlp] LLM gave an empty reply; using a pre-written one")
            except _FallBackToTemplate as e:
                print(f"[nlp] {e}; using a pre-written reply")
        return self._template(history)

    def _template(self, history: list[tuple[str, str]]) -> tuple[str, str, str]:
        last = next((t for who, t in reversed(history) if who != "Reachy"), "")
        text, emotion = self.templates.reply(last)
        return text, self.emotions.match(emotion), "template"

    def _reply_streaming(self, history, on_sentence, on_emotion) -> tuple[str, str, str]:
        """Stream Claude's reply sentence by sentence. If it fails before saying anything, a
        pre-written reply is said instead; a reply cut off part-way keeps what was said."""
        said: list[str] = []
        emotion: Optional[str] = None

        def emotion_ready(name: str) -> None:
            nonlocal emotion
            if emotion is None:
                emotion = self.emotions.match(name)
                if on_emotion is not None:
                    on_emotion(emotion)

        def sentence_ready(sentence: str) -> None:
            # One spoken chunk; it may hold a short sentence joined to the next ("Oh! Japan...").
            text = clean_reply(sentence, max_sentences=10)
            if text and len(said) < self.cfg.max_reply_sentences:
                if emotion is None:
                    emotion_ready(self.emotions.default)
                said.append(text)
                on_sentence(text)

        try:
            self._stream_claude(self._prompt(history), emotion_ready, sentence_ready)
        except _FallBackToTemplate as e:
            print(f"[nlp] {e}" + ("; keeping what was already said" if said else "; using a pre-written reply"))
        if not said:
            text, template_emotion, source = self._template(history)
            emotion_ready(template_emotion)    # no-op if the movement already started
            on_sentence(text)
            return text, emotion, source
        return " ".join(said), emotion, "llm"

    def _claude_request(self, prompt: str) -> dict:
        return dict(
            model=self.cfg.llm_model,
            max_tokens=2048,
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.cfg.llm_effort,
                           "format": {"type": "json_schema", "schema": self.schema}},
        )

    def _no_api_key(self, e: Exception) -> "_FallBackToTemplate":
        # The SDK found no API key or `ant auth login` profile. Say so once, not every reply.
        self.client = None
        return _FallBackToTemplate(
            "no Anthropic API key found, so replies will be pre-written lines from now on. "
            "Set ANTHROPIC_API_KEY, or use --llm ollama for free local replies")

    def _ask_claude(self, prompt: str) -> dict:
        a = self._anthropic
        request = self._claude_request(prompt)
        try:
            if self.cfg.llm_model in FALLBACK_MODELS:
                response = self.client.beta.messages.create(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                response = self.client.messages.create(**request)
        except TypeError as e:
            raise self._no_api_key(e) from e
        except (a.APIConnectionError, a.RateLimitError, a.APIStatusError) as e:
            raise _FallBackToTemplate(f"LLM error: {e}") from e
        if response.stop_reason in ("refusal", "max_tokens"):
            raise _FallBackToTemplate(f"LLM stopped early ({response.stop_reason})")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise _FallBackToTemplate("LLM reply wasn't valid JSON") from e

    def _claude_deltas(self, prompt: str):
        """Yield the reply's raw JSON text as it streams in. Only SDK errors are turned into
        _FallBackToTemplate here; errors in the caller's code are left alone."""
        a = self._anthropic
        request = self._claude_request(prompt)
        try:
            if self.cfg.llm_model in FALLBACK_MODELS:
                manager = self.client.beta.messages.stream(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                manager = self.client.messages.stream(**request)
            with manager as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except TypeError as e:
            raise self._no_api_key(e) from e
        except (a.APIConnectionError, a.RateLimitError, a.APIStatusError) as e:
            raise _FallBackToTemplate(f"LLM error: {e}") from e
        if final.stop_reason in ("refusal", "max_tokens"):
            raise _FallBackToTemplate(f"LLM stopped early ({final.stop_reason})")

    def _stream_claude(self, prompt: str, on_emotion: Callable[[str], None],
                       on_sentence: Callable[[str], None]) -> None:
        parser = ReplyStreamParser()
        for delta in self._claude_deltas(prompt):
            emotion, sentences = parser.feed(delta)
            if emotion is not None:
                on_emotion(emotion)
            for sentence in sentences:
                on_sentence(sentence)
        for sentence in parser.finish():
            on_sentence(sentence)

    def _ask_ollama(self, prompt: str) -> dict:
        import httpx

        o = self._ollama
        try:
            response = self.client.chat(
                model=self.cfg.ollama_model,
                messages=[{"role": "system", "content": self.system},
                          {"role": "user", "content": prompt}],
                format=self.schema,
                keep_alive="30m",
                options={"num_predict": 150, "temperature": 0.8},
            )
            return json.loads(response["message"]["content"])
        except (o.ResponseError, o.RequestError, httpx.HTTPError, ConnectionError) as e:
            raise _FallBackToTemplate(f"Ollama error: {e}") from e
        except json.JSONDecodeError as e:
            raise _FallBackToTemplate("Ollama reply wasn't valid JSON") from e


class _FallBackToTemplate(Exception):
    pass


# ---------------------------------------------------------------------------
# The model the controller calls
# ---------------------------------------------------------------------------

LOG_FIELDS = ("t", "utterance_id", "text", "intent", "said_name", "answering_robot",
              "interrupting", "robot_speaking", "replied_to", "reply_emotion", "side",
              "doa_angle", "doa_speech", "mic_rms", "vad_prob")
CONVERSATION_FIELDS = ("time", "t", "speaker", "text", "utterance_id", "intent", "said_name",
                       "answering_robot", "interrupting", "reply_to", "emotion", "reply_source")


class SpeakerNLPModel:
    def __init__(self, cfg: Optional[Config] = None, mini=None, mic=None, emotions=None,
                 voice=None, doa=None, engine: Optional[ReplyEngine] = None):
        """
        mini:     the controller's ReachyMini connection, only used for the sound direction.
        mic:      RobotMic(mini) on the robot, LaptopMic(cfg) on a laptop, None for typed input.
        emotions: the controller's emotion names (a list, or a dict whose keys are the names).
        voice:    the ReachyVoice the controller speaks with; only `voice.speaking` is read, to
                  ignore Reachy's own voice and to notice interruptions.
        """
        self.cfg = cfg = cfg or Config()
        self.emotions = Emotions(emotions)
        self.voice = voice
        self.doa = doa or (RobotDoA(mini) if mini is not None else MockDoA())
        self.mic = mic
        self.engine = engine or ReplyEngine(cfg, self.emotions)
        self.segmenter = UtteranceSegmenter(cfg)
        self.vad = make_vad(cfg) if mic is not None else EnergyVAD(cfg)
        self.transcriber = Transcriber(cfg) if mic is not None else None

        self._lock = threading.Lock()            # respond() may run in another thread
        self._reply_lock = threading.Lock()      # one reply at a time
        self._typed: queue.Queue = queue.Queue()
        self.history: list[tuple[str, str]] = []
        self._sides: list = []
        self._held = None                        # unfinished sentence waiting for the rest
        self._last_in_speech_t = 0.0
        self._next_id = 1
        self._last_id: Optional[int] = None

        self._was_speaking = False
        self._robot_end_t = -1e9                 # when Reachy last stopped talking
        self._robot_said_t = -1e9                # when the last reply was written
        self._robot_asked = False                # the last reply ended with a question
        self._loud_since: Optional[float] = None
        self._interrupting = False
        self._capturing = False                  # recording the sentence that interrupted
        self._prebuffer: collections.deque = collections.deque(maxlen=8)   # ~0.8 s of audio
        self._replied: Optional[tuple[int, str]] = None
        self._reply_active = False               # a streamed reply is being spoken
        self._next_job = 1                       # speech-to-text jobs (see _listen)
        self._early_job: Optional[int] = None    # transcribing the sentence before it's over
        self._early_job_seq = 0                  # ...of this segmenter snapshot
        self._job_text: dict[int, str] = {}      # finished, not yet known to be final
        self._job_meta: dict[int, tuple] = {}    # final, waiting for the text
        self._ready: collections.deque = collections.deque()
        self._log_file = self._log = self._conv_file = self._conv = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self.mic is not None:
            self.transcriber.start()
            self.mic.start()
        if self.cfg.log_csv:
            self._log_file, self._log = self._open_csv(self.cfg.log_csv, LOG_FIELDS)
        if self.cfg.conversation_csv:
            self._conv_file, self._conv = self._open_csv(self.cfg.conversation_csv, CONVERSATION_FIELDS)

    def stop(self) -> None:
        if self.mic is not None:
            self.mic.stop()
        for f in (self._log_file, self._conv_file):
            if f:
                f.close()
        self._log_file = self._conv_file = self._log = self._conv = None

    def inject_transcript(self, text: str, side: Optional[str] = None) -> None:
        """Typed input for testing without a microphone."""
        now = time.monotonic()
        self._typed.put((text, side, now, now, self._robot_speaking(), False))

    # -- the two calls the controller uses --------------------------------

    def update(self) -> dict:
        """Call about 10 times a second. Returns {"utterance": dict or None, "interrupting": bool}."""
        now = time.monotonic()
        speaking = self._track_robot_speaking(now)
        audio = self.mic.read() if self.mic is not None else None
        mic_rms = rms(audio)
        voiced = self.vad.is_speech(audio) if audio is not None else False

        reading = self.doa.read(voiced)
        angle, doa_speech = reading if reading is not None else (None, False)
        side = doa_to_side(angle, self.cfg) if doa_speech else None

        if audio is not None:
            if speaking and not self._capturing:
                if self.cfg.allow_interruptions:
                    self._check_interrupt(audio, mic_rms, voiced, now)
                # else: Reachy's own voice; drop it
            elif self._capturing or now - self._robot_end_t >= self.cfg.echo_guard_s:
                self._listen(audio, voiced, side, now)

        utt = self._next_utterance(now)
        # A typed sentence sent while Reachy talks also counts, for testing without a mic.
        interrupting = speaking and (self._interrupting or bool(utt and utt["interrupting"]))
        self._write_log(utt, interrupting, speaking, side, angle, doa_speech, mic_rms)
        return {"utterance": utt, "interrupting": interrupting}

    def respond(self, utterance_id: Optional[int] = None,
                on_sentence: Optional[Callable[[str], None]] = None,
                on_emotion: Optional[Callable[[str], None]] = None) -> dict:
        """Write Reachy's reply to the conversation so far (blocks for about 1 s).

        `utterance_id` is only recorded as `reply_to`; it defaults to the latest utterance.
        Returns {"text", "emotion", "reply_to", "source"}. It never plays sound itself.

        Streaming: pass on_sentence (e.g. voice.say) to get each sentence as soon as it is
        written, and on_emotion to get the emotion first. From the first sentence until this
        returns, the microphone treats Reachy as talking, so gaps between sentences aren't
        mistaken for the end of its turn.
        """
        def sentence_ready(sentence: str) -> None:
            self._reply_active = True
            on_sentence(sentence)

        with self._reply_lock:
            with self._lock:
                history = list(self.history)
                reply_to = utterance_id if utterance_id is not None else self._last_id
            try:
                text, emotion, source = self.engine.reply(
                    history, on_sentence=sentence_ready if on_sentence else None, on_emotion=on_emotion)
            finally:
                self._reply_active = False
            with self._lock:
                self.history.append(("Reachy", text))
                self._robot_asked = text.rstrip().endswith("?")
                self._robot_said_t = time.monotonic()
                self._replied = (reply_to, emotion)
            self._log_line("Reachy", text, reply_to=reply_to, emotion=emotion, reply_source=source)
            return {"text": text, "emotion": emotion, "reply_to": reply_to, "source": source}

    # -- helpers ----------------------------------------------------------

    def _robot_speaking(self) -> bool:
        return self._reply_active or bool(self.voice is not None and self.voice.speaking)

    def _track_robot_speaking(self, now: float) -> bool:
        speaking = self._robot_speaking()
        if self._was_speaking and not speaking:
            self._robot_end_t = now
            self._interrupting = False
            self._loud_since = None
            self._prebuffer.clear()
        self._was_speaking = speaking
        return speaking

    def _check_interrupt(self, audio: np.ndarray, mic_rms: float, voiced: bool, now: float) -> None:
        """While Reachy talks the mic is ignored, unless someone is clearly louder than Reachy."""
        self._prebuffer.append(audio)
        if mic_rms < self.cfg.interrupt_rms:
            self._loud_since = None
            return
        if self._loud_since is None:
            self._loud_since = now
        if now - self._loud_since >= self.cfg.interrupt_s:
            print("[nlp] someone is talking over Reachy")
            self._interrupting = self._capturing = True
            # Start the sentence from the buffered audio, so its first words aren't lost.
            for chunk in self._prebuffer:
                self.segmenter.feed(chunk, now, voiced=True)
            self._prebuffer.clear()

    def _listen(self, audio: np.ndarray, voiced: bool, side: Optional[str], now: float) -> None:
        self._collect_transcripts()                  # an early transcript may decide the pause
        if side is not None:
            self._sides.append(side)
        sentence = self.segmenter.feed(audio, now, voiced)
        early = self.segmenter.pop_early()
        if early is not None:
            # A short pause: start transcribing now, in case the sentence is over.
            self._early_job = self._submit(early)
            self._early_job_seq = self.segmenter.early_seq
        if sentence is not None:
            sides = [s for s in self._sides if s is not None]
            majority = max(set(sides), key=sides.count) if sides else None
            if self.segmenter.final_matches_early and self._early_job is not None:
                job = self._early_job                # nobody spoke again: reuse it
            else:
                job = self._submit(sentence)
            self._early_job = None
            self._finalize(job, (majority, self.segmenter.last_start_t,
                                 self.segmenter.last_end_t, self._capturing,
                                 self.segmenter.last_waited))
            self._sides = []
        if self._capturing and not self.segmenter.in_speech:
            self._capturing = False

    def _submit(self, audio: np.ndarray) -> int:
        job, self._next_job = self._next_job, self._next_job + 1
        self.transcriber.submit(job, audio)
        return job

    def _finalize(self, job: int, meta: tuple) -> None:
        """The sentence is over and `job` transcribes it. Older unfinished jobs are stale."""
        for stale in [j for j in self._job_text if j < job]:
            del self._job_text[stale]
        if job in self._job_text:
            self._deliver(self._job_text.pop(job), meta)
        else:
            self._job_meta[job] = meta

    def _deliver(self, text: str, meta: tuple) -> None:
        if text:
            side, start_t, end_t, interrupting, waited = meta
            self._ready.append((text, side, start_t, end_t, interrupting, waited))

    def _collect_transcripts(self) -> None:
        while self.transcriber is not None and not self.transcriber.results.empty():
            job, text = self.transcriber.results.get_nowait()
            if job in self._job_meta:
                self._deliver(text, self._job_meta.pop(job))
            elif job == self._early_job:
                self._job_text[job] = text           # early: we'll know soon if it's final
                self.segmenter.set_early_verdict(self._early_job_seq, not looks_unfinished(text))
            # else: an early transcript of a sentence that went on; not needed

    def _next_transcript(self):
        if not self._typed.empty():
            return self._typed.get_nowait()
        self._collect_transcripts()
        return self._ready.popleft() if self._ready else None

    def _join_unfinished(self, item, now: float):
        """Hold "I think Japan is nice because" until the rest of the sentence arrives.

        item: (text, side, start_t, end_t, interrupting, waited). A sentence ending in "?" is
        never held: it's finished, and waiting would only delay Reachy's answer. Nor is one that
        already got the longer pause while it was being recorded (waited).
        """
        if self.segmenter.in_speech:
            self._last_in_speech_t = now
        held = self._held
        if item is not None and held is not None:
            self._held = None
            return (f"{held[0]} {item[0]}", held[1] or item[1], held[2], item[3], held[4] or item[4],
                    item[5])
        if (item is not None and not item[5] and UNFINISHED_RE.search(item[0])
                and not item[0].rstrip().endswith("?")):
            self._held = (*item, now)
            return None
        if held is not None and now - max(held[6], self._last_in_speech_t) > self.cfg.unfinished_wait_s:
            self._held = None
            return held[:6]
        return item

    def _next_utterance(self, now: float) -> Optional[dict]:
        item = self._join_unfinished(self._next_transcript(), now)
        if item is None:
            return None
        text, side, start_t, end_t, interrupting, _ = item
        c = classify_utterance(text)
        with self._lock:
            answering = self._answer_expected(start_t)
            self._robot_asked = False
            self.history.append((SIDE_LABEL[side], c["text"]))
            uid, self._next_id = self._next_id, self._next_id + 1
            self._last_id = uid
        wall_offset = time.time() - now
        utt = {"id": uid, "t": round(start_t + wall_offset, 2), "end_t": round(end_t + wall_offset, 2),
               **c, "answering_robot": answering, "interrupting": bool(interrupting)}
        self._log_line(SIDE_LABEL[side], c["text"], t=utt["t"], utterance_id=uid, intent=c["intent"],
                       said_name=c["said_name"], answering_robot=answering, interrupting=utt["interrupting"])
        return utt

    def _answer_expected(self, start_t: float) -> bool:
        """Only the first sentence after Reachy's question counts as the answer. The window runs
        from when Reachy stopped talking to when the person *started*, so a long answer counts."""
        robot_done_t = max(self._robot_said_t, self._robot_end_t)
        return (self._robot_asked and self.cfg.followup_window_s > 0
                and start_t - robot_done_t <= self.cfg.followup_window_s)

    @staticmethod
    def _open_csv(path: str, fields):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        f = open(path, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0:
            writer.writeheader()
        return f, writer

    def _log_line(self, speaker: str, text: str, t: Optional[float] = None, **fields) -> None:
        """One row per line said, by a person or Reachy, written straight to disk."""
        with self._lock:
            if not self._conv:
                return
            t = round(time.time(), 2) if t is None else t
            self._conv.writerow({"time": time.strftime("%H:%M:%S", time.localtime(t)), "t": t,
                                 "speaker": speaker, "text": text, **fields})
            self._conv_file.flush()

    def _write_log(self, utt, interrupting, speaking, side, angle, doa_speech, mic_rms) -> None:
        with self._lock:
            replied, self._replied = self._replied, None
            if not self._log:
                return
            utt = utt or {}
            self._log.writerow({
                "t": round(time.time(), 3), "utterance_id": utt.get("id"), "text": utt.get("text"),
                "intent": utt.get("intent"), "said_name": utt.get("said_name"),
                "answering_robot": utt.get("answering_robot"), "interrupting": interrupting,
                "robot_speaking": speaking, "replied_to": replied[0] if replied else None,
                "reply_emotion": replied[1] if replied else None, "side": side,
                "doa_angle": None if angle is None else round(angle, 3), "doa_speech": doa_speech,
                "mic_rms": round(mic_rms, 4), "vad_prob": round(getattr(self.vad, "prob", 0.0), 3),
            })


# ---------------------------------------------------------------------------
# Standalone test (laptop only; never connects to the robot)
# ---------------------------------------------------------------------------

HELP = """\
Commands (type, then Enter):
  l: <sentence>     Person Left says <sentence>   (typed mode; "r:" for Person Right)
  <sentence>        someone says <sentence>       (typed mode)
  l / r / f         which side you're on: left / right / front (mic mode)
  respond           make Reachy answer now (what the controller does)
  q                 quit"""


def watch_conversation(path: str, show_all: bool) -> None:
    """Print the conversation log as it grows, one readable line per thing said."""
    while not os.path.exists(path):
        print(f"waiting for {path} ...", flush=True)
        time.sleep(1)
    with open(path, newline="") as f:
        header = f.readline().strip().split(",")
        if not show_all:
            f.seek(0, os.SEEK_END)
        print(f"watching {path}  (Ctrl-C to stop)", flush=True)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            for row in csv.DictReader(io.StringIO(line), fieldnames=header):
                if row["speaker"] == "Reachy":
                    print(f"{row['time']}  Reachy        {row['text']}  "
                          f"({row.get('emotion')}, reply to #{row['reply_to']}, {row['reply_source']})", flush=True)
                else:
                    flags = [row["intent"]] + [name.replace("_", " ") for name in
                                               ("said_name", "answering_robot", "interrupting")
                                               if row.get(name) == "True"]
                    print(f"{row['time']}  #{row['utterance_id']:<3} {row['speaker']:<12} {row['text']}  "
                          f"[{', '.join(flags)}]", flush=True)


def _stdin_loop(model: SpeakerNLPModel, typed: bool, commands: queue.Queue) -> None:
    sides = {"l": "left", "r": "right", "f": "front"}
    for line in sys.stdin:
        line = line.strip()
        if line == "q":
            break
        if line == "respond":
            commands.put("respond")
        elif line in sides and isinstance(model.doa, MockDoA):
            model.doa.set_side(sides[line])
        elif typed and len(line) > 2 and line[1] == ":" and line[0] in "lr":
            model.inject_transcript(line[2:].strip(), sides[line[0]])
        elif typed and line:
            model.inject_transcript(line, None)
    commands.put("quit")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--typed", action="store_true", help="type what people say instead of using the mic")
    p.add_argument("--speak", action="store_true", help="speak replies aloud with reachy_voice.py")
    p.add_argument("--manual", action="store_true",
                   help="only answer after `respond`; otherwise Reachy answers when you say its name")
    p.add_argument("--llm", choices=("claude", "ollama"), default="claude")
    p.add_argument("--llm-model", help=f"default {Config.llm_model} / {Config.ollama_model}")
    p.add_argument("--no-llm", action="store_true", help="pre-written replies only")
    p.add_argument("--free-chat", action="store_true", help="no trip-planning task; answers everything")
    p.add_argument("--stt-model", default=Config.stt_model)
    p.add_argument("--watch", action="store_true", help="show the conversation log live, then exit")
    p.add_argument("--all", action="store_true", help="with --watch: show the whole log first")
    args = p.parse_args()

    if args.watch:
        try:
            watch_conversation(Config.conversation_csv, args.all)
        except KeyboardInterrupt:
            pass
        return

    cfg = Config(stt_model=args.stt_model, use_llm=not args.no_llm, llm_backend=args.llm,
                 free_chat=args.free_chat)
    if args.llm_model:
        setattr(cfg, "ollama_model" if args.llm == "ollama" else "llm_model", args.llm_model)
    voice = None
    if args.speak:
        from reachy_voice import ReachyVoice

        voice = ReachyVoice()
    model = SpeakerNLPModel(cfg, mic=None if args.typed else LaptopMic(cfg), voice=voice)
    print(HELP)
    print("Reachy answers when you say its name" + (" (or anything, in free chat)" if args.free_chat else "")
          + "." if not args.manual else "Reachy only answers after `respond`.")
    commands: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_loop, args=(model, args.typed, commands), daemon=True).start()
    model.start()

    def answer(uid=None):
        reply = model.respond(uid)
        print(f"[reachy] ({reply['emotion']}) {reply['text']}   [{reply['source']}]")
        if voice is not None:
            voice.say(reply["text"])

    try:
        while True:
            out = model.update()
            if out["interrupting"] and voice is not None and voice.speaking:
                voice.stop()                       # what the controller does
            utt = out["utterance"]
            if utt:
                flags = [utt["intent"]] + [k.replace("_", " ") for k in
                                           ("said_name", "answering_robot", "interrupting") if utt[k]]
                print(f"[heard #{utt['id']}] {utt['text']}   ({', '.join(flags)})")
                if not args.manual and (utt["said_name"] or utt["answering_robot"] or args.free_chat):
                    answer(utt["id"])
            try:
                command = commands.get_nowait()
            except queue.Empty:
                command = None
            if command == "quit":
                break
            if command == "respond":
                answer()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        model.stop()
        if voice is not None:
            voice.stop()


if __name__ == "__main__":
    main()
