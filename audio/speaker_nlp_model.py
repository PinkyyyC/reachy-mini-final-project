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
from typing import Optional

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
    silence_end_s: float = 0.5           # pause that ends a sentence
    min_utterance_s: float = 0.35
    max_utterance_s: float = 15.0
    unfinished_wait_s: float = 1.2       # "...is nice because" -> wait this long for the rest

    # Speech-to-text.
    stt_model: str = "base.en"           # "tiny.en" is faster, "small.en" more accurate

    # While Reachy talks.
    echo_guard_s: float = 0.5            # ignore the mic this long after Reachy stops talking
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
    """Cuts the microphone stream into sentences, using the speech detector and a pause."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()
        self._last_voice_t = 0.0
        self._start_t = 0.0
        self.last_start_t = 0.0          # when the sentence just returned began

    def reset(self) -> None:
        self._buf: list[np.ndarray] = []
        self._voiced_s = 0.0
        self.in_speech = False

    def feed(self, chunk: np.ndarray, now: float, voiced: Optional[bool] = None) -> Optional[np.ndarray]:
        loud = rms(chunk) > self.cfg.vad_rms if voiced is None else voiced
        if loud:
            self._last_voice_t = now
            if not self.in_speech:
                self._start_t = now
            self.in_speech = True
        if not self.in_speech:
            return None
        self._buf.append(chunk)
        if loud:
            self._voiced_s += len(chunk) / self.cfg.sample_rate
        total_s = sum(len(b) for b in self._buf) / self.cfg.sample_rate
        if now - self._last_voice_t >= self.cfg.silence_end_s or total_s >= self.cfg.max_utterance_s:
            audio, voiced_s = np.concatenate(self._buf), self._voiced_s
            self.last_start_t = self._start_t
            self.reset()
            if voiced_s >= self.cfg.min_utterance_s:
                return audio
        return None


class Transcriber:
    """faster-whisper in a background thread, so update() never waits for it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._in: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()

    def start(self) -> None:
        from faster_whisper import WhisperModel   # fail here, not silently inside the thread

        threading.Thread(target=self._loop, args=(WhisperModel,), daemon=True).start()

    def submit(self, audio: np.ndarray, side: Optional[str], start_t: float, interrupting: bool) -> None:
        self._in.put((audio, side, start_t, interrupting))

    def _loop(self, WhisperModel) -> None:
        # The first run downloads the model (about 150 MB for base.en).
        model = WhisperModel(self.cfg.stt_model, device="cpu", compute_type="int8")
        print(f"[nlp] speech-to-text ready ({self.cfg.stt_model})")
        while True:
            audio, side, start_t, interrupting = self._in.get()
            # No vad_filter: Silero already cut this sentence, so a second pass only costs time.
            segments, _ = model.transcribe(
                audio, language="en", beam_size=1, vad_filter=False, without_timestamps=True,
                condition_on_previous_text=False,
                hotwords="Reachy " + " ".join(self.cfg.task_items),   # nudges "Richie" -> "Reachy"
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                self.results.put((text, side, start_t, interrupting))


# ---------------------------------------------------------------------------
# What was said
# ---------------------------------------------------------------------------

# Whisper often mishears the robot's name; these spellings are rewritten to "Reachy".
# "Rachel" is a real name: drop it from this list if a participant is called Rachel.
MISHEARD_NAME_RE = re.compile(
    r"\b(reachie|reechy|reachi|reeky|reiki|reaky|richie|ritchie|richy|ricci|rachel)\b", re.I)
NAME_RE = re.compile(r"\b(reachy|reach me|robot)\b", re.I)
QUESTION_START_RE = re.compile(
    r"^((hey|hi|so|and|but|ok(ay)?|reachy)\W+)*"
    r"(what|why|how|which|who|where|when|should|could|would|do|does|did|is|are|can|shall|will)\b",
    re.I)
# Ends on a word nobody finishes a sentence with: they paused mid-thought.
UNFINISHED_RE = re.compile(
    r"\b(because|and|but|so|or|then|like|um|uh|the|a|an|to|of|with|for|if|that|my|your|is|are)"
    r"\W*$", re.I)


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
        # The reply must be exactly this shape, and the emotion one of the names.
        self.schema = {
            "type": "object",
            "properties": {"text": {"type": "string"},
                           "emotion": {"type": "string", "enum": list(emotions.names)}},
            "required": ["text", "emotion"],
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

    def reply(self, history: list[tuple[str, str]]) -> tuple[str, str, str]:
        """Returns (text, emotion, source)."""
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
        last = next((t for who, t in reversed(history) if who != "Reachy"), "")
        text, emotion = self.templates.reply(last)
        return text, self.emotions.match(emotion), "template"

    def _ask_claude(self, prompt: str) -> dict:
        a = self._anthropic
        request = dict(
            model=self.cfg.llm_model,
            max_tokens=2048,
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.cfg.llm_effort,
                           "format": {"type": "json_schema", "schema": self.schema}},
        )
        try:
            if self.cfg.llm_model in FALLBACK_MODELS:
                response = self.client.beta.messages.create(
                    **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                response = self.client.messages.create(**request)
        except TypeError as e:
            # The SDK found no API key or `ant auth login` profile. Say so once, not every reply.
            self.client = None
            raise _FallBackToTemplate(
                "no Anthropic API key found, so replies will be pre-written lines from now on. "
                "Set ANTHROPIC_API_KEY, or use --llm ollama for free local replies") from e
        except (a.APIConnectionError, a.RateLimitError, a.APIStatusError) as e:
            raise _FallBackToTemplate(f"LLM error: {e}") from e
        if response.stop_reason in ("refusal", "max_tokens"):
            raise _FallBackToTemplate(f"LLM stopped early ({response.stop_reason})")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise _FallBackToTemplate("LLM reply wasn't valid JSON") from e

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
        self._typed.put((text, side, time.monotonic(), self._robot_speaking()))

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
                self._check_interrupt(audio, mic_rms, voiced, now)
            elif self._capturing or now - self._robot_end_t >= self.cfg.echo_guard_s:
                self._listen(audio, voiced, side, now)

        utt = self._next_utterance(now)
        # A typed sentence sent while Reachy talks also counts, for testing without a mic.
        interrupting = speaking and (self._interrupting or bool(utt and utt["interrupting"]))
        self._write_log(utt, interrupting, speaking, side, angle, doa_speech, mic_rms)
        return {"utterance": utt, "interrupting": interrupting}

    def respond(self, utterance_id: Optional[int] = None) -> dict:
        """Write Reachy's reply to the conversation so far (blocks for about 1 s).

        `utterance_id` is only recorded as `reply_to`; it defaults to the latest utterance.
        Returns {"text", "emotion", "reply_to", "source"}. It never plays sound.
        """
        with self._reply_lock:
            with self._lock:
                history = list(self.history)
                reply_to = utterance_id if utterance_id is not None else self._last_id
            text, emotion, source = self.engine.reply(history)
            with self._lock:
                self.history.append(("Reachy", text))
                self._robot_asked = text.rstrip().endswith("?")
                self._robot_said_t = time.monotonic()
                self._replied = (reply_to, emotion)
            self._log_line("Reachy", text, reply_to=reply_to, emotion=emotion, reply_source=source)
            return {"text": text, "emotion": emotion, "reply_to": reply_to, "source": source}

    # -- helpers ----------------------------------------------------------

    def _robot_speaking(self) -> bool:
        return bool(self.voice is not None and self.voice.speaking)

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
        if side is not None:
            self._sides.append(side)
        sentence = self.segmenter.feed(audio, now, voiced)
        if sentence is not None:
            sides = [s for s in self._sides if s is not None]
            majority = max(set(sides), key=sides.count) if sides else None
            self.transcriber.submit(sentence, majority, self.segmenter.last_start_t, self._capturing)
            self._sides = []
        if self._capturing and not self.segmenter.in_speech:
            self._capturing = False

    def _next_transcript(self):
        for q in (self._typed, self.transcriber.results if self.transcriber else None):
            if q is not None and not q.empty():
                return q.get_nowait()
        return None

    def _join_unfinished(self, item, now: float):
        """Hold "I think Japan is nice because" until the rest of the sentence arrives."""
        if self.segmenter.in_speech:
            self._last_in_speech_t = now
        held = self._held
        if item is not None and held is not None:
            self._held = None
            return (f"{held[0]} {item[0]}", held[1] or item[1], held[2], held[3] or item[3])
        if item is not None and UNFINISHED_RE.search(item[0]):
            self._held = (*item, now)
            return None
        if held is not None and now - max(held[4], self._last_in_speech_t) > self.cfg.unfinished_wait_s:
            self._held = None
            return held[:4]
        return item

    def _next_utterance(self, now: float) -> Optional[dict]:
        item = self._join_unfinished(self._next_transcript(), now)
        if item is None:
            return None
        text, side, start_t, interrupting = item
        c = classify_utterance(text)
        with self._lock:
            answering = self._answer_expected(start_t)
            self._robot_asked = False
            self.history.append((SIDE_LABEL[side], c["text"]))
            uid, self._next_id = self._next_id, self._next_id + 1
            self._last_id = uid
        utt = {"id": uid, "t": round(time.time() - (now - start_t), 2), **c,
               "answering_robot": answering, "interrupting": bool(interrupting)}
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
