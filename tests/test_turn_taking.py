"""Tests for Reachy's faster replies: streaming, end-of-speech detection, early
transcription and the voice. (When Reachy decides to answer is unchanged by this work.)

Run from the project root:  python -m pytest tests
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from audio.speaker_nlp_model import (Config, ReplyEngine, ReplyStreamParser, SpeakerNLPModel,
                                     classify_utterance, _FallBackToTemplate)


# ---------------------------------------------------------------------------
# Streaming the JSON reply
# ---------------------------------------------------------------------------

REPLY = '{"emotion": "enthusiastic", "text": "Oh! Japan is lovely in spring. Have you two been before?"}'


def stream_char_by_char(raw):
    parser, events = ReplyStreamParser(), []
    for ch in raw:
        emotion, sentences = parser.feed(ch)
        if emotion:
            events.append(("emotion", emotion))
        events += [("sentence", s) for s in sentences]
    events += [("sentence", s) for s in parser.finish()]
    return events


def test_stream_gives_emotion_first_then_sentences():
    events = stream_char_by_char(REPLY)
    assert events == [
        ("emotion", "enthusiastic"),
        ("sentence", "Oh! Japan is lovely in spring."),     # "Oh!" joined to the next sentence
        ("sentence", "Have you two been before?"),
    ]


def test_first_sentence_arrives_before_the_reply_ends():
    parser = ReplyStreamParser()
    cut = REPLY.index("Have")
    _, sentences = parser.feed(REPLY[:cut])
    assert sentences == ["Oh! Japan is lovely in spring."]


def test_stream_decodes_escapes():
    raw = '{"emotion": "helpful", "text": "I\'d say \\"pack light\\". It\\u2019s easier."}'
    sentences = [s for kind, s in stream_char_by_char(raw) if kind == "sentence"]
    assert sentences == ['I\'d say "pack light".', "It’s easier."]


# ---------------------------------------------------------------------------
# respond() with streaming, using a fake Claude stream
# ---------------------------------------------------------------------------

def make_model(chunks=None, error_after=None):
    cfg = Config(use_llm=False, log_csv=None, conversation_csv=None)
    model = SpeakerNLPModel(cfg, mic=None)
    engine = model.engine
    engine.cfg.llm_backend = "claude"
    engine.client = object()        # pretend Claude is configured

    def fake_deltas(prompt):
        for i, chunk in enumerate(chunks or []):
            if error_after is not None and i == error_after:
                raise _FallBackToTemplate("LLM error: connection lost")
            yield chunk

    engine._claude_deltas = fake_deltas
    model.history.append(("Person Left", "Reachy, should we go to Japan?"))
    return model


def chunks_of(raw, n=7):
    return [raw[i:i + n] for i in range(0, len(raw), n)]


def test_respond_streams_and_counts_as_speaking():
    model = make_model(chunks_of(REPLY))
    events, speaking_during = [], []

    def on_sentence(s):
        events.append(("sentence", s))
        speaking_during.append(model._robot_speaking())

    reply = model.respond(on_sentence=on_sentence, on_emotion=lambda e: events.append(("emotion", e)))
    assert events[0] == ("emotion", "enthusiastic")
    assert [s for k, s in events if k == "sentence"] == ["Oh! Japan is lovely in spring.",
                                                       "Have you two been before?"]
    assert all(speaking_during)                 # the mic ignores Reachy between sentences
    assert not model._robot_speaking()          # ...and listens again afterwards
    assert reply["text"] == "Oh! Japan is lovely in spring. Have you two been before?"
    assert reply["source"] == "llm"
    assert model.history[-1] == ("Reachy", reply["text"])


def test_respond_keeps_at_most_two_sentences():
    raw = '{"emotion": "helpful", "text": "One two three. Four five six. Seven eight nine."}'
    model = make_model(chunks_of(raw))
    said = []
    reply = model.respond(on_sentence=said.append)
    assert said == ["One two three.", "Four five six."]
    assert reply["text"] == "One two three. Four five six."


def test_stream_failure_before_any_sentence_uses_a_template():
    model = make_model(chunks_of(REPLY), error_after=0)
    said, emotions = [], []
    reply = model.respond(on_sentence=said.append, on_emotion=emotions.append)
    assert reply["source"] == "template" and said == [reply["text"]]
    assert emotions == [reply["emotion"]]


def test_stream_failure_part_way_keeps_what_was_said():
    cut = REPLY.index("Have") + 2
    model = make_model([REPLY[:cut], REPLY[cut:]], error_after=1)   # dies after sentence one
    said = []
    reply = model.respond(on_sentence=said.append)
    assert said == ["Oh! Japan is lovely in spring."]
    assert reply["text"] == "Oh! Japan is lovely in spring." and reply["source"] == "llm"


def test_respond_without_streaming_is_unchanged():
    cfg = Config(use_llm=False, log_csv=None, conversation_csv=None)
    model = SpeakerNLPModel(cfg, mic=None)
    model.history.append(("Person Left", "Hi Reachy"))
    reply = model.respond()
    assert reply["source"] == "template" and reply["text"]


# ---------------------------------------------------------------------------
# Sentence timing and the unfinished-sentence wait
# ---------------------------------------------------------------------------

def typed_model():
    return SpeakerNLPModel(Config(use_llm=False, log_csv=None, conversation_csv=None), mic=None)


def test_utterance_has_start_and_end_times():
    model = typed_model()
    before = time.time()
    model.inject_transcript("Reachy, what about Japan?")
    u = model.update()["utterance"]
    assert u["said_name"] and before - 1 <= u["t"] <= u["end_t"] <= time.time() + 1


def test_question_is_never_held_as_unfinished():
    model = typed_model()
    model.inject_transcript("Is it cheaper to take the?")
    assert model.update()["utterance"] is not None


def test_unfinished_statement_waits_for_the_rest():
    model = typed_model()
    model.inject_transcript("I think Japan is nice because")
    assert model.update()["utterance"] is None
    model.inject_transcript("the food is great")
    u = model.update()["utterance"]
    assert u["text"] == "I think Japan is nice because the food is great"


# ---------------------------------------------------------------------------
# Early transcription: start speech-to-text at a short pause, reuse it if the sentence is over
# ---------------------------------------------------------------------------

import queue
import numpy as np
from audio.speaker_nlp_model import UtteranceSegmenter

CHUNK = 800                      # 50 ms at 16 kHz
LOUD = np.full(CHUNK, 0.1, np.float32)
QUIET = np.zeros(CHUNK, np.float32)


class FakeTranscriber:
    def __init__(self):
        self.jobs, self.results = [], queue.Queue()

    def submit(self, job_id, audio):
        self.jobs.append((job_id, len(audio)))

    def finish(self, job_id, text):
        self.results.put((job_id, text))


def mic_model():
    cfg = Config(use_llm=False, log_csv=None, conversation_csv=None, vad="energy")
    model = SpeakerNLPModel(cfg, mic=None)
    model.transcriber = FakeTranscriber()
    return model


def speak(model, t, seconds, loud=True):
    """Feed `seconds` of loud or quiet audio starting at time t; returns the end time."""
    for _ in range(round(seconds / 0.05)):
        model._listen(LOUD if loud else QUIET, loud, None, t)
        t += 0.05
    return t


def run_segmenter(seg, speech_s, pause_s, verdict=None):
    """speech_s of speech then up to pause_s of quiet. verdict: what the early transcript says
    (True finished, False unfinished, None never arrives). Returns (early_at, final_at)."""
    t, early_at, final_at = 0.0, None, None
    for i in range(round((speech_s + pause_s) / 0.05)):
        loud = i < round(speech_s / 0.05)
        out = seg.feed(LOUD if loud else QUIET, t, loud)
        if seg.pop_early() is not None:
            early_at = t
            if verdict is not None:
                seg.set_early_verdict(seg.early_seq, verdict)
        if out is not None:
            final_at = t
            break
        t += 0.05
    return early_at, final_at


def test_finished_sentence_ends_after_the_short_pause():
    seg = UtteranceSegmenter(Config(vad="energy"))
    early_at, final_at = run_segmenter(seg, 0.5, 2.0, verdict=True)
    assert early_at == pytest.approx(0.45 + 0.15)    # last voice at 0.45 s + early pause
    assert final_at == pytest.approx(0.45 + 0.40)    # ...+ the short pause
    assert seg.final_matches_early and not seg.last_waited


def test_unfinished_words_get_the_longer_pause():
    seg = UtteranceSegmenter(Config(vad="energy"))
    _, final_at = run_segmenter(seg, 0.5, 2.0, verdict=False)
    assert final_at == pytest.approx(0.45 + 1.0)
    assert seg.last_waited


def test_no_verdict_yet_waits_a_little_longer():
    seg = UtteranceSegmenter(Config(vad="energy"))
    _, final_at = run_segmenter(seg, 0.5, 2.0, verdict=None)
    assert final_at == pytest.approx(0.45 + 0.40 + 0.20)


def test_stale_verdict_is_ignored():
    seg = UtteranceSegmenter(Config(vad="energy"))
    for i in range(10):                              # 0.5 s of speech
        seg.feed(LOUD, i * 0.05, True)
    for i in range(10, 14):                          # 0.2 s pause: early snapshot
        seg.feed(QUIET, i * 0.05, False)
    stale = seg.early_seq
    assert seg.pop_early() is not None
    seg.feed(LOUD, 0.70, True)                       # they spoke again
    seg.set_early_verdict(stale, False)              # the old snapshot's verdict arrives late
    assert seg._verdict is None


@pytest.mark.parametrize("text, unfinished", [
    ("It's just", True), ("first", True), ("season", True),
    ("I think Japan is nice because", True), ("with right hand left", False),
    ("What should I have for dinner?", False), ("Hi Reachy", False), ("Reachy?", False),
    ("We need a cursor, we need to buy it.", False), ("Okay.", True),
])
def test_looks_unfinished(text, unfinished):
    from audio.speaker_nlp_model import looks_unfinished
    assert looks_unfinished(text) is unfinished


def test_preroll_keeps_the_start_of_the_first_word():
    seg = UtteranceSegmenter(Config(vad="energy"))
    t = 0.0
    for _ in range(10):                      # 0.5 s of quiet before speaking
        seg.feed(QUIET, t, False); t += 0.05
    for _ in range(10):
        seg.feed(LOUD, t, True); t += 0.05
    out = None
    while out is None:
        out = seg.feed(QUIET, t, False); t += 0.05
        if seg.pop_early() is not None:
            seg.set_early_verdict(seg.early_seq, True)
    # 0.25 s pre-roll + 0.5 s speech + 0.4 s pause
    assert len(out) == pytest.approx(int(0.25 * 16000) + 10 * CHUNK + 8 * CHUNK, abs=CHUNK)


def test_early_transcript_is_reused_when_nobody_speaks_again():
    model = mic_model()
    t = speak(model, 0.0, 1.0)
    t = speak(model, t, 0.2, loud=False)                 # early pause: transcription starts
    assert [j for j, _ in model.transcriber.jobs] == [1]
    model.transcriber.finish(1, "Reachy, what about Japan?")
    assert model._next_transcript() is None              # not over yet: nothing delivered
    speak(model, t, 0.3, loud=False)                     # the short pause: sentence over
    item = model._next_transcript()
    assert item[0] == "Reachy, what about Japan?"
    assert len(model.transcriber.jobs) == 1              # no second transcription


def test_unfinished_start_keeps_one_sentence_and_transcribes_it_whole():
    model = mic_model()
    t = speak(model, 0.0, 0.6)                           # "It's just..."
    t = speak(model, t, 0.2, loud=False)                 # early job 1
    model.transcriber.finish(1, "It's just")
    t = speak(model, t, 0.5, loud=False)                 # 0.7 s pause: would have ended before
    assert model._next_transcript() is None              # ...but it looked unfinished: wait
    t = speak(model, t, 0.8)                             # "...that I'm hungry."
    t = speak(model, t, 0.2, loud=False)                 # early job 2: the whole sentence
    model.transcriber.finish(2, "It's just that I'm hungry.")
    speak(model, t, 0.3, loud=False)
    item = model._next_transcript()
    assert item[0] == "It's just that I'm hungry."       # one sentence, not "It's just" + rest
    assert model._next_transcript() is None


def test_early_transcript_dropped_when_they_keep_talking():
    model = mic_model()
    t = speak(model, 0.0, 1.0)
    t = speak(model, t, 0.2, loud=False)                 # early job 1 for "Reachy, what about"
    model.transcriber.finish(1, "Reachy, what about")
    t = speak(model, t, 0.6)                             # ...they carry on: "...Japan?"
    t = speak(model, t, 0.2, loud=False)                 # early job 2 for the whole sentence
    speak(model, t, 0.5, loud=False)                     # no verdict yet: ends after 0.6 s
    assert [j for j, _ in model.transcriber.jobs] == [1, 2]
    assert model._next_transcript() is None              # job 2 still running
    model.transcriber.finish(2, "Reachy, what about Japan?")
    assert model._next_transcript()[0] == "Reachy, what about Japan?"
    assert model._next_transcript() is None              # the stale "Reachy, what about" is gone


def test_final_transcript_that_arrives_late_is_delivered():
    model = mic_model()
    t = speak(model, 0.0, 1.0)
    speak(model, t, 0.7, loud=False)                     # early + final, job not done yet
    assert model._next_transcript() is None
    model.transcriber.finish(1, "Hi Reachy")
    assert model._next_transcript()[0] == "Hi Reachy"


def test_misheard_riki_is_reachy():
    assert classify_utterance("Riki, what do you think?")["said_name"]


# ---------------------------------------------------------------------------
# Voice: start with a short first piece so the first sound comes sooner
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio"))
from reachy_voice import split_first_clause  # noqa: E402


@pytest.mark.parametrize("sentence, pieces", [
    ("Sunscreen, a hat, and sandals are the basics for me.",
     ["Sunscreen, a hat,", "and sandals are the basics for me."]),
    ("Hmm, I'm not sure what this is yet, can you tell me more?",
     ["Hmm, I'm not sure what this is yet,", "can you tell me more?"]),
    ("Oh, Japan sounds wonderful!", ["Oh, Japan sounds wonderful!"]),        # "Oh," too short
    ("Kyoto is lovely in spring.", ["Kyoto is lovely in spring."]),          # no comma
    ("Good idea, honestly.", ["Good idea, honestly."]),                      # rest too short
    ("Ooh, dinner, my favorite topic from up here on the desk, honestly!",   # rest > 3x first:
     ["Ooh, dinner, my favorite topic from up here on the desk, honestly!"]),  # would leave a gap
])
def test_split_first_clause(sentence, pieces):
    assert split_first_clause(sentence) == pieces
    assert " ".join(split_first_clause(sentence)) == sentence


def test_reachy_is_pronounced_ree_chee():
    from reachy_voice import say_it_right
    assert say_it_right("Hi, I'm Reachy! That's Reachy's job.") == "Hi, I'm Reechy! That's Reechy's job."
    assert say_it_right("I can reach you.") == "I can reach you."
