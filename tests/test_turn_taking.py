"""Tests for when Reachy speaks or nods, and for streaming its reply sentence by sentence.

Run from the project root:  python -m pytest tests
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from audio.speaker_nlp_model import (Config, ReplyEngine, ReplyStreamParser, SpeakerNLPModel,
                                     classify_utterance, _FallBackToTemplate)
from decision.fusion import make_decision, summarize_gaze


# ---------------------------------------------------------------------------
# Name at the start of the sentence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, name_first", [
    ("Reachy, what do you think?", True),
    ("Hi Reachy, how are you doing?", True),
    ("Hey, Reachy what's up", True),
    ("richie, should we take the train?", True),          # misheard name is fixed first
    ("Okay so Reachy which one?", True),
    ("I'll tell you why. Hi, Reachy, I'm talking to you.", False),
    ("What do you think, Reachy?", False),
    ("What do you think about Japan?", False),
])
def test_name_first(text, name_first):
    assert classify_utterance(text)["name_first"] is name_first


def test_said_name_still_matches_anywhere():
    c = classify_utterance("What do you think, Reachy?")
    assert c["said_name"] and not c["name_first"]


# ---------------------------------------------------------------------------
# The decision rules
# ---------------------------------------------------------------------------

def utt(text, answering_robot=False):
    return {"utterance": {"id": 1, **classify_utterance(text), "answering_robot": answering_robot}}


LOOKING = {"speaker": "person_left", "looking_at_robot": True, "looking_fraction": 0.9, "target_norm_x": -0.4}
AWAY = {"speaker": "person_left", "looking_at_robot": False, "looking_fraction": 0.1, "target_norm_x": -0.4}
NO_FACE = {"speaker": "unknown", "looking_at_robot": False, "looking_fraction": 0.0, "target_norm_x": None}


@pytest.mark.parametrize("audio, gaze, action, method", [
    (utt("Reachy, I love Japan."), AWAY, "speak", "name_first"),
    (utt("Hi Reachy"), NO_FACE, "speak", "name_first"),
    (utt("What do you think about Japan?"), LOOKING, "speak", "looking_and_question"),
    (utt("I think Japan would be fun."), LOOKING, "nod", "looking_and_statement"),
    (utt("What do you think about Japan?"), AWAY, "none", "looking_away"),
    (utt("What do you think about Japan?"), NO_FACE, "none", "insufficient_evidence"),
    (utt("Yes, the train.", answering_robot=True), AWAY, "speak", "answering_robot"),
])
def test_decision(audio, gaze, action, method):
    d = make_decision(audio, gaze)
    assert d["action"] == action
    assert d["evidence"]["addressee_method"] == method
    assert d["should_speak"] is (action == "speak")
    assert d["should_nod"] is (action == "nod")


# ---------------------------------------------------------------------------
# Gaze over the sentence, not at the moment the transcript arrives
# ---------------------------------------------------------------------------

def frame(speaker="none", left=None, right=None):
    cv = {}
    if left is not None:
        cv["person_left"] = {"looking_at_robot": left, "norm_x": -0.5, "is_speaking": speaker == "person_left"}
    if right is not None:
        cv["person_right"] = {"looking_at_robot": right, "norm_x": 0.5, "is_speaking": speaker == "person_right"}
    return {"active_speaker": speaker, "cv": cv}


def test_gaze_uses_the_frames_while_speaking():
    frames = [(10.0 + i * 0.1, frame("person_left", left=True, right=False)) for i in range(20)]
    # After the sentence ended, they turned to their partner: that must not count.
    frames += [(13.0 + i * 0.1, frame("none", left=False, right=False)) for i in range(20)]
    g = summarize_gaze(frames, 10.0, 12.0)
    assert g["speaker"] == "person_left" and g["speaker_method"] == "lips_doa"
    assert g["looking_at_robot"] and g["looking_fraction"] == 1.0
    assert g["target_norm_x"] == -0.5


def test_gaze_needs_half_the_sentence():
    frames = [(10.0 + i * 0.1, frame("person_right", left=False, right=i < 8)) for i in range(20)]
    g = summarize_gaze(frames, 10.0, 12.0)
    assert g["speaker"] == "person_right"
    assert g["looking_fraction"] == 0.4 and not g["looking_at_robot"]


def test_single_face_is_the_speaker_when_lips_missed():
    frames = [(10.0 + i * 0.1, frame("none", left=True)) for i in range(10)]
    g = summarize_gaze(frames, 10.0, 11.0)
    assert g["speaker"] == "person_left" and g["speaker_method"] == "only_face"
    assert g["looking_at_robot"]


def test_two_faces_and_no_lips_is_unknown():
    frames = [(10.0 + i * 0.1, frame("none", left=True, right=True)) for i in range(10)]
    assert summarize_gaze(frames, 10.0, 11.0)["speaker"] == "unknown"


def test_no_frames():
    g = summarize_gaze([], 10.0, 11.0)
    assert g["speaker"] == "unknown" and g["frames"] == 0


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
    assert u["name_first"] and before - 1 <= u["t"] <= u["end_t"] <= time.time() + 1


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
