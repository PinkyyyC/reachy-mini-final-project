
# import math
from collections import Counter



# ==========================================
# Configuration
# ==========================================

SPEAKING_SCORE_THRESHOLD = 0.5
SPEAKING_SCORE_MARGIN = 0.2


# ==========================================
# 1. Speaker Identification
# ==========================================

def identify_speaker(vision_output: dict) -> dict:
    """
    Identify who is currently speaking.

    Priority:
    1. Use visual_active_speaker if available.
    2. Otherwise, compare speaking_score from both people.

    Returns:
        {
            "speaker": "person_left" / "person_right" / "unknown",
            "confidence": float
        }
    """

    # Check the explicit active speaker output first
    active_speaker = vision_output.get("visual_active_speaker")

    if active_speaker in ("person_left", "person_right"):
        person_data = vision_output.get(active_speaker, {})

        # Do not trust a speaker prediction if that person is not detected
        if person_data.get("detected", False):
            confidence = person_data.get("speaking_score", 0.5)

            return {
                "speaker": active_speaker,
                "confidence": round(float(confidence), 3),
                "method": "visual_active_speaker"
            }

    # Fallback: compare speaking scores
    left = vision_output.get("person_left", {})
    right = vision_output.get("person_right", {})

    left_score = (
        left.get("speaking_score", 0.0)
        if left.get("detected", False) else 0.0
    )

    right_score = (
        right.get("speaking_score", 0.0)
        if right.get("detected", False) else 0.0
    )

    # Neither person has a sufficiently high speaking score
    if max(left_score, right_score) < SPEAKING_SCORE_THRESHOLD:
        return {
            "speaker": "unknown",
            "confidence": 0.0,
            "method": "speaking_score"
        }

    # Scores are too close to distinguish the speaker
    if abs(left_score - right_score) < SPEAKING_SCORE_MARGIN:
        return {
            "speaker": "unknown",
            "confidence": 0.0,
            "method": "speaking_score"
        }

    if left_score > right_score:
        return {
            "speaker": "person_left",
            "confidence": round(float(left_score), 3),
            "method": "speaking_score"
        }

    return {
        "speaker": "person_right",
        "confidence": round(float(right_score), 3),
        "method": "speaking_score"
    }


# ==========================================
# 2. Addressee Detection
# ==========================================

def detect_addressee(
    audio_output: dict,
    vision_output: dict,
    speaker: str
) -> dict:
    """
    Determine whether the speaker is addressing the robot or another person.

    Evidence priority:
    1. answering_robot
    2. said_name
    3. Looking_at_robot

    Note:
    A question intent alone does NOT mean the question is directed
    at the robot.
    """

    utterance = audio_output.get("utterance", {})

    said_name = utterance.get("said_name", False)
    answering_robot = utterance.get("answering_robot", False)

    # Strong evidence: the speaker is answering the robot
    if answering_robot:
        return {
            "addressee": "robot",
            "confidence": 1.0,
            "method": "answering_robot"
        }

    # Strong evidence: the speaker explicitly said the robot's name
    if said_name:
        return {
            "addressee": "robot",
            "confidence": 0.95,
            "method": "said_name"
        }

    # Use the speaker's gaze direction as visual evidence
    if speaker in ("person_left", "person_right"):
        person_data = vision_output.get(speaker, {})

        if person_data.get("detected", False):
            looking_at_robot = person_data.get("Looking_at_robot")

            if looking_at_robot is True:
                return {
                    "addressee": "robot",
                    "confidence": 0.8,
                    "method": "gaze"
                }

            if looking_at_robot is False:
                return {
                    "addressee": "human",
                    "confidence": 0.7,
                    "method": "gaze"
                }

    # Insufficient evidence
    return {
        "addressee": "unknown",
        "confidence": 0.0,
        "method": "insufficient_evidence"
    }


# ==========================================
# 3. Dialogue Decision
# ==========================================

# def make_decision(
#     audio_output: dict,
#     vision_output: dict
# ) -> dict:
#     """
#     Combine speaker identification and addressee detection.

#     Returns a decision dictionary for the NLP and robot-control modules.
#     """

#     utterance = audio_output.get("utterance", {})

#     # Step 1: Identify the speaker
#     speaker_result = identify_speaker(vision_output)
#     speaker = speaker_result["speaker"]

#     # Step 2: Detect the addressee
#     addressee_result = detect_addressee(
#         audio_output,
#         vision_output,
#         speaker
#     )

#     addressee = addressee_result["addressee"]

#     # Step 3: Decide whether the robot should respond
#     should_speak = addressee == "robot"

#     # The robot should look toward the person who is speaking
#     attention_target = speaker if speaker != "unknown" else None

#     return {
#         "utterance_id": utterance.get("id"),
#         "speaker": speaker,
#         "speaker_confidence": speaker_result["confidence"],
#         "addressee": addressee,
#         "addressee_confidence": addressee_result["confidence"],
#         "should_speak": should_speak,
#         "attention_target": attention_target,
#         "intent": utterance.get("intent"),
#         "text": utterance.get("text"),
#         "evidence": {
#             "speaker_method": speaker_result["method"],
#             "addressee_method": addressee_result["method"],
#             "said_name": utterance.get("said_name", False),
#             "answering_robot": utterance.get("answering_robot", False)
#         }
#     }



PEOPLE = ("person_left", "person_right")
LOOKING_FRACTION = 0.5     # looking at Reachy for at least half of the sentence counts
GAZE_PAD_S = 0.3           # frames this close to the sentence's start/end also count


def summarize_gaze(frames, start_t: float, end_t: float) -> dict:
    """Who spoke, and were they looking at Reachy, over the time the sentence was said.

    frames: [(time.time(), fused_cv_state), ...] recorded by the vision thread.
    The speaker is whoever the lips + DoA fusion picked most often. If it never picked
    anyone but only one face was in view, that face is the speaker.
    """
    window = [f for t, f in frames if start_t - GAZE_PAD_S <= t <= end_t + GAZE_PAD_S]
    summary = {"frames": len(window), "speaker": "unknown", "speaker_method": "none",
               "looking_at_robot": False, "looking_fraction": 0.0, "target_norm_x": None}
    if not window:
        return summary

    votes = Counter(f.get("active_speaker") for f in window if f.get("active_speaker") in PEOPLE)
    if votes:
        speaker, method = votes.most_common(1)[0][0], "lips_doa"
    else:
        faces_seen = {slot for f in window for slot in (f.get("cv") or {}) if slot in PEOPLE}
        if len(faces_seen) != 1:
            return summary
        speaker, method = faces_seen.pop(), "only_face"
    summary.update(speaker=speaker, speaker_method=method)

    faces = [f["cv"][speaker] for f in window if speaker in (f.get("cv") or {})]
    if faces:
        fraction = sum(bool(face.get("looking_at_robot")) for face in faces) / len(faces)
        summary.update(looking_fraction=round(fraction, 2),
                       looking_at_robot=fraction >= LOOKING_FRACTION,
                       target_norm_x=faces[-1].get("norm_x"))
    return summary


def make_decision(audio_output: dict, gaze: dict) -> dict:
    """Decide what Reachy does with a finished sentence: "speak", "nod" or "none".

    Reachy speaks when:
      1. the person is answering a question Reachy just asked, or
      2. they said "Reachy" at the start of the sentence ("Reachy, ...", "Hi Reachy, ..."), or
      3. they looked at Reachy while asking a question.
    Looking at Reachy while making a statement gets a nod. Otherwise Reachy stays quiet.

    Args:
        audio_output: {"utterance": dict} from SpeakerNLPModel
        gaze: summarize_gaze() over the time the sentence was said
    """
    utterance = audio_output.get("utterance", {})
    said_name = utterance.get("said_name", False)
    name_first = utterance.get("name_first", False)
    answering_robot = utterance.get("answering_robot", False)
    is_question = utterance.get("intent") == "question"

    speaker = gaze.get("speaker", "unknown")
    looking_at_robot = bool(gaze.get("looking_at_robot", False))

    if answering_robot:
        action, addressee, method = "speak", "robot", "answering_robot"
    elif name_first:
        action, addressee, method = "speak", "robot", "name_first"
    elif speaker in PEOPLE and looking_at_robot and is_question:
        action, addressee, method = "speak", "robot", "looking_and_question"
    elif speaker in PEOPLE and looking_at_robot:
        action, addressee, method = "nod", "robot", "looking_and_statement"
    elif speaker in PEOPLE:
        action, addressee, method = "none", "human", "looking_away"
    else:
        action, addressee, method = "none", "unknown", "insufficient_evidence"

    return {
        "utterance_id": utterance.get("id"),
        "speaker": speaker,
        "speaker_method": gaze.get("speaker_method", "none"),
        "addressee": addressee,
        "action": action,
        "should_speak": action == "speak",
        "should_nod": action == "nod",
        "target_norm_x": gaze.get("target_norm_x"),
        "intent": utterance.get("intent"),
        "text": utterance.get("text"),
        "evidence": {
            "addressee_method": method,
            "said_name": said_name,
            "name_first": name_first,
            "answering_robot": answering_robot,
            "looking_at_robot": looking_at_robot,
            "looking_fraction": gaze.get("looking_fraction", 0.0),
        },
    }
