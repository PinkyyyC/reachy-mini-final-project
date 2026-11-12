
# import math



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




def make_decision(audio_output: dict, fused_cv_state: dict) -> dict:
    """
    Combine NLP utterance metadata with CV+DoA fused state.
    
    Args:
        audio_output: {"utterance": dict} from SpeakerNLPModel
        fused_cv_state: output dictionary from fuse_cv_and_doa(cv_state, doa_state)
    """
    utterance = audio_output.get("utterance", {})
    said_name = utterance.get("said_name", False)
    answering_robot = utterance.get("answering_robot", False)

    # 1. Identify speaker from CV + DoA fusion
    active_speaker = fused_cv_state.get("active_speaker", "none")
    speaker = active_speaker if active_speaker in ("person_left", "person_right") else "unknown"
    speaker_confidence = fused_cv_state.get("confidence", "NONE")

    # 2. Detect addressee (Robot vs Partner)
    # Priority: answering_robot > said_name > visual gaze (looking_at_robot)
    looking_at_robot = fused_cv_state.get("looking_at_robot", False)

    if answering_robot:
        addressee = "robot"
        addressee_confidence = 1.0
        addressee_method = "answering_robot"
    elif said_name:
        addressee = "robot"
        addressee_confidence = 0.95
        addressee_method = "said_name"
    elif speaker != "unknown":
        if looking_at_robot:
            addressee = "robot"
            addressee_confidence = 0.85
            addressee_method = "gaze_and_lips"
        else:
            addressee = "human"
            addressee_confidence = 0.80
            addressee_method = "gaze_and_lips"
    else:
        addressee = "unknown"
        addressee_confidence = 0.0
        addressee_method = "insufficient_evidence"

    should_speak = (addressee == "robot")

    return {
        "utterance_id": utterance.get("id"),
        "speaker": speaker,
        "speaker_confidence": speaker_confidence,
        "addressee": addressee,
        "addressee_confidence": addressee_confidence,
        "should_speak": should_speak,
        "target_norm_x": fused_cv_state.get("target_norm_x"),
        "intent": utterance.get("intent"),
        "text": utterance.get("text"),
        "evidence": {
            "addressee_method": addressee_method,
            "said_name": said_name,
            "answering_robot": answering_robot,
            "looking_at_robot": looking_at_robot,
        }
    }