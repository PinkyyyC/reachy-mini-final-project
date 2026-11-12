# EXPRESSION_TO_MOTION = {
#     "welcoming": "welcoming",
#     "thoughtful": "thoughtful",
#     "surprised": "surprised",
#     "helpful": "helpful",
#     "anxiety": "anxiety",
#     "uncertain": "uncertain",
#     "enthusiastic": "enthusiastic",
#     "neutral": "understanding",
# }


# class ActionPlanner:

#     def __init__(self):
#         self.state = "IDLE"

#     def process_decision(self, decision):

#         if not decision.get("should_speak", False):
#             self.state = "LISTENING"

#             return {
#                 "state": self.state,
#                 "should_speak": False,
#                 "expression": None,
#                 "action": "continue_listening"
#             }

#         self.state = "THINKING"

#         return {
#             "state": self.state,
#             "should_speak": True,
#             "expression": "thoughtful",
#             "action": "generate_response",
#             "text": decision.get("text")
#         }

#     def process_response(self, nlp_output):

#         self.state = "SPEAKING"

#         expression = nlp_output.get(
#             "response_expression", "neutral"
#         )

#         if expression not in EXPRESSION_TO_MOTION:
#             expression = "neutral"

#         return {
#             "state": self.state,
#             "should_speak": True,
#             "expression": EXPRESSION_TO_MOTION[expression],
#             "action": "speak",
#             "response_text": nlp_output.get("response_text"),
#             "response_expression": expression
#         }

#     def finish_speaking(self):

#         self.state = "LISTENING"

#         return {
#             "state": self.state,
#             "action": "continue_listening",
#             "expression": None
#         }

#     def reset(self):
#         self.state = "IDLE"


import time

EXPRESSION_TO_MOTION = {
    "welcoming": "welcoming",
    "thoughtful": "thoughtful",
    "surprised": "surprised",
    "helpful": "helpful",
    "anxious": "anxiety",      
    "uncertain": "uncertain",
    "enthusiastic": "enthusiastic",
    "neutral": "understanding",
}


class ActionPlanner:
    def __init__(self, nod_cooldown_sec: float = 3.5):
        self.state = "IDLE"
        self.nod_cooldown_sec = nod_cooldown_sec
        self._last_nod_time = 0.0
        self._gaze_start_time = None
        self._last_gaze_speaker = None

    def check_active_listening(self, fused_cv_state: dict) -> dict:
        """
        Called continuously (~10-20Hz) while the robot is NOT speaking.
        Decides if the robot should turn its head toward the speaker and/or nod.
        """
        now = time.time()
        active_speaker = fused_cv_state.get("active_speaker", "none")
        looking_at_robot = fused_cv_state.get("looking_at_robot", False)
        target_norm_x = fused_cv_state.get("target_norm_x")

        should_nod = False

        if active_speaker in ("person_left", "person_right") and looking_at_robot:
            # Track how long this person has been speaking while looking at the robot
            if self._last_gaze_speaker != active_speaker:
                self._last_gaze_speaker = active_speaker
                self._gaze_start_time = now
            elif (now - self._gaze_start_time >= 1.2) and (now - self._last_nod_time >= self.nod_cooldown_sec):
                # Trigger an understanding nod after 1.2s of sustained eye contact while speaking
                should_nod = True
                self._last_nod_time = now
        else:
            self._last_gaze_speaker = None
            self._gaze_start_time = None

        return {
            "target_norm_x": target_norm_x if active_speaker != "none" else None,
            "should_nod": should_nod,
        }

    def process_decision(self, decision: dict) -> dict:
        if not decision.get("should_speak", False):
            self.state = "LISTENING"
            return {
                "state": self.state,
                "should_speak": False,
                "expression": None,
                "action": "continue_listening"
            }

        self.state = "THINKING"
        return {
            "state": self.state,
            "should_speak": True,
            "expression": "thoughtful",
            "action": "generate_response",
            "text": decision.get("text")
        }

    def process_response(self, nlp_output: dict) -> dict:
        self.state = "SPEAKING"
        raw_emotion = nlp_output.get("emotion", "neutral")

        if raw_emotion not in EXPRESSION_TO_MOTION:
            raw_emotion = "neutral"

        return {
            "state": self.state,
            "should_speak": True,
            "expression": EXPRESSION_TO_MOTION[raw_emotion],
            "action": "speak",
            "response_text": nlp_output.get("text"),
            "response_expression": raw_emotion
        }

    def finish_speaking(self) -> dict:
        self.state = "LISTENING"
        return {
            "state": self.state,
            "action": "continue_listening",
            "expression": None
        }

    def reset(self):
        self.state = "IDLE"