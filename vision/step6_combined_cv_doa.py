import threading
import time
import cv2
import numpy as np
import requests

# Import your working CV tracker directly from step4_gaze_and_lips_2.py
from vision.step4_gaze_and_lips import GazeAndLipTracker


class DoAPoller:
    """
    Runs test_doa_only.py in a background thread so HTTP requests
    never slow down the 30 FPS OpenCV + MediaPipe loop.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.64",
        left_Right_threshold_deg: float = 15.0,
        speech_hold_sec: float = 0.4,
        invert_left_right: bool = False,
    ):
        self.doa_url = f"http://{robot_ip}:8000/api/state/doa"
        self.threshold_deg = left_Right_threshold_deg
        self.speech_hold_sec = speech_hold_sec
        self.invert_left_right = invert_left_right

        self._lock = threading.Lock()
        self._running = True

        # Shared state read by the main loop
        self.raw_deg = 90.0
        self.centered_deg = 0.0
        self.is_speech = False
        self.audio_side = "SILENT"  # "LEFT", "RIGHT", "CENTER", or "SILENT"
        self._last_speech_time = 0.0
        self._last_active_side = "CENTER"

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        session = requests.Session()
        while self._running:
            try:
                resp = session.get(self.doa_url, timeout=0.6)
                if resp.status_code == 200:
                    data = resp.json()
                    angle_rad = data.get("angle", np.pi / 2)
                    raw_speech = bool(data.get("speech_detected", False))

                    angle_deg = float(np.rad2deg(angle_rad))
                    centered = 90.0 - angle_deg
                    if self.invert_left_right:
                        centered = -centered

                    now = time.time()
                    if raw_speech:
                        self._last_speech_time = now
                        if centered > self.threshold_deg:
                            self._last_active_side = "LEFT"
                        elif centered < -self.threshold_deg:
                            self._last_active_side = "RIGHT"
                        else:
                            self._last_active_side = "CENTER"

                    # Hold speech active briefly so short pauses between words don't flicker
                    effective_speech = raw_speech or ((now - self._last_speech_time) < self.speech_hold_sec)
                    effective_side = self._last_active_side if effective_speech else "SILENT"

                    with self._lock:
                        self.raw_deg = round(angle_deg, 1)
                        self.centered_deg = round(centered, 1)
                        self.is_speech = effective_speech
                        self.audio_side = effective_side
            except requests.exceptions.RequestException:
                pass

            time.sleep(0.1)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "is_speech": self.is_speech,
                "raw_deg": self.raw_deg,
                "centered_deg": self.centered_deg,
                "audio_side": self.audio_side,
            }

    def stop(self):
        self._running = False


def fuse_cv_and_doa(cv_state: dict, doa_state: dict) -> dict:
    """
    Stage 1: Confirm active speaker using DoA + Visual Lip Movement.
    Stage 2: Determine if the active speaker is talking to the ROBOT or PARTNER.
    """
    audio_side = doa_state["audio_side"]  # "LEFT", "RIGHT", "CENTER", or "SILENT"
    audio_speech = doa_state["is_speech"]

    left_person = cv_state.get("person_left")
    right_person = cv_state.get("person_right")

    left_lips = bool(left_person and left_person["is_speaking"])
    right_lips = bool(right_person and right_person["is_speaking"])

    active_speaker = "none"
    confidence = "NONE"

    # --- STAGE 1: MULTIMODAL SPEAKER CONFIRMATION ---
    # Case 1: Strong Audio + Visual Lip Agreement (Highest Confidence)
    if audio_speech and audio_side in ("LEFT", "CENTER") and left_lips and not right_lips:
        active_speaker = "person_left"
        confidence = "CONFIRMED (DOA + LIPS)"
    elif audio_speech and audio_side in ("RIGHT", "CENTER") and right_lips and not left_lips:
        active_speaker = "person_right"
        confidence = "CONFIRMED (DOA + LIPS)"
    elif audio_speech and left_lips and right_lips:
        # Both moving lips -> break tie using DoA microphone direction
        active_speaker = "person_left" if doa_state["centered_deg"] >= 0.0 else "person_right"
        confidence = "CONFIRMED (DOA TIE-BREAK)"

    # Case 2: Visual Lips Moving, Quiet Mic (Visual Fallback)
    elif left_lips and not right_lips:
        active_speaker = "person_left"
        confidence = "VISION ONLY (QUIET MIC)"
    elif right_lips and not left_lips:
        active_speaker = "person_right"
        confidence = "VISION ONLY (QUIET MIC)"

    # Case 3: Audio Speech Detected, Speaker Out of Frame or Lips Occluded
    elif audio_speech and audio_side == "LEFT" and left_person is None:
        active_speaker = "person_left"
        confidence = "AUDIO CUE (OUT OF FRAME)"
    elif audio_speech and audio_side == "RIGHT" and right_person is None:
        active_speaker = "person_right"
        confidence = "AUDIO CUE (OUT OF FRAME)"

    # --- STAGE 2: ADDRESSEE DETECTION (ROBOT vs PARTNER) ---
    interaction_mode = "IDLE / LISTENING"
    looking_at_robot = False
    target_norm_x = None

    if active_speaker in ("person_left", "person_right"):
        person_info = cv_state.get(active_speaker)
        if person_info is not None:
            looking_at_robot = person_info["looking_at_robot"]
            target_norm_x = person_info["norm_x"]
            if looking_at_robot:
                interaction_mode = f"{active_speaker.upper()} -> TALKING TO ROBOT"
            else:
                interaction_mode = f"{active_speaker.upper()} -> TALKING TO PARTNER"
        else:
            interaction_mode = f"{active_speaker.upper()} -> AUDIO CUE (SEARCHING)"

    return {
        "active_speaker": active_speaker,
        "confidence": confidence,
        "looking_at_robot": looking_at_robot,
        "interaction_mode": interaction_mode,
        "target_norm_x": target_norm_x,
        "doa": doa_state,
        "cv": cv_state,
    }


def draw_fusion_hud(frame: np.ndarray, fused: dict):
    """Draws a clean status dashboard at the top of the OpenCV window."""
    h, w, _ = frame.shape
    doa = fused["doa"]

    # Top dark banner
    cv2.rectangle(frame, (0, 0), (w, 85), (25, 25, 25), -1)

    # Line 1: Live DoA status
    doa_color = (0, 255, 255) if doa["is_speech"] else (160, 160, 160)
    doa_text = (
        f"DOA MIC: {'SPEECH' if doa['is_speech'] else 'SILENT'} | "
        f"Side: {doa['audio_side']} (Centered: {doa['centered_deg']:+.1f}d)"
    )
    cv2.putText(frame, doa_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, doa_color, 2)

    # Line 2: Fused Speaker & Interaction Mode
    if "CONFIRMED" in fused["confidence"]:
        fus_color = (0, 255, 0)
    elif fused["active_speaker"] != "none":
        fus_color = (0, 200, 255)
    else:
        fus_color = (180, 180, 180)

    fus_text = f"FUSION: {fused['interaction_mode']} [{fused['confidence']}]"
    cv2.putText(frame, fus_text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58, fus_color, 2)


if __name__ == "__main__":
    # =====================================================================
    # TUNABLE FUSION SETTINGS
    # =====================================================================
    ROBOT_IP = "192.168.1.64"
    CAMERA_INDEX = 0
    DOA_SIDE_THRESHOLD_DEG = 15.0  # Degrees from center to count as LEFT or RIGHT

    # Set to True if standing on the left of your webcam image shows "RIGHT" on DoA
    INVERT_DOA_LEFT_RIGHT = False
    # =====================================================================

    cv_tracker = GazeAndLipTracker(camera_index=CAMERA_INDEX)
    doa_poller = DoAPoller(
        robot_ip=ROBOT_IP,
        left_Right_threshold_deg=DOA_SIDE_THRESHOLD_DEG,
        invert_left_right=INVERT_DOA_LEFT_RIGHT,
    )

    print("Running Step 6: Combined CV + DoA Fusion Tracker. Press 'q' to quit.")

    try:
        while True:
            annotated_frame, cv_state = cv_tracker.process_frame()
            if annotated_frame is None:
                continue

            doa_state = doa_poller.get_state()
            fused_state = fuse_cv_and_doa(cv_state, doa_state)

            draw_fusion_hud(annotated_frame, fused_state)
            cv2.imshow("Step 6: Combined CV + DoA Fusion", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        doa_poller.stop()
        cv_tracker.release()