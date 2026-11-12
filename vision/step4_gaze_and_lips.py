from collections import deque
import cv2
import mediapipe as mp
import numpy as np


class GazeAndLipTracker:

    def __init__(self, camera_index: int = 0, history_size: int = 10):
        # =====================================================================
        # TUNABLE PARAMETERS (PLAY WITH THESE VALUES!)
        # =====================================================================
        
        # 1. GAZE / HEAD YAW THRESHOLDS (in degrees, range roughly -90 to +90)
        # Positive (+) yaw = turning head toward the RIGHT side of the camera image
        # Negative (-) yaw = turning head toward the LEFT side of the camera image
        
        # For PERSON_LEFT (sitting on the left, partner is to their right -> positive yaw):
        # If yaw is between these min/max values, they are counted as LOOKING AT ROBOT.
        # Once yaw goes above LEFT_PERSON_MAX_ROBOT_YAW (e.g., > 42 deg), it switches to AT PARTNER.
        self.LEFT_PERSON_MIN_ROBOT_YAW = -35.0
        self.LEFT_PERSON_MAX_ROBOT_YAW = 42.0

        # For PERSON_RIGHT (sitting on the right, partner is to their left -> negative yaw):
        # If yaw is between these min/max values, they are counted as LOOKING AT ROBOT.
        # Once yaw drops below RIGHT_PERSON_MIN_ROBOT_YAW (e.g., < -42 deg), it switches to AT PARTNER.
        self.RIGHT_PERSON_MIN_ROBOT_YAW = -42.0
        self.RIGHT_PERSON_MAX_ROBOT_YAW = 35.0

        # 2. SECOND-LAYER CONFIRMATION WEIGHTS
        # Blends Layer 1 (2D Cheek/Jaw width ratio) and Layer 2 (3D Cheek/Ear depth 'z')
        self.WEIGHT_2D_JAW_RATIO = 0.65
        self.WEIGHT_3D_DEPTH = 0.35

        # 3. SMOOTHING FILTER (0.1 = very slow/smooth, 1.0 = instant/jittery)
        self.YAW_SMOOTHING_ALPHA = 0.35

        # 4. LIP MOVEMENT (SPEAKING) THRESHOLDS
        self.MAR_STD_THRESHOLD = 0.022   # Rolling variance required for speech
        self.MAR_OPEN_THRESHOLD = 0.12   # Minimum mouth opening required for speech
        # =====================================================================

        self.cap = cv2.VideoCapture(camera_index)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmark indices
        self.NOSE_TIP = 1
        self.UPPER_INNER_LIP = 13
        self.LOWER_INNER_LIP = 14
        self.LEFT_MOUTH_CORNER = 78
        self.RIGHT_MOUTH_CORNER = 308

        # Outer cheek / ear-tragus jawline landmarks for Layer 1 & Layer 2
        self.LEFT_CHEEK_EDGE = 234
        self.RIGHT_CHEEK_EDGE = 454
        self.FOREHEAD = 10
        self.CHIN = 152

        # State buffers for lip MAR and smoothed yaw
        self.mar_history = {
            "person_left": deque(maxlen=history_size),
            "person_right": deque(maxlen=history_size),
        }
        self.smoothed_yaw = {
            "person_left": 0.0,
            "person_right": 0.0,
        }
        self.smoothed_pitch = {
            "person_left": 0.0,
            "person_right": 0.0,
        }

    @staticmethod
    def _euclidean(p1, p2):
        return float(np.linalg.norm(p1 - p2))

    def _compute_mar(self, lms, w, h):
        top = np.array([lms[self.UPPER_INNER_LIP].x * w, lms[self.UPPER_INNER_LIP].y * h])
        bottom = np.array([lms[self.LOWER_INNER_LIP].x * w, lms[self.LOWER_INNER_LIP].y * h])
        left = np.array([lms[self.LEFT_MOUTH_CORNER].x * w, lms[self.LEFT_MOUTH_CORNER].y * h])
        right = np.array([lms[self.RIGHT_MOUTH_CORNER].x * w, lms[self.RIGHT_MOUTH_CORNER].y * h])

        width = self._euclidean(left, right)
        return self._euclidean(top, bottom) / width if width > 1e-5 else 0.0

    def _estimate_two_layer_gaze(self, slot_name: str, lms, w: int, h: int):
        """
        Combines Layer 1 (2D Cheek/Jawline asymmetry) and Layer 2 (3D Landmark Z-depth)
        to compute a glitch-free Yaw angle (-90 to +90 deg) and visual pointer vector.
        """
        nose_x = lms[self.NOSE_TIP].x * w
        nose_y = lms[self.NOSE_TIP].y * h
        left_cheek_x = lms[self.LEFT_CHEEK_EDGE].x * w
        right_cheek_x = lms[self.RIGHT_CHEEK_EDGE].x * w

        # --- LAYER 1: 2D Jaw/Cheek Width Ratio ---
        # Distance from nose tip to left and right outer cheek edges
        dist_left = max(1.0, nose_x - left_cheek_x)
        dist_right = max(1.0, right_cheek_x - nose_x)
        total_width = dist_left + dist_right

        # Asymmetry ratio in [-1.0, +1.0] -> scaled to [-90, +90] degrees
        jaw_ratio = (dist_left - dist_right) / total_width
        yaw_2d_deg = jaw_ratio * 90.0

        # --- LAYER 2: 3D Depth (Z) Confirmation ---
        # In MediaPipe, smaller (more negative) z is closer to the camera.
        # When turning right (+yaw), left cheek (234) moves closer than right cheek (454).
        z_left = lms[self.LEFT_CHEEK_EDGE].z
        z_right = lms[self.RIGHT_CHEEK_EDGE].z
        yaw_3d_deg = float(np.clip((z_right - z_left) * 220.0, -90.0, 90.0))

        # Fuse Layer 1 and Layer 2
        raw_yaw_deg = (
            self.WEIGHT_2D_JAW_RATIO * yaw_2d_deg
            + self.WEIGHT_3D_DEPTH * yaw_3d_deg
        )

        # Estimate simple vertical pitch for pointer visualization
        forehead_y = lms[self.FOREHEAD].y * h
        chin_y = lms[self.CHIN].y * h
        upper_h = max(1.0, nose_y - forehead_y)
        lower_h = max(1.0, chin_y - nose_y)
        raw_pitch_deg = ((upper_h - lower_h) / (upper_h + lower_h)) * 60.0

        # Apply Exponential Moving Average (EMA) smoothing
        alpha = self.YAW_SMOOTHING_ALPHA
        self.smoothed_yaw[slot_name] = (
            alpha * raw_yaw_deg + (1.0 - alpha) * self.smoothed_yaw[slot_name]
        )
        self.smoothed_pitch[slot_name] = (
            alpha * raw_pitch_deg + (1.0 - alpha) * self.smoothed_pitch[slot_name]
        )

        yaw_deg = self.smoothed_yaw[slot_name]
        pitch_deg = self.smoothed_pitch[slot_name]

        # Compute clean 2D pointer endpoint from smoothed yaw/pitch
        line_length = 110.0
        end_x = int(nose_x + line_length * np.sin(np.deg2rad(yaw_deg)))
        end_y = int(nose_y + line_length * np.sin(np.deg2rad(pitch_deg)))

        return yaw_deg, (int(nose_x), int(nose_y)), (end_x, end_y)

    def _is_looking_at_robot(self, slot_name: str, yaw_deg: float) -> bool:
        """Check side-specific yaw thresholds."""
        if slot_name == "person_left":
            return self.LEFT_PERSON_MIN_ROBOT_YAW <= yaw_deg <= self.LEFT_PERSON_MAX_ROBOT_YAW
        else:
            return self.RIGHT_PERSON_MIN_ROBOT_YAW <= yaw_deg <= self.RIGHT_PERSON_MAX_ROBOT_YAW

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, {}

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        faces_found = []
        if results.multi_face_landmarks:
            for lms_obj in results.multi_face_landmarks:
                lms = lms_obj.landmark
                px_x = int(lms[self.NOSE_TIP].x * w)
                px_y = int(lms[self.NOSE_TIP].y * h)
                norm_x = round((px_x - (w / 2.0)) / (w / 2.0), 3)
                faces_found.append({"pixel": (px_x, px_y), "norm_x": norm_x, "landmarks": lms})

        faces_found.sort(key=lambda f: f["pixel"][0])

        assignments = []
        if len(faces_found) == 1:
            slot = "person_left" if faces_found[0]["pixel"][0] < (w // 2) else "person_right"
            assignments.append((slot, faces_found[0]))
        elif len(faces_found) >= 2:
            assignments.append(("person_left", faces_found[0]))
            assignments.append(("person_right", faces_found[1]))

        annotated = frame.copy()
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
        state_output = {}

        for slot_name, face_data in assignments:
            lms = face_data["landmarks"]
            pt = face_data["pixel"]

            # 1. Lip Movement (MAR)
            mar = self._compute_mar(lms, w, h)
            self.mar_history[slot_name].append(mar)
            mar_arr = np.array(self.mar_history[slot_name])
            mar_std = float(np.std(mar_arr)) if len(mar_arr) >= 4 else 0.0
            mar_mean = float(np.mean(mar_arr))
            is_speaking = (mar_std > self.MAR_STD_THRESHOLD) and (mar_mean > self.MAR_OPEN_THRESHOLD)

            # 2. Two-Layer Gaze Estimation (Cheek Ratio + 3D Depth)
            yaw_deg, nose_pt, nose_end = self._estimate_two_layer_gaze(slot_name, lms, w, h)
            looking_at_robot = self._is_looking_at_robot(slot_name, yaw_deg)

            state_output[slot_name] = {
                "norm_x": face_data["norm_x"],
                "is_speaking": bool(is_speaking),
                "looking_at_robot": bool(looking_at_robot),
                "yaw_deg": round(yaw_deg, 1),
            }

            # Draw Cheek Landmarks (small white dots) so you can see Layer 1 working
            lc_pt = (int(lms[self.LEFT_CHEEK_EDGE].x * w), int(lms[self.LEFT_CHEEK_EDGE].y * h))
            rc_pt = (int(lms[self.RIGHT_CHEEK_EDGE].x * w), int(lms[self.RIGHT_CHEEK_EDGE].y * h))
            cv2.circle(annotated, lc_pt, 4, (255, 255, 255), -1)
            cv2.circle(annotated, rc_pt, 4, (255, 255, 255), -1)

            # Draw Nose Direction Vector & Status Text
            # Green = AT ROBOT, Orange = AT PARTNER
            color = (0, 255, 0) if looking_at_robot else (0, 165, 255)
            cv2.arrowedLine(annotated, nose_pt, nose_end, (255, 255, 0), 3, tipLength=0.25)
            cv2.circle(annotated, nose_pt, 6, color, -1)

            gaze_str = "AT ROBOT" if looking_at_robot else "AT PARTNER"
            speak_str = "SPEAKING" if is_speaking else "SILENT"
            label = f"{slot_name}: {speak_str} | {gaze_str} (yaw={yaw_deg:.0f}d)"
            cv2.putText(annotated, label, (max(10, pt[0] - 140), max(30, pt[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        return annotated, state_output

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = GazeAndLipTracker(camera_index=0)
    print("Running Updated Step 4: Two-Layer Gaze & Lip Tracker. Press 'q' to quit.")

    try:
        while True:
            annotated_frame, state = tracker.process_frame()
            if annotated_frame is None:
                continue

            cv2.imshow("Step 4: Gaze & Lip Tracker (Two-Layer Fix)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.release()