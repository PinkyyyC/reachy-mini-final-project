import time
import threading
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves
from reachy_mini.motion.move import Move
from scipy.spatial.transform import Rotation as R

DATASET = "pollen-robotics/reachy-mini-emotions-library"

EXPRESSION_MAP = {
    "neutral": "understanding1",
    "welcoming": "welcoming1",
    "thoughtful": "thoughtful1",
    "surprised": "surprised1",
    "helpful": "helpful1",
    "anxiety": "anxiety1",
    "uncertain": "uncertain1",
    "enthusiastic": "enthusiastic1",
    "understanding": "understanding1",
}


import numpy as np
from reachy_mini.utils import create_head_pose

class YawOffsetMove(Move):
    """將原始 RecordedMove 的頭部動作，整體旋轉一個水平 Yaw 偏移角度"""

    def __init__(self, base_move, yaw_offset_deg: float):
        self._base_move = base_move
        # Build a 4x4 Z-axis (yaw) rotation matrix
        rot = R.from_euler("z", yaw_offset_deg, degrees=True).as_matrix()
        self._rot_4x4 = np.eye(4)
        self._rot_4x4[:3, :3] = rot

    @property
    def duration(self) -> float:
        return self._base_move.duration

    def evaluate(self, t: float):
        # Get original (head_pose, antennas, body_yaw) at time t
        head, antennas, body_yaw = self._base_move.evaluate(t)

        if head is not None:
            # Left-multiply the forward head pose matrix by the Z-axis rotation matrix to face the target person
            head = self._rot_4x4 @ head

        return head, antennas, body_yaw
    

class ExpressionController:
    def __init__(self, mini):
        self.mini = mini
        self.library = RecordedMoves(DATASET)
        self.current_expression = None
        self._motion_lock = threading.Lock()
        self._current_yaw_deg = 0.0

    def look_at_norm_x(self, norm_x: float, max_yaw_deg: float = 35.0):
        """
        Smoothly turn Reachy's head toward norm_x (-1.0 left to +1.0 right).
        Skipped if a full body/emotion move is currently holding the lock.
        """
        if norm_x is None or self._motion_lock.locked():
            return

        # In camera coordinates, norm_x < 0 is image left (robot turns right/left depending on mirror)
        # Adjust the sign (-norm_x vs norm_x) based on your physical/simulator setup
        target_yaw = float(np.clip(-norm_x * max_yaw_deg, -max_yaw_deg, max_yaw_deg))

        # Only send command if angle changed by more than 2 degrees to prevent motor jitter
        if abs(target_yaw - self._current_yaw_deg) < 2.0:
            return

        self._current_yaw_deg = target_yaw
        pose = create_head_pose(yaw=target_yaw, degrees=True)
        self.mini.set_target(head=pose)

    def _get_offset_move(self, move_name: str) -> YawOffsetMove:
        """取得加上當前頭轉角偏移的動作"""
        raw_move = self.library.get(move_name)
        # create a RecordedMove without built-in audio (which is itself a valid Move object)
        motion_only = RecordedMove(move=raw_move.move, sound_path=None)
        # pass motion_only into YawOffsetMove
        return YawOffsetMove(motion_only, yaw_offset_deg=self._current_yaw_deg)
    
    def play_quick_nod(self):
        """Non-blocking friendly backchannel nod ('understanding2') while listening."""
        if self._motion_lock.locked():
            return
        with self._motion_lock:
            offset_nod = self._get_offset_move("understanding2")
            self.mini.play_move(offset_nod, initial_goto_duration=0.25)

    def play(self, expression: str, voice=None):
        if expression not in EXPRESSION_MAP:
            print(f"Unknown expression: {expression}")
            return

        with self._motion_lock:
            move_name = EXPRESSION_MAP[expression]
            # move = self.library.get(move_name)
            # motion_only = RecordedMove(move=move.move, sound_path=None)
            offset_main_move = self._get_offset_move(move_name)

            # face the speaker and play move
            self.mini.play_move(offset_main_move, initial_goto_duration=0.4)
            
            # if robot hasn't end talking, face the speaker and play 'attentive'
            if voice is not None and voice.speaking:
                self._play_filler_while_speaking(voice)

            self.current_expression = expression

    def _play_filler_while_speaking(self, voice):
        """Loop the 'attentive' move facing the speaker while Reachy is still talking, and cut
        it off as soon as the speech ends instead of finishing the 4 s move. Call with the
        lock held."""
        done = threading.Event()

        def cancel_when_quiet():
            while not done.wait(0.05):
                if not voice.speaking:
                    self.mini.cancel_move()
                    return

        watcher = threading.Thread(target=cancel_when_quiet, daemon=True)
        watcher.start()
        try:
            while voice.speaking:
                offset_fill_move = self._get_offset_move("attentive1")
                self.mini.play_move(offset_fill_move, initial_goto_duration=0.2)
                time.sleep(0.05)
        finally:
            done.set()
            watcher.join()
