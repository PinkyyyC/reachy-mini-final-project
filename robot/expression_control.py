import time
import threading
from reachy_mini.motion.recorded_move import RecordedMove, RecordedMoves

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




# class ExpressionController:

#     def __init__(self, mini):
#         self.mini = mini
#         self.library = RecordedMoves(DATASET)
#         self.current_expression = None

#     # def play(self, expression: str, target_duration: float = None):
#     #     if expression not in EXPRESSION_MAP:
#     #         print(f"Unknown expression: {expression}")
#     #         return

#     #     move_name = EXPRESSION_MAP[expression]
#     #     move = self.library.get(move_name)
#     #     motion_only = RecordedMove(move=move.move, sound_path=None)

#     #     start_time = time.time()
        
#     #     # 1. 先播放主表情動作
#     #     self.mini.play_move(motion_only, initial_goto_duration=0.5)

#     #     # 2. 如果語音時間還沒到，持續做自然的節奏性說話小動作 (例如微點頭)
#     #     if target_duration:
#     #         elapsed = time.time() - start_time
#     #         remaining = target_duration - elapsed
            
#     #         # 若剩餘時間超過 1 秒，循環播放短的自然微動動作 (如 attentive 或 nod)
#     #         while remaining > 0.8:
#     #             fill_move = self.library.get("attentive1")
#     #             fill_motion = RecordedMove(move=fill_move.move, sound_path=None)
#     #             self.mini.play_move(fill_motion, initial_goto_duration=0.2)
                
#     #             elapsed = time.time() - start_time
#     #             remaining = target_duration - elapsed

#     #     self.current_expression = expression
#     def play(self, expression: str, voice=None):
#         if expression not in EXPRESSION_MAP:
#             print(f"Unknown expression: {expression}")
#             return

#         move_name = EXPRESSION_MAP[expression]
#         move = self.library.get(move_name)
#         motion_only = RecordedMove(move=move.move, sound_path=None)

#         # 1. 先播放主要表情動作 (例如 enthusiastic1, thoughtful1)
#         self.mini.play_move(motion_only, initial_goto_duration=0.5)

#         # 2. 如果語音還在播 (voice.speaking 為 True)，就循環播放自然微動填充動作
#         fill_move = self.library.get("attentive1")
#         fill_motion = RecordedMove(move=fill_move.move, sound_path=None)

#         while voice is not None and voice.speaking:
#             # 只要講話還沒結束，就持續播放短動作
#             self.mini.play_move(fill_motion, initial_goto_duration=0.2)
#             time.sleep(0.05)

#         self.current_expression = expression

import numpy as np
from reachy_mini.utils import create_head_pose

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

    def play_quick_nod(self):
        """Non-blocking friendly backchannel nod ('understanding1') while listening."""
        if self._motion_lock.locked():
            return
        with self._motion_lock:
            move = self.library.get("understanding1")
            motion_only = RecordedMove(move=move.move, sound_path=None)
            self.mini.play_move(motion_only, initial_goto_duration=0.3)

    def play(self, expression: str, voice=None):
        if expression not in EXPRESSION_MAP:
            print(f"Unknown expression: {expression}")
            return

        with self._motion_lock:
            move_name = EXPRESSION_MAP[expression]
            move = self.library.get(move_name)
            motion_only = RecordedMove(move=move.move, sound_path=None)

            self.mini.play_move(motion_only, initial_goto_duration=0.5)

            if voice is not None and voice.speaking:
                self._play_filler_while_speaking(voice)

            self.current_expression = expression

    def _play_filler_while_speaking(self, voice):
        """Loop a gentle 'attentive' move while Reachy is still talking, and cut it off as
        soon as the speech ends instead of finishing the 4 s move. Call with the lock held."""
        fill_move = self.library.get("attentive1")
        fill_motion = RecordedMove(move=fill_move.move, sound_path=None)
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
                self.mini.play_move(fill_motion, initial_goto_duration=0.2)
        finally:
            done.set()
            watcher.join()