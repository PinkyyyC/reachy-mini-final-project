import time
import threading
from reachy_mini import ReachyMini
import cv2
from collections import deque

from decision.fusion import make_decision
from decision.action_planner import ActionPlanner
from robot.expression_control import ExpressionController

from audio.speaker_nlp_model import Config, SpeakerNLPModel, RobotMic, LaptopMic
from audio.reachy_voice import ReachyVoice

from vision.step4_gaze_and_lips import GazeAndLipTracker
from vision.step6_combined_cv_doa import DoAPoller, fuse_cv_and_doa, draw_fusion_hud

EMOTION_MOVES = {
    "welcoming": "welcoming_move",
    "thoughtful": "thoughtful_move",
    "surprised": "surprised_move",
    "helpful": "helpful_move",
    "anxious": "anxious_move",
    "uncertain": "uncertain_move",
    "enthusiastic": "enthusiastic_move",
    "neutral": "neutral_move",
}


class VisionPipelineThread:
    """Runs CV + DoA fusion continuously in a background thread at ~30 FPS."""
    def __init__(self, camera_index: int = 0, robot_ip: str = "192.168.1.64", use_robot_doa: bool = False):
        self.tracker = GazeAndLipTracker(camera_index=camera_index)
        self.doa_poller = DoAPoller(robot_ip=robot_ip) if use_robot_doa else None
        self._lock = threading.Lock()
        self._running = True
        self._latest_fused = {
            "active_speaker": "none",
            "confidence": "NONE",
            "looking_at_robot": False,
            "interaction_mode": "IDLE / LISTENING",
            "target_norm_x": None,
            "doa": {"is_speech": False, "raw_deg": 90.0, "centered_deg": 0.0, "audio_side": "SILENT"},
            "cv": {},
        }
        
        # 新增：記憶「最近一次有效說話者」的狀態與時間戳記，解決 Whisper 延遲問題
        self._last_active_speaker = "none"
        self._last_looking_at_robot = False
        self._last_target_norm_x = None
        self._last_active_time = 0.0

        # # 儲存最近 90 幀（約 3 秒）的歷史狀態：(timestamp, active_speaker, looking_at_robot, target_norm_x)
        # self._history = deque(maxlen=90)
        

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        silent_doa = {"is_speech": False, "raw_deg": 90.0, "centered_deg": 0.0, "audio_side": "SILENT"}
        while self._running:
            annotated_frame, cv_state = self.tracker.process_frame()
            if annotated_frame is None:
                time.sleep(0.01)
                continue

            doa_state = self.doa_poller.get_state() if self.doa_poller else silent_doa
            fused = fuse_cv_and_doa(cv_state, doa_state)
            now = time.time()

            with self._lock:
                self._latest_fused = fused

                # 只要偵測到有人正在說話，就刷新記憶緩衝區
                if fused["active_speaker"] in ("person_left", "person_right"):
                    self._last_active_speaker = fused["active_speaker"]
                    self._last_looking_at_robot = fused["looking_at_robot"]
                    self._last_target_norm_x = fused["target_norm_x"]
                    self._last_active_time = now

            
                # # 只記錄確實有偵測到人在說話的影格
                # if fused["active_speaker"] in ("person_left", "person_right"):
                #     self._history.append((
                #         now,
                #         fused["active_speaker"],
                #         fused["looking_at_robot"],
                #         fused["target_norm_x"]
                #     ))

            draw_fusion_hud(annotated_frame, fused)
            cv2.imshow("Reachy Multimodal Vision", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest_fused)

    def stop(self):
        self._running = False
        if self.doa_poller:
            self.doa_poller.stop()
        self.tracker.release()

    def get_utterance_context(self, lookback_sec: float = 4.0) -> dict:
        """
        取得供 Turn-taking 決策使用的視覺狀態：
        若當下嘴巴已閉上，則回溯過去 lookback_sec 秒內最後一位說話者的視線狀態。
        """
        with self._lock:
            fused = dict(self._latest_fused)
            # 如果當下已經沒人動嘴，但 2 秒內剛有人講完話，就套用那個人講話時的視線狀態
            if fused["active_speaker"] == "none" and (time.time() - self._last_active_time <= lookback_sec):
                fused["active_speaker"] = self._last_active_speaker
                fused["looking_at_robot"] = self._last_looking_at_robot
                fused["target_norm_x"] = self._last_target_norm_x
                fused["confidence"] = "BUFFERED_SPEAKER"
            return fused

    # def get_utterance_context(self, lookback_sec: float = 2.5, gaze_ratio_threshold: float = 0.4) -> dict:
    #     """
    #     回溯過去 lookback_sec 秒內的所有說話影格：
    #     1. 找出這段期間說話幀數最多的人作為 speaker。
    #     2. 只要他在講話期間有超過 40% (gaze_ratio_threshold) 的時間看著機器人，就判定 looking_at_robot = True。
    #     """
    #     now = time.time()
    #     with self._lock:
    #         fused = dict(self._latest_fused)
    #         recent_frames = [item for item in self._history if (now - item[0]) <= lookback_sec]

    #     if not recent_frames:
    #         return fused

    #     # 統計這段期間誰說最多話
    #     speakers = [f[1] for f in recent_frames]
    #     dominant_speaker = max(set(speakers), key=speakers.count)

    #     # 篩選出該說話者的影格，計算看機器人的比例與平均座標
    #     speaker_frames = [f for f in recent_frames if f[1] == dominant_speaker]
    #     gaze_true_count = sum(1 for f in speaker_frames if f[2] is True)
    #     gaze_ratio = gaze_true_count / len(speaker_frames)

    #     valid_norm_x = [f[3] for f in speaker_frames if f[3] is not None]
    #     avg_norm_x = sum(valid_norm_x) / len(valid_norm_x) if valid_norm_x else None

    #     fused["active_speaker"] = dominant_speaker
    #     fused["looking_at_robot"] = (gaze_ratio >= gaze_ratio_threshold)
    #     fused["target_norm_x"] = avg_norm_x
    #     fused["confidence"] = f"BUFFERED_RATIO ({gaze_ratio:.0%})"

    #     return fused

def play_robot_response(voice: ReachyVoice, expression_controller: ExpressionController, text: str, emotion: str):
    """
    When the speech starts (on_start callback is called), 
    start a new thread to play the motion,
    so the motion can run concurrently with the speech.
    """
    expression_thread = None
    def _trigger_motion():
        nonlocal expression_thread
        if emotion:
            # expression_controller.play(emotion)
            # Start a new thread to play the motion, so it doesn't block the voice's _play_loop
            expression_thread = threading.Thread(
                target=expression_controller.play,
                args=(emotion,),
                kwargs={"voice": voice},
                daemon=True
            )
            expression_thread.start()

    # voice is non-blocking, so can trigger motion concurrently with speech
    voice.say(text, on_start=_trigger_motion)
    
    # 若此處仍需等待講完再繼續下一個 stage，可呼叫 wait()
    voice.wait()

    # wait until the expression thread is finished 
    # if the motion is longer than the speech, ensure the motion is completed before proceeding to the next round
    if expression_thread is not None:
        expression_thread.join()
    


def main():
    print("Connecting to Reachy Mini...")

    # set to False when running simulator, True when using physical robot
    # USE_REAL_ROBOT_HARDWARE = False
    USE_REAL_ROBOT_HARDWARE = True
    ROBOT_IP = "192.168.1.64"

    # with ReachyMini(host=ROBOT_IP, connection_mode="network") as mini:
    with ReachyMini() as mini:
        voice = ReachyVoice()
        # mic = None  # 若在沒有實體機器人/測試階段可暫時為 None
        
        
        cfg = Config(use_llm=True, llm_backend="claude") # 或 "ollama"
        # mic = LaptopMic(cfg) 
        mic = RobotMic(mini) if USE_REAL_ROBOT_HARDWARE else LaptopMic(cfg)
        nlp = SpeakerNLPModel(
            cfg=cfg,
            mini=mini if USE_REAL_ROBOT_HARDWARE else None,
            mic=mic,
            emotions=EMOTION_MOVES,
            voice=voice
        )
        
        planner = ActionPlanner()
        expression_controller = ExpressionController(mini)
        vision_pipeline = VisionPipelineThread(
            camera_index=0,
            robot_ip=ROBOT_IP,
            use_robot_doa=USE_REAL_ROBOT_HARDWARE
        )

        nlp.start()
        print("\n========== START INTERACTION LOOP ==========")

        last_nod_time = 0.0

        try:
            while True:
                # update NLP status (about 10Hz)
                nlp_status = nlp.update()
                # vision_output = vision_tracker.get_latest()
                fused_cv = vision_pipeline.get_latest()
                # now = time.time()

                # interrupting: True if user is speaking while robot is speaking
                if nlp_status["interrupting"] and voice.speaking:
                    print("\n[Robot] Interrupted by user! Stopping speech...")
                    voice.stop()
                    # motion stop?

                # If the robot is not speaking, check if the user is speaking and if they are looking at the robot
                # gaze tracking the speaker + nodding
                if not voice.speaking:
                    listening_action = planner.check_active_listening(fused_cv)

                    if listening_action["target_norm_x"] is not None:
                        expression_controller.look_at_norm_x(listening_action["target_norm_x"])

                    if listening_action["should_nod"]:
                        print(f"[Robot] Nodding to {fused_cv['active_speaker']}...")
                        threading.Thread(
                            target=expression_controller.play_quick_nod,
                            daemon=True
                        ).start()

                #turn-taking
                utt = nlp_status["utterance"]
                

                # if utt:
                #     print(f"\n[Heard #{utt['id']}] {utt['text']} "
                #           f"(said_name={utt['said_name']}, cv_speaker={fused_cv['active_speaker']}, "
                #           f"looking={fused_cv['looking_at_robot']})")

                #     audio_output = {"utterance": utt}
                #     decision = make_decision(audio_output, fused_cv)
                #     action = planner.process_decision(decision)

                if utt:
                    # 使用帶有 2 秒記憶緩衝的視覺狀態，避免因為閉嘴等待 Whisper 辨識而漏掉狀態
                    utterance_cv = vision_pipeline.get_utterance_context(lookback_sec=2.0)

                    print(f"\n[Heard #{utt['id']}] {utt['text']} "
                          f"(said_name={utt['said_name']}, cv_speaker={utterance_cv['active_speaker']}, "
                          f"looking={utterance_cv['looking_at_robot']})")

                    audio_output = {"utterance": utt}
                    decision = make_decision(audio_output, utterance_cv)
                    action = planner.process_decision(decision)    


                    print(f"[Decision] addressee={decision['addressee']} "
                          f"({decision['evidence']['addressee_method']}) -> should_speak={action['should_speak']}")

                    if action["should_speak"]:
                        # face the spaeker before replying
                        if decision["target_norm_x"] is not None:
                            expression_controller.look_at_norm_x(decision["target_norm_x"])

                        print(f"[Robot] Generating LLM response...")
                        nlp_reply = nlp.respond(utt["id"])
                        response_plan = planner.process_response(nlp_reply)

                        print(f"[Robot] Reply: {response_plan['response_text']} "
                              f"(Emotion: {response_plan['expression']})")

                        play_robot_response(
                            voice=voice,
                            expression_controller=expression_controller,
                            text=response_plan["response_text"],
                            emotion=response_plan["expression"]
                        )
                        planner.finish_speaking()

                time.sleep(0.05)  # ~20Hz main loop freq

        except KeyboardInterrupt:
            print("\nExiting loop...")
        finally:
            nlp.stop()
            voice.stop()

if __name__ == "__main__":
    main()