import time
import threading
from robot.connection import NetworkReachyMini
import cv2

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
        self._latest_frame = None
        self._error = None
        self._latest_fused = {
            "active_speaker": "none",
            "confidence": "NONE",
            "looking_at_robot": False,
            "interaction_mode": "IDLE / LISTENING",
            "target_norm_x": None,
            "doa": {"is_speech": False, "raw_deg": 90.0, "centered_deg": 0.0, "audio_side": "SILENT"},
            "cv": {},
        }
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._process_frames()
        except Exception as exc:
            with self._lock:
                self._error = exc
            self._running = False

    def _process_frames(self):
        silent_doa = {"is_speech": False, "raw_deg": 90.0, "centered_deg": 0.0, "audio_side": "SILENT"}
        last_frame_time = time.monotonic()
        while self._running:
            annotated_frame, cv_state = self.tracker.process_frame()
            if annotated_frame is None:
                if time.monotonic() - last_frame_time > 5:
                    raise RuntimeError("No camera frames received. Check macOS Camera permissions and camera_index.")
                time.sleep(0.01)
                continue
            last_frame_time = time.monotonic()

            doa_state = self.doa_poller.get_state() if self.doa_poller else silent_doa
            fused = fuse_cv_and_doa(cv_state, doa_state)

            draw_fusion_hud(annotated_frame, fused)
            with self._lock:
                self._latest_fused = fused
                self._latest_frame = annotated_frame

    def show_preview(self):
        """Pump the native window on the main thread (required by macOS)."""
        with self._lock:
            frame, error = self._latest_frame, self._error
        if error is not None:
            raise RuntimeError("Vision pipeline failed") from error
        if frame is not None:
            cv2.imshow("Reachy Multimodal Vision", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest_fused)

    def stop(self):
        self._running = False
        if self.doa_poller:
            self.doa_poller.stop()
        self._thread.join()
        self.tracker.release()

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
    ROBOT_IP = "reachy-mini.local"

    with NetworkReachyMini(host=ROBOT_IP, connection_mode="network") as mini:
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

        stop_event = threading.Event()
        interaction_errors = []

        def interact():
            try:
                while not stop_event.is_set():
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
                    #     print(f"\n[Heard] {utt['text']} (said_name: {utt['said_name']}, intent: {utt['intent']})")
                    # # 3. 取得 Vision 資訊 (真實環境中由你的視覺執行緒/模組取得)
                    # # vision_output = vision_module.get_latest()
                
                    # if utt:
                    #     # 4. Multi-modal Fusion decision-making
                    #     audio_output = {"utterance": utt}
                    
                    #     # 帶入決策模組判斷是否該由機器人回話
                    #     # should_reply = make_decision(utt, vision_output)
                    #     # decision = make_decision(audio_output, vision_output)
                    #     # action = planner.process_decision(decision)
                    #     # speaker = vision_output.get("visual_active_speaker")
                    #     # looking = vision_output[speaker]["Looking_at_robot"] if speaker else False
                    
                    #     should_reply = utt["said_name"] or utt["answering_robot"] # 或依據 action["should_speak"]  #or looking

                    #     if should_reply:
                    #         print(f"\n[Robot] Thinking response to: {utt['text']}")
                        
                    #         # 5. trigger LLM to generate response and emotion (blocking)
                    #         reply = nlp.respond(utt["id"])
                    #         print(f"[Robot] Reply: {reply['text']} (Emotion: {reply['emotion']})")

                    #         # 6. 同步執行 Motion + TTS
                    #         play_robot_response(
                    #             voice=voice,
                    #             expression_controller=expression_controller,
                    #             text=reply["text"],
                    #             emotion=reply["emotion"]
                    #         )
                    #         planner.finish_speaking()

                    # time.sleep(0.1) #0.05

                    if utt:
                        print(f"\n[Heard #{utt['id']}] {utt['text']} "
                              f"(said_name={utt['said_name']}, cv_speaker={fused_cv['active_speaker']}, "
                              f"looking={fused_cv['looking_at_robot']})")

                        audio_output = {"utterance": utt}
                        decision = make_decision(audio_output, fused_cv)
                        action = planner.process_decision(decision)

                        print(f"[Decision] addressee={decision['addressee']} "
                              f"({decision['evidence']['addressee_method']}) -> should_speak={action['should_speak']}")

                        if action["should_speak"]:
                            # face the spaeker before replying
                            if decision["target_norm_x"] is not None:
                                expression_controller.look_at_norm_x(decision["target_norm_x"])

                            print(f"[Robot] Generating LLM response...")
                            nlp_reply = nlp.respond(utt["id"])
                            if stop_event.is_set():
                                break
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

            except Exception as exc:
                interaction_errors.append(exc)
            finally:
                stop_event.set()

        interaction_thread = threading.Thread(target=interact, daemon=True)
        try:
            cv2.namedWindow("Reachy Multimodal Vision", cv2.WINDOW_NORMAL)
            interaction_thread.start()
            while not stop_event.is_set():
                if not vision_pipeline.show_preview():
                    break
                if cv2.getWindowProperty("Reachy Multimodal Vision", cv2.WND_PROP_VISIBLE) < 1:
                    break
                time.sleep(0.01)
            if interaction_errors:
                raise RuntimeError("Interaction loop failed") from interaction_errors[0]
        except KeyboardInterrupt:
            print("\nExiting loop...")
        finally:
            stop_event.set()
            voice.stop()
            vision_pipeline.stop()
            if interaction_thread.ident is not None:
                interaction_thread.join()
            nlp.stop()


if __name__ == "__main__":
    main()
