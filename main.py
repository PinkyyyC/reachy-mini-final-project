import argparse
import sys
import time
import threading
import traceback
from collections import deque
from reachy_mini import ReachyMini
from robot.connection import NetworkReachyMini
import cv2

from decision.fusion import make_decision, summarize_gaze
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
    def __init__(self, camera_index: int = 0, robot_ip: str = "reachy-mini.local", use_robot_doa: bool = False):
        self.tracker = GazeAndLipTracker(camera_index=camera_index)
        self.doa_poller = DoAPoller(robot_ip=robot_ip) if use_robot_doa else None
        self._lock = threading.Lock()
        self._running = True
        self._latest_frame = None
        self._error = None
        self._history = deque(maxlen=900)   # (time.time(), fused) for the last ~30 s at 30 FPS
        self._last_frame_t = None
        self._stall_reported = False
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
            print("\n[Vision] The vision thread crashed:")
            traceback.print_exc()
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
                self._history.append((time.time(), fused))
                self._last_frame_t = time.monotonic()

    def show_preview(self):
        """Pump the native window on the main thread (required by macOS)."""
        with self._lock:
            frame, error, last_frame_t = self._latest_frame, self._error, self._last_frame_t
        if error is not None:
            raise RuntimeError("Vision pipeline failed") from error
        self._report_if_stalled(last_frame_t)
        if frame is not None:
            cv2.imshow("Reachy Multimodal Vision", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def _report_if_stalled(self, last_frame_t):
        """If no new frame arrives for 2 s, print once where the vision thread is stuck."""
        stalled = last_frame_t is not None and time.monotonic() - last_frame_t > 2.0
        if stalled and not self._stall_reported:
            self._stall_reported = True
            print(f"\n[Vision] No new camera frame for {time.monotonic() - last_frame_t:.1f}s. "
                  "The vision thread is currently here:")
            stack = sys._current_frames().get(self._thread.ident)
            if stack is not None:
                print("".join(traceback.format_stack(stack)))
        elif not stalled and self._stall_reported:
            self._stall_reported = False
            print("[Vision] Camera frames are coming in again.")

    def get_latest(self) -> dict:
        with self._lock:
            return dict(self._latest_fused)

    def gaze_during(self, start_t: float, end_t: float) -> dict:
        """Who spoke and whether they looked at Reachy between two time.time() stamps."""
        with self._lock:
            frames = list(self._history)
        return summarize_gaze(frames, start_t, end_t)

    def stop(self):
        self._running = False
        if self.doa_poller:
            self.doa_poller.stop()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            print("[Vision] The vision thread didn't stop (stuck reading the camera?); leaving it.")
        else:
            self.tracker.release()

class ReplyTurn:
    """Reachy's turn: from the start of a reply until its last sentence has been played.

    Passed to ExpressionController.play() as `voice`, so the filler movement keeps going in
    the gap while Claude is still writing the next sentence, and stops when the turn ends.
    """

    def __init__(self, voice: ReachyVoice):
        self.voice = voice
        self.writing = True

    @property
    def speaking(self) -> bool:
        return self.writing or self.voice.speaking


def speak_reply(nlp: SpeakerNLPModel, voice: ReachyVoice, expression_controller: ExpressionController,
                planner: ActionPlanner, utt: dict, heard_at: float) -> dict:
    """Stream Reachy's reply: each sentence is spoken as soon as Claude has written it, and the
    emotion movement starts with the first sound. Returns once the last sentence has played."""
    turn = ReplyTurn(voice)
    lock = threading.Lock()
    state = {"expression": None, "sound_started": False, "motion_started": False,
             "first_sentence_t": None, "first_sound_t": None}

    def start_motion_locked():
        if state["expression"] and state["sound_started"] and not state["motion_started"]:
            state["motion_started"] = True
            threading.Thread(target=expression_controller.play, args=(state["expression"],),
                             kwargs={"voice": turn}, daemon=True).start()

    def on_emotion(emotion: str):
        with lock:
            state["expression"] = planner.process_response({"emotion": emotion})["expression"]
            start_motion_locked()

    def on_first_sound():
        with lock:
            state["sound_started"] = True
            state["first_sound_t"] = time.time()
            start_motion_locked()

    def on_sentence(sentence: str):
        first = state["first_sentence_t"] is None
        if first:
            state["first_sentence_t"] = time.time()
        print(f"[Robot] says: {sentence}")
        voice.say(sentence, on_start=on_first_sound if first else None)

    try:
        reply = nlp.respond(utt["id"], on_sentence=on_sentence, on_emotion=on_emotion)
    finally:
        turn.writing = False
    voice.wait()
    planner.finish_speaking()

    end_t = utt.get("end_t") or heard_at
    parts = [f"end of speech -> sentence ready {heard_at - end_t:.2f}s"]
    if state["first_sentence_t"]:
        parts.append(f"-> first reply sentence {state['first_sentence_t'] - end_t:.2f}s")
    if state["first_sound_t"]:
        parts.append(f"-> first sound {state['first_sound_t'] - end_t:.2f}s")
    print(f"[Timing] {' '.join(parts)}  (emotion: {reply['emotion']}, {reply['source']})")
    return reply


def parse_args(argv=None, description="Reachy Mini multimodal interaction (real robot over Wi-Fi)"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    return parser.parse_args(argv)


def main(argv=None):
    """The real robot over Wi-Fi. For the simulator, run main_sim.py instead."""
    run(use_real_robot=True, camera_index=parse_args(argv).camera)


def run(use_real_robot: bool, camera_index: int = 0):
    """use_real_robot=True: the robot over Wi-Fi with its mic and sound direction.
    False: a simulator daemon on this Mac with the laptop mic (see main_sim.py)."""
    USE_REAL_ROBOT_HARDWARE = use_real_robot
    ROBOT_IP = "reachy-mini.local"

    if USE_REAL_ROBOT_HARDWARE:
        print(f"Connecting to Reachy Mini at {ROBOT_IP}...")
        robot = NetworkReachyMini(host=ROBOT_IP, connection_mode="network")
    else:
        print("Connecting to the simulator daemon on this Mac (start it first)...")
        robot = ReachyMini()  # local simulator / daemon on this machine

    with robot as mini:
        voice = ReachyVoice()
        # mic = None  # 若在沒有實體機器人/測試階段可暫時為 None
        
        
        # Nobody talks while Reachy talks, so the mic is ignored then (no false interruptions
        # from Reachy's own voice).
        cfg = Config(use_llm=True, llm_backend="claude", allow_interruptions=False) # 或 "ollama"
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
            camera_index=camera_index,
            robot_ip=ROBOT_IP,
            use_robot_doa=USE_REAL_ROBOT_HARDWARE
        )

        nlp.start()
        print("\n========== START INTERACTION LOOP ==========")

        stop_event = threading.Event()
        interaction_errors = []

        def run_reply(utt: dict, heard_at: float):
            try:
                speak_reply(nlp, voice, expression_controller, planner, utt, heard_at)
            except Exception as exc:
                print("\n[Robot] Replying failed:")
                traceback.print_exc()
                interaction_errors.append(exc)
                stop_event.set()

        def interact():
            reply_thread = None
            try:
                while not stop_event.is_set():
                    # Keep calling update() even while Reachy talks: it drops Reachy's own voice
                    # from the mic and starts the echo guard when Reachy stops.
                    nlp_status = nlp.update()
                    fused_cv = vision_pipeline.get_latest()
                    replying = reply_thread is not None and reply_thread.is_alive()

                    # interrupting: True if user is speaking while robot is speaking
                    if nlp_status["interrupting"] and voice.speaking:
                        print("\n[Robot] Interrupted by user! Stopping speech...")
                        voice.stop()

                    # While listening, the head follows whoever is speaking. Nods now happen at
                    # the end of a statement made while looking at Reachy (see make_decision).
                    if not replying and not voice.speaking:
                        listening_action = planner.check_active_listening(fused_cv)
                        if listening_action["target_norm_x"] is not None:
                            expression_controller.look_at_norm_x(listening_action["target_norm_x"])

                    utt = nlp_status["utterance"]
                    if utt:
                        heard_at = time.time()
                        gaze = vision_pipeline.gaze_during(utt["t"], utt["end_t"])
                        print(f"\n[Heard #{utt['id']}] {utt['text']} "
                              f"(intent={utt['intent']}, name_first={utt['name_first']}, "
                              f"speaker={gaze['speaker']}, looking={gaze['looking_fraction']:.0%})")

                        if replying:
                            print("[Decision] Reachy is still replying; not answering this one")
                        else:
                            decision = make_decision({"utterance": utt}, gaze)
                            planner.process_decision(decision)
                            print(f"[Decision] {decision['action']} "
                                  f"({decision['evidence']['addressee_method']})")

                            if decision["action"] == "speak":
                                # face the speaker before replying
                                if decision["target_norm_x"] is not None:
                                    expression_controller.look_at_norm_x(decision["target_norm_x"])
                                reply_thread = threading.Thread(
                                    target=run_reply, args=(utt, heard_at), daemon=True)
                                reply_thread.start()
                            elif decision["action"] == "nod":
                                threading.Thread(target=expression_controller.play_quick_nod,
                                                 daemon=True).start()

                    time.sleep(0.05)  # ~20Hz main loop freq

            except Exception as exc:
                print("\n[Robot] The interaction loop crashed:")
                traceback.print_exc()
                interaction_errors.append(exc)
            finally:
                stop_event.set()
                if reply_thread is not None:
                    reply_thread.join(timeout=cfg.llm_timeout_s + 2)

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
            print("\n[Main] Shutting down...")
            stop_event.set()
            # Close the window first, so it doesn't sit there frozen while the rest shuts down.
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
            voice.stop()
            vision_pipeline.stop()
            if interaction_thread.ident is not None:
                interaction_thread.join(timeout=cfg.llm_timeout_s + 5)
            nlp.stop()
            print("[Main] Disconnecting from the robot...")


if __name__ == "__main__":
    main()
