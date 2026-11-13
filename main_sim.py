"""Run Reachy's interaction loop against the simulator instead of the real robot.

Uses the same code as main.py, but with the laptop mic and camera and a simulator daemon
running on this Mac (no Wi-Fi robot, no sound-direction sensor).

1. Start the MuJoCo simulator (a 3D window opens):
       .venv/bin/mjpython -m reachy_mini.daemon.app.main --sim
   (or `.venv/bin/reachy-mini-daemon --mockup-sim` for a lightweight one without MuJoCo)
2. Then, in another terminal:
       .venv/bin/python main_sim.py
"""

from main import parse_args, run


if __name__ == "__main__":
    args = parse_args(description="Reachy Mini multimodal interaction (simulator on this Mac)")
    run(use_real_robot=False, camera_index=args.camera)
