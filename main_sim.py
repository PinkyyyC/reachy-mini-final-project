"""Run Reachy's interaction loop against the simulator instead of the real robot.

Uses the same code as main.py, but with the laptop mic and camera and a simulator daemon
running on this Mac (no Wi-Fi robot, no sound-direction sensor).

1. Quit the Reachy Mini Control app (it runs its own simulator on the same port).
2. Start the MuJoCo simulator (a 3D window opens; it may appear behind other windows):
       .venv/bin/mjpython -m reachy_mini.daemon.app.main --sim
3. Then, in another terminal:
       .venv/bin/python main_sim.py
"""

import sys

import requests

from main import parse_args, run

START_MUJOCO = "    .venv/bin/mjpython -m reachy_mini.daemon.app.main --sim"


def check_simulator() -> bool:
    """Say which simulator is running. Returns False if none is."""
    try:
        status = requests.get("http://localhost:8000/api/daemon/status", timeout=2).json()
    except (requests.RequestException, ValueError):
        print("No simulator is running on this Mac. Start the MuJoCo simulator first:\n" + START_MUJOCO)
        return False

    if status.get("simulation_enabled"):
        print("Using the MuJoCo simulator (the 3D window shows the robot).")
    elif status.get("mockup_sim_enabled"):
        source = "the Reachy Mini Control app" if status.get("desktop_app_daemon") else "a mockup daemon"
        print(f"WARNING: connected to the mockup simulator from {source}. It has no 3D window.\n"
              "For MuJoCo, quit that first (Ctrl+C, or quit the Reachy Mini Control app), then run:\n"
              + START_MUJOCO)
    else:
        print("WARNING: the daemon on this Mac isn't a simulator; is a real robot plugged in by USB?")
    return True


if __name__ == "__main__":
    args = parse_args(description="Reachy Mini multimodal interaction (simulator on this Mac)")
    if not check_simulator():
        sys.exit(1)
    run(use_real_robot=False, camera_index=args.camera, mode=args.mode)