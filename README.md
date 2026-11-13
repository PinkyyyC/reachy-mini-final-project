# Reachy Mini: multi-party conversation

Reachy Mini sits in front of two people and takes part in their conversation. It works out
**who is speaking** and **whether they are talking to Reachy or to each other**, and only
answers when it's being spoken to. It replies out loud with a matching emotion move, turned
toward the person it's answering.

Everything runs on a Mac. Replies are written by Claude; speech-to-text, speech detection,
face tracking and the voice all run locally.

## How it works

```
 robot mic ──► speech detection (Silero) ──► speech-to-text (Whisper) ──► utterance
                                                                              │
 camera ─────► face, gaze and lip tracking (MediaPipe) ─┐                     ▼
                                                        ├─► who is speaking ─► decision:
 robot mic array ─► sound direction (DoA) ──────────────┘   + looking?         answer or not?
                                                                              │ answer
                                                                              ▼
            voice (Kokoro) ◄── sentences as they're written ◄── Claude reply + emotion
                 │                                                            │
                 └── Reachy speaks ──────────── emotion move facing the speaker ◄┘
```

- **Vision** (`vision/`): MediaPipe tracks up to two faces. For each it estimates head
  direction (looking at Reachy or at the partner) and lip movement (speaking or not). This is
  fused with the robot's sound direction to pick the active speaker. A short memory buffer
  keeps the last speaker for 2 s, so the decision still knows who spoke after their mouth
  has closed.
- **Audio** (`audio/speaker_nlp_model.py`): splits the mic stream into sentences and
  transcribes them with Whisper. It also marks whether the robot's name was said and whether
  the sentence answers a question Reachy just asked.
- **Decision** (`decision/fusion.py`): decides whether the sentence was meant for Reachy
  (rules below).
- **Reply and voice** (`audio/`): Claude writes a short reply and picks an emotion. The reply
  is streamed, so Reachy starts speaking at the first sentence while the rest is still being
  written.
- **Robot** (`robot/`): plays an emotion move from Pollen's emotions library, rotated to face
  the speaker. Between replies the head follows whoever is talking, with an occasional nod.

### When Reachy answers

In order of priority (`make_decision` in `decision/fusion.py`):

| Situation | Reachy |
|---|---|
| Answering a question Reachy just asked **and** looking at Reachy | answers |
| Said "Reachy" (anywhere in the sentence) | answers |
| The speaker is looking at Reachy | answers |
| The speaker is looking away (at their partner) | stays quiet |
| Nobody identified as the speaker | stays quiet |

"Answering Reachy" means the sentence started within 8 s of Reachy finishing a question.
While someone talks and looks at Reachy for 1.2 s, Reachy nods.

## Project layout

| Path | What it does |
|---|---|
| `main.py` | The real robot over Wi-Fi: the full interaction loop |
| `main_sim.py` | The same loop against a simulator on this Mac, with the laptop mic and camera |
| `audio/speaker_nlp_model.py` | Speech detection, speech-to-text, utterance flags, Claude replies |
| `audio/reachy_voice.py` | Text-to-speech (Kokoro, with macOS `say` as a fallback) |
| `decision/fusion.py` | Who is being addressed, and should Reachy answer |
| `decision/action_planner.py` | Listening behaviour (head follows the speaker, nods) and emotion-to-move mapping |
| `vision/step4_gaze_and_lips.py` | Face, gaze and lip tracking |
| `vision/step6_combined_cv_doa.py` | Combines vision with the robot's sound direction |
| `robot/expression_control.py` | Emotion moves and nods, turned toward the speaker |
| `robot/connection.py` | Connects to the robot over Wi-Fi (works around an SDK streaming issue) |
| `tests/` | Tests for the speech pipeline |

## Setup

You need a Mac with Apple Silicon (M1 or later) and **Python 3.12**.

1. Create the environment and install the packages:
   ```bash
   python3.12 -m venv .venv
   ```
   ```bash
   .venv/bin/pip install -r requirements.txt
   ```

2. Download the Kokoro voice into `audio/models/`. These files are too big for git.
   ```bash
   mkdir -p audio/models
   ```
   ```bash
   curl -L -o audio/models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
   ```
   ```bash
   curl -L -o audio/models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
   ```
   `kokoro-v1.0.onnx` (326 MB) is about twice as fast on a Mac as the smaller
   `kokoro-v1.0.int8.onnx` (92 MB); either works and the fastest one present is used. Without
   them, Reachy falls back to the macOS `say` voice.

3. Add your Claude API key to `audio/.env` (this file is git-ignored):
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
   Without a key, Reachy answers with pre-written lines.

4. Allow your terminal to use the **camera** and **microphone** in System Settings →
   Privacy & Security.

The face model (`vision/models/face_landmarker.task`) is already in the repo. The Whisper
speech-to-text model (about 150 MB) downloads automatically on the first run.

## Run it

### With the real robot

Put the robot and the Mac on the same Wi-Fi, then:

```bash
.venv/bin/python main.py
```

It connects to `reachy-mini.local`. A camera window opens with the tracking overlay. Close
it or press `q` to stop. If the wrong camera opens (for example an iPhone through Continuity
Camera), pick another one with `--camera 1`.

### In the simulator

1. Quit the **Reachy Mini Control** app if it's open. It runs its own simulator on the same
   port.
2. Start the MuJoCo simulator. A 3D window with the robot opens; it may appear behind other
   windows.
   ```bash
   .venv/bin/mjpython -m reachy_mini.daemon.app.main --sim
   ```
3. In a second terminal:
   ```bash
   .venv/bin/python main_sim.py
   ```

`main_sim.py` says which simulator it found. The simulator has no sound-direction sensor, so
the speaker is worked out from the camera alone. For a lighter simulator without the 3D view,
use `.venv/bin/reachy-mini-daemon --mockup-sim` in step 2.

### Try one part at a time

None of these need the robot.

| Command | What it does |
|---|---|
| `.venv/bin/python audio/speaker_nlp_model.py --speak` | Talk to Reachy with the laptop mic and speakers (it answers when you say "Reachy") |
| `.venv/bin/python audio/speaker_nlp_model.py --typed` | Type what people say instead of speaking |
| `.venv/bin/python audio/speaker_nlp_model.py --watch` | Show the conversation log live, from another terminal |
| `.venv/bin/python audio/reachy_voice.py "Hello, I'm Reachy."` | Test the voice |
| `.venv/bin/python -m vision.step4_gaze_and_lips` | Test face, gaze and lip tracking with the camera |
| `.venv/bin/python -m pytest tests` | Run the tests |

## Reading the output

```
[Heard #3] What should we pack for the beach? (said_name=False, cv_speaker=person_right, looking=True)
[Decision] addressee=robot (gaze_and_lips) -> should_speak=True
[Robot] says: Sunscreen, a hat, and sandals are the basics.
[Timing] end of speech -> sentence ready 0.05s -> first reply sentence 0.79s -> first sound 1.24s
```

- **`[Heard]`**: what was transcribed, who the camera thinks said it, and whether they were
  looking at Reachy.
- **`[Decision]`**: whether Reachy answers and which rule decided it.
- **`[Timing]`**: from the end of the person's sentence to the transcript, to Claude's first
  sentence, to Reachy's first sound.
- **`[voice] … silent gap …`**: Reachy paused for more than 0.3 s mid-reply, and why: either
  the next sentence wasn't written yet, or the voice was slow to produce it.

Every conversation is also saved to `audio/logs/conversation.csv` (one row per sentence) and
`audio/logs/speaker_nlp.csv` (10 rows a second). Both are git-ignored.

## Settings

Most behaviour is set in `Config` in `audio/speaker_nlp_model.py`:

| Setting | Default | Meaning |
|---|---|---|
| `silence_end_s` | `0.4` | Pause that ends a sentence |
| `long_silence_s` | `1.0` | Pause allowed instead when the words so far look unfinished ("It's just…") |
| `stt_model` | `"base.en"` | Whisper model; `"small.en"` is more accurate but about 0.3 s slower per sentence |
| `stt_device` | `"auto"` | Apple GPU if available, else CPU; `"cpu"` to force the CPU |
| `llm_model` | `"claude-haiku-5-5"` | The Claude model for replies |
| `llm_backend` | `"claude"` | Or `"ollama"` for free local replies (needs the Ollama app) |
| `followup_window_s` | `8.0` | How soon after Reachy's question a sentence counts as an answer |
| `max_reply_sentences` | `2` | Reply length |
| `task_items` | trip-planning topics | The topics of the study task; edit to match your study |

Head-direction thresholds for "looking at Reachy" are at the top of
`vision/step4_gaze_and_lips.py`.

Reachy's voice is Kokoro's `am_fenrir`, the closest match we found to the voice Pollen's own
app uses. To try another one, pass `kokoro_voice=` (and optionally `speed=`) to
`ReachyVoice()` in `main.py`, for example `ReachyVoice(kokoro_voice="am_michael")`. Kokoro
has about 20 American and 8 British English voices; see `Kokoro(...).get_voices()`.

## Troubleshooting

- **The camera window freezes.** Check the terminal: the program now prints the cause. The
  usual one is `Lost connection with the server`, meaning the robot dropped off the Wi-Fi.
- **No MuJoCo window.** Another simulator is running, usually the Reachy Mini Control app's.
  `main_sim.py` warns about this. Quit it, then start MuJoCo again.
- **`Cannot open camera 0`.** Allow camera access for your terminal in System Settings →
  Privacy & Security → Camera, or try `--camera 1`.
- **The voice is slow or robotic.** The first line printed should be
  `[voice] kokoro (am_fenrir, kokoro-v1.0.onnx)`. If it says `say`, the Kokoro files are
  missing from `audio/models/`.
- **Harmless warnings:** `No Reachy Mini Audio USB device found` (the laptop isn't the robot),
  `libgstpython.dylib … Library not loaded` (an optional GStreamer plugin), and occasional
  `IK error: … head pose not achievable` (some recorded moves reach the edge of the head's
  range).
