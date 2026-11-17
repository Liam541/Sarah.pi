# SARAH — Raspberry Pi 5 robot

SARAH has explicit **remote control** and **onboard AI autonomous** driving modes.
It starts stopped in manual mode. The Xbox controller can connect by USB or
Bluetooth; losing an attached controller stops the robot and latches the stop.

Both entry points launch the same application:

```bash
python sarah_pi.py
python "STL Files/sarah_pi.py"
```

**Copy the entire repository to the Pi.** The script in `STL Files` is a compatible
launcher, not a standalone copy. Keep `sarah/` and `robot_config.json` with it.

## How autonomous driving works

1. Stop and obtain a fresh front-camera image and front-left, rear and front-right
   ultrasonic readings.
2. Ask the **local vision-language model on the Pi** to choose `FORWARD`, `LEFT`,
   `RIGHT`, `BACKWARD` or `STOP`, using the image, distances and recent maneuvers.
3. Validate the response, mode generation, current camera and current distances.
   Discard an answer if the user changed modes or the scene changed substantially.
4. Execute one bounded maneuver while checking fresh observations every 50 ms.
   Forward steps are at most 0.6 seconds by default; turns/reverses at most 0.35.
5. Stop, observe and ask the onboard AI again.

Navigation uses Ollama at `127.0.0.1:11434`, independently of the companion chat
settings. The default vision model is **moondream**. It receives actual camera
JPEGs and measured distances. A missing model, timeout, malformed response or
unavailable sensor leaves the robot stopped. There is no remote navigation server
or automatic fallback that drives without the AI.

The motor controller runs an independent watchdog every 20 ms. Commands expire
after 250 ms without renewal. It checks directional obstacle clearance, sensor
age and camera age; an AI decision cannot override those checks. Motion is
serialized, and results from a previous mode cannot restart the motors.

This is **reactive indoor exploration**, not SLAM or goal navigation. The earlier
time-based pretend map/return-home system was removed. “Return to start” now stops
and explains that localization or encoders are needed. There is no wheel odometry,
side or cliff sensing. Camera scene-change detection is only an extra veto, not
depth measurement or reliable stair detection. Use a flat, enclosed test area.

Local model inference on a Pi can take seconds or tens of seconds. The robot
intentionally remains stopped while thinking. Real inference time, motor startup
duty cycle and stopping distance still require measurements on your robot.

## Raspberry Pi setup

Use current 64-bit Raspberry Pi OS and its system Python packages. The GPIO
backend is `lgpio`; do not use the old pigpio/native fallback or auto-sudo launcher.

```bash
sudo apt update
sudo apt install python3-venv python3-pip python3-gpiozero python3-lgpio \
  python3-picamera2 python3-opencv python3-numpy python3-pygame
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install [Ollama for Linux](https://docs.ollama.com/linux) on the **Pi**, start its
service, and install the vision model there:

```bash
ollama pull moondream
ollama list
python sarah_pi.py --check
python sarah_pi.py --drive-only
```

`--check` validates configuration and reports import availability without opening
GPIO or moving anything. It does not test the camera, wiring, model or motors.
If GPIO access is denied, check your user's access to `/dev/gpiochip*` and the
`gpio` group. With older GPIO Zero installations, Pi 5 chip numbering may need
an explicit `gpiochip` value in the configuration; prefer updated packages and
automatic detection (`null`).

For the retained companion features:

```bash
sudo apt install python3-pyaudio flac mpg123
python -m pip install -r requirements-companion.txt
ollama pull llama3.2:1b
python sarah_pi.py
```

Avatar/eye expressions, voice commands, conversational chat, scene descriptions,
dance/music, the Dino game and battery mood remain in `sarah/companion.py`.
Voice recognition retains the previous Google SpeechRecognition path and needs
internet access; autonomous driving does not. Piper speech output still requires
your Piper executable and voice model. Existing audio/Piper environment settings
remain supported. Point `SARAH_DANCE_SONG` to your own MP3 (not supplied).
`SARAH_AVATAR=0` disables the display. Use `--drive-only` to troubleshoot driving
without loading microphone/chat/display features.

Optional chat uses `SARAH_CHAT_MODEL` (default `llama3.2:1b`),
`SARAH_CHAT_VISION_MODEL` (default `moondream`) and `SARAH_OLLAMA_URL`.
Those settings do not change autonomous AI's local-only endpoint.

## Controls

| Input | Behavior |
| --- | --- |
| Hold LB + D-pad or left stick | Remote driving; release LB or direction to stop |
| A | Latched emergency stop in every mode |
| Start, with controls neutral and LB released | Reset stop and select manual |
| X, with controls neutral | Explicitly select autonomous |
| Console `manual` | Reset stop and select manual |
| Console `autonomous` | Select autonomous (cannot clear a latched stop) |
| Console `stop` | Latched stop |
| Console `status` | Show mode, ranges, sensor health, motor status and AI status |
| Console `dance`, `game`, `chat` | Retained companion modes |
| Console `quit` or Ctrl+C | Stop motors and shut down |

After connecting or changing modes, release the stick/D-pad and LB once before
driving. Button indices are configurable because SDL mappings vary by controller.
The default Xbox mapping is A=0, X=2, LB=4, Start=7. Controller polling renews a
command while held; a stalled control thread cannot leave it running indefinitely.
Disconnect detection still depends on the OS/SDL reporting the radio disconnect.

Voice examples: “Sarah manual”, “Sarah autonomous”, “Sarah stop”, “Sarah dance”,
“Sarah activate game mode”, “Sarah chat”, “Sarah what do you see”, and
“Sarah forward for 1 second”. In chat mode the wake word is optional. Voice
movement explicitly selects manual and is capped at two seconds. Game/dance
motion uses the same obstacle checks, speed caps and watchdog. Ending a
performance returns to stopped manual mode.

## Configuration and first hardware test

`robot_config.json` is the single source of driving settings. It preserves the
working motor pinout and the confirmed sensor placement. See
[the wiring guide](Wiring/WIRING_GUIDE_WORKING.md) for the complete table.

Speed now actually controls PWM duty on the **direction inputs**; ENA/ENB jumpers
stay installed. Default caps are 55% manual and 40% autonomous. There is no hidden
100% startup kick. These are duty cycles, not calibrated vehicle speeds.

Before floor testing:

1. Raise the wheels. Verify the motor supply, shared grounds and level-shifted
   HC-SR04 echo connections against the wiring guide.
2. Run `rpicam-hello --list-cameras`, then start `--drive-only` in manual.
   Use `status` and move a flat target in front of each ultrasonic. The default
   CSI camera is index 0; use `camera.backend: "usb"` for a USB camera.
3. All three readings must be valid for autonomous mode. For manual, forward
   needs both front sensors, reverse needs the rear, and pivot turns need all
   three. No echo, stale readings and readings under 2 cm are never treated as
   open space. HC-SR04s have blind spots and cannot detect every surface.
4. With clear sensor readings and raised wheels, use short LB + D-pad inputs.
   Verify each wheel's physical direction and that A immediately stops it.
   Correct `motor_polarity` only for wheels wired/mounted differently.
5. Check releasing LB, controller disconnection and Ctrl+C. Reconnect and press
   Start with neutral controls before driving again.
6. Test autonomous in a clear, flat area while supervising. Check the `[AUTO]`
   output: it should show local AI choices and stop on faults. Cover/disconnect
   the camera or a sensor to verify the stop behavior. Measure stopping distance
   at your chosen duty cycles and increase clearance accordingly.

Do not lower stop thresholds to work around a wiring fault. If motors stall,
check supply/driver losses first, then calibrate duty within the configuration
limits. A software watchdog cannot stop motors after OS failure or power loss;
keep a physical motor-power cutoff accessible.

## Development checks and repository layout

```bash
python -m unittest discover -s tests -v
python sarah_pi.py --check
python "STL Files/sarah_pi.py" --check
python sarah_pi.py --simulate --drive-only --no-console --run-seconds 1
```

Tests use fake time, model replies, sensors and GPIO to cover mode races, AI
failures, stale frames, obstacle changes, motor polarity/PWM, controller behavior,
echo timing and emergency stops. Simulation never enables GPIO and deliberately
leaves range readings unknown. It is not a hardware calibration or model benchmark.
The GPIO mock tests require `gpiozero` on the development machine.

- `sarah_pi.py`, `STL Files/sarah_pi.py` and `Code/Use in Pi/sarah_pi.py`: supported launchers.
- `sarah/`: motion controller, hardware, onboard planner, controls, application,
  and preserved companion functions.
- `robot_config.json`: pins, camera, onboard AI and driving limits.
- `tests/`: hardware-free driving regression tests.
- `STL Files/`: original CAD/print models, preserved because distinct revisions
  may still be useful manufacturing sources.
- `Wiring/`: current guide and original hardware references.

Obsolete `Code/` iterations, avatar demos, old tests/startup instructions, cached
bytecode, generated demo audio and old navigation state were removed. The nested
`Code/.git` metadata and its staged index were preserved unchanged; Git may
therefore still show the previous files as staged until you review that repository. Identical
wiring references were consolidated into `Wiring/`; the license is at the root.
A recovery ZIP of the earlier scripts and notes was saved before cleanup at
`C:\Users\dusan\AppData\Local\Temp\sarah-pi-before-rebuild-20261117-124121.zip`.
Move it to permanent backup storage if you want to retain it beyond Windows' temp cleanup.

Implementation references: [GPIO Zero pin and edge APIs](https://gpiozero.readthedocs.io/en/stable/api_pins.html),
[GPIO Zero output APIs](https://gpiozero.readthedocs.io/en/latest/api_output.html),
[Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf),
[Ollama generation API](https://github.com/ollama/ollama/blob/main/docs/api.md),
and [Moondream's model page](https://ollama.com/library/moondream).
