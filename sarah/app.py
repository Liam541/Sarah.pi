"""Application lifecycle and explicit mode selection."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import signal
import threading
import time

from .autonomy import AutonomousThread, LocalPlanner
from .controls import ControllerThread, DriveFacade
from .hardware import CameraThread, MotorOutput, RangeSensors, load_config
from .motion import MotionController

ROOT = Path(__file__).resolve().parent.parent


class Application:
    def __init__(self, data, safety, simulate=False):
        # Validate the planner BEFORE opening any hardware.
        self.planner = LocalPlanner(data.get("ai", {}))
        self.data = data
        self.stop_event = threading.Event()
        self.companion = None
        self._mode_lock = threading.RLock()
        self.threads = []
        self.controller_connected = False
        self.output = MotorOutput(data, simulate=simulate)
        self.sensors = None
        self.motion = None
        try:
            self.sensors = RangeSensors(data, self.output.pin_factory, self.stop_event, simulate=simulate)
            self.motion = MotionController(self.output, self.sensors.snapshot, safety)
            self.camera = CameraThread(self.stop_event, data.get("camera"))
            self.autonomous = AutonomousThread(self.motion, self.stop_event, self.camera, data.get("ai", {}), self.planner)
        except Exception:
            self.close()
            raise

    def mode(self, name, source="console"):
        with self._mode_lock:
            if self.companion:
                self.companion.set_current_mode(name, source=source)
            else:
                self.motion.set_mode(name)
                print(f"[MODE] {name.upper()} ({source})")

    def connected(self, value):
        self.controller_connected = value
        if self.companion:
            self.companion.set_controller_connected(value)

    def game_dpad(self, x, y):
        if self.companion:
            self.companion.set_game_dpad(x, y)

    def start_companion(self):
        """Optional audio/chat initialization never delays driving or the watchdog."""
        try:
            from . import companion as features
            if self.stop_event.is_set():
                return
            with self._mode_lock:
                features._MOTION = self.motion
                features._APP_STOP = self.stop_event
                features.CURRENT_MODE = self.motion.token()[0]
                features.set_controller_connected(self.controller_connected)
                self.companion = features
            if features.AVATAR_ENABLED and features.pygame:
                features._AVATAR = features.AvatarDisplay()
                features._AVATAR.start()
            for cls, mode in ((features.DanceThread, "dance"), (features.GameMovementThread, "game")):
                if self.stop_event.is_set():
                    return
                worker = cls(DriveFacade(self.motion, mode), self.stop_event)
                self.threads.append(worker)
                worker.start()
            if self.stop_event.is_set():
                return
            voice = features.VoiceListenerThread(DriveFacade(self.motion), self.stop_event, self.camera)
            self.threads.append(voice)
            voice.start()
            print("[FEATURES] Avatar, voice/chat, dance, game and battery monitoring available")
        except Exception as exc:
            print(f"[FEATURES] Companion startup failed: {exc}. Driving and console remain available.")

    def start(self, companion=True):
        try:
            import pygame
        except ImportError:
            pygame = None
            print("[CTRL] pygame missing: console mode selection remains available")
        controller = ControllerThread(self.motion, self.stop_event, self.mode, self.connected,
                                      self.game_dpad, self.data.get("controller"), pygame)
        self.threads.extend([self.sensors, self.camera, self.autonomous, controller])
        for thread in self.threads:
            thread.start()
        if companion:
            worker = threading.Thread(target=self.start_companion, name="companion-startup", daemon=True)
            self.threads.append(worker)
            worker.start()

    def status(self):
        now = time.monotonic()
        return {
            "mode": self.motion.token()[0], "emergency_stop": self.motion.estopped,
            "motors": self.motion.wheels, "motion_status": self.motion.reason,
            "controller": self.controller_connected,
            "ranges": {name: {"cm": sample.cm, "fresh": sample.fresh(now, self.motion.config.range_max_age_s)}
                       for name, sample in self.sensors.snapshot().items()},
            "camera": self.camera.error or "capturing", "ai": self.autonomous.status,
            "ai_model": self.planner.model, "ai_url": self.planner.url,
        }

    def console_command(self, command):
        command = command.strip().lower()
        if command in ("quit", "exit"):
            self.motion.emergency_stop("Quitting")
            self.stop_event.set()
        elif command == "stop":
            self.motion.emergency_stop("Console emergency stop")
            self.mode("manual")
        elif command == "manual":
            self.motion.resume_manual()
            self.mode("manual")
        elif command in ("auto", "autonomous", "dance", "game", "chat"):
            if self.motion.estopped:
                print("[SYS] Emergency stop active. Select manual to reset first.")
            elif command in ("dance", "game", "chat") and not self.companion:
                print("[SYS] Companion features are unavailable or still starting.")
            else:
                self.mode("autonomous" if command == "auto" else command)
        elif command == "status":
            print(json.dumps(self.status(), indent=2))
        else:
            print("Commands: manual, autonomous, stop, status, dance, game, chat, quit")

    def console(self):
        while not self.stop_event.is_set():
            try:
                command = input()
            except (EOFError, OSError):
                return
            self.console_command(command)

    def battery_tick(self):
        if not self.companion:
            return
        percent = self.companion._read_battery_percent()
        level = self.companion._battery_level_from_percent(percent)
        emotion = {"critical": "exhausted", "low": "tired"}.get(level)
        self.companion.set_avatar_battery_mood(emotion)
        if level == "critical":
            self.motion.emergency_stop("Battery critical")

    def close(self):
        self.stop_event.set()
        # Stop and permanently disable motor writes BEFORE waiting on workers.
        if self.motion:
            try:
                self.motion.emergency_stop("Shutdown")
            except Exception as exc:
                print(f"[SYS] Motor stop error: {exc}")
        if self.companion and self.companion._AVATAR:
            try:
                self.companion._AVATAR.stop()
            except Exception as exc:
                print(f"[SYS] Avatar shutdown error: {exc}")
        for thread in list(self.threads):
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=0.3)
        try:
            if self.sensors:
                self.sensors.close()
        finally:
            if self.motion:
                self.motion.close()
            else:
                self.output.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="SARAH Raspberry Pi 5: remote control and onboard AI navigation")
    parser.add_argument("--config", type=Path, default=ROOT / "robot_config.json")
    parser.add_argument("--mode", choices=("manual", "autonomous"), default="manual")
    parser.add_argument("--drive-only", action="store_true", help="Do not load optional voice/avatar/chat features")
    parser.add_argument("--simulate", action="store_true", help="No GPIO outputs; range readings remain unknown")
    parser.add_argument("--check", action="store_true", help="Validate configuration/dependencies without opening hardware")
    parser.add_argument("--no-console", action="store_true")
    parser.add_argument("--run-seconds", type=float, default=0, help="Optional bounded smoke-test runtime")
    args = parser.parse_args(argv)
    data, safety = load_config(args.config)
    LocalPlanner(data.get("ai", {}))
    if args.check:
        modules = ("cv2", "numpy", "requests", "pygame", "gpiozero", "lgpio", "picamera2")
        print(json.dumps({"configuration": "valid", "platform": platform.platform(),
                          "dependencies": {name: importlib.util.find_spec(name) is not None for name in modules}}, indent=2))
        return 0
    if not args.simulate and (platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64", "armv7l")):
        parser.error("Real GPIO driving requires Raspberry Pi Linux; use --simulate on a PC")
    app = Application(data, safety, simulate=args.simulate)
    old_handlers = {}
    try:
        def shutdown(signum, frame):
            app.motion.emergency_stop(f"Signal {signum}")
            app.stop_event.set()
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, shutdown)
        app.start(companion=not args.drive_only)
        app.mode(args.mode, source="startup")
        print("[SYS] SARAH ready. A=stop, Start=manual/reset, X=autonomous; hold LB to drive.")
        print("[SYS] Console: manual, autonomous, stop, status, dance, game, chat, quit")
        if not args.no_console:
            thread = threading.Thread(target=app.console, name="console", daemon=True)
            app.threads.append(thread)
            thread.start()
        started = time.monotonic()
        next_battery = started
        while not app.stop_event.wait(0.1):
            now = time.monotonic()
            if args.run_seconds > 0 and now - started >= args.run_seconds:
                break
            if now >= next_battery:
                app.battery_tick()
                next_battery = now + 30
    finally:
        app.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    print("[SYS] Shutdown complete; motors disabled.")
    return 0
