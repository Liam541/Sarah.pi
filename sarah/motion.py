"""Single motor owner, expiring commands, and direction-aware obstacle checks.

No camera, network, GPIO, or speech dependencies: these rules are testable on a PC.
All timestamps use the same monotonic clock. Unknown range is never free space.
"""
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class Sample:
    cm: float | None = None
    at: float = 0.0

    def fresh(self, now, max_age):
        return (self.cm is not None and math.isfinite(self.cm)
                and 0 <= self.cm <= 400 and 0 <= now - self.at <= max_age)


@dataclass(frozen=True)
class SafetyConfig:
    stop_cm: float = 25.0
    turn_cm: float = 22.0
    slow_cm: float = 70.0
    range_max_age_s: float = 0.5
    camera_max_age_s: float = 0.75
    command_lease_s: float = 0.25
    manual_speed: float = 55.0
    auto_speed: float = 40.0
    turn_speed: float = 40.0
    reverse_speed: float = 35.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid safety setting {name}: {value!r}")
        if not 15 <= self.stop_cm < self.slow_cm <= 200:
            raise ValueError("Require 15 <= stop_cm < slow_cm <= 200")
        if not 15 <= self.turn_cm <= self.slow_cm:
            raise ValueError("Invalid turn clearance")
        if not 0.1 <= self.command_lease_s <= 0.5:
            raise ValueError("command_lease_s must be 0.1..0.5")
        if not 0.2 <= self.range_max_age_s <= 1.0 or not 0.1 <= self.camera_max_age_s <= 2.0:
            raise ValueError("Sensor freshness limits are outside supported bounds")
        if any(getattr(self, n) > 80 for n in ("manual_speed", "auto_speed", "turn_speed", "reverse_speed")):
            raise ValueError("Driving speed caps must be <= 80 percent")


def clearance_reason(left, right, ranges, now, config):
    """Check both front beams for forward motion, rear for reverse, all for pivots."""
    if left == right == 0:
        return ""
    pivot = left * right < 0
    required = ("front_left", "rear", "front_right") if pivot else (
        ("front_left", "front_right") if max(left, right) > 0 else ("rear",))
    threshold = config.turn_cm if pivot else config.stop_cm
    for name in required:
        sample = ranges.get(name, Sample())
        if not sample.fresh(now, config.range_max_age_s):
            return f"{name}: missing/stale range"
        if sample.cm <= threshold:
            return f"{name}: obstacle at {sample.cm:.1f} cm"
    return ""


class MotionController:
    """Serializes all outputs and stops independently of the AI/control threads.

    A token represents a mode generation. Returning to the same mode does not
    revive a worker or AI result that belonged to an earlier generation.
    """
    MODES = {"manual", "autonomous", "dance", "game", "chat"}

    def __init__(self, output, ranges, config=None, clock=time.monotonic, start_watchdog=True):
        self.output = output
        self.ranges = ranges
        self.config = config or SafetyConfig()
        self.clock = clock
        self.lock = threading.RLock()
        self.mode = "manual"
        self.generation = 0
        self.serial = 0
        self.owner = None
        self.estopped = False
        self.closed = False
        self.wheels = (0.0, 0.0)
        self.deadline = 0.0
        self.camera_at = None
        self.reason = "Waiting for operator"
        self._done = threading.Event()
        self._thread = None
        self.output.set_wheels(0, 0)
        if start_watchdog:
            self._thread = threading.Thread(target=self._watch, name="motor-watchdog", daemon=True)
            self._thread.start()

    def token(self):
        with self.lock:
            return (self.mode, self.generation)

    def set_mode(self, mode):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode}")
        with self.lock:
            # An explicit mode selection invalidates all outstanding work.
            self._stop_locked(f"Mode: {mode}")
            self.mode = mode
            self.generation += 1

    def emergency_stop(self, reason="Emergency stop"):
        with self.lock:
            self.estopped = True
            self.generation += 1
            self._stop_locked(reason)

    def resume_manual(self):
        with self.lock:
            if self.closed:
                return
            self.set_mode("manual")
            self.estopped = False

    def _stop_locked(self, reason):
        self.serial += 1
        self.deadline = 0.0
        self.camera_at = None
        self.owner = None
        self.reason = reason
        self.wheels = (0.0, 0.0)
        self.output.set_wheels(0, 0)

    def stop(self, token=None, receipt=None, reason="Stopped", owner=None):
        with self.lock:
            if token is not None and token != self.token():
                return
            if receipt is not None and receipt != self.serial:
                return
            if owner is not None and self.owner != owner:
                return
            self._stop_locked(reason)

    def command(self, left, right, token, camera_at=None, owner=None):
        with self.lock:
            if self.closed or self.estopped or token != self.token() or self.mode == "chat":
                return None
            now = self.clock()
            owner = owner or self.mode
            # RC can take over from a voice movement. A still-running voice
            # worker cannot then overwrite the controller on its next renewal.
            if self.owner == "controller" and owner != "controller" and now < self.deadline:
                return None
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (left, right)):
                self._stop_locked("Invalid motor command")
                return None
            cap = self.config.auto_speed if self.mode == "autonomous" else self.config.manual_speed
            left, right = (max(-cap, min(cap, v)) for v in (left, right))
            try:
                reason = clearance_reason(left, right, self.ranges(), now, self.config)
            except Exception:
                self._stop_locked("Range provider failed")
                return None
            if self.mode == "autonomous" and (left or right):
                if camera_at is None or not 0 <= now - camera_at <= self.config.camera_max_age_s:
                    reason = "Camera missing/stale"
            if reason:
                self._stop_locked(reason)
                return None
            try:
                self.output.set_wheels(left, right)
            except Exception:
                self.estopped = True
                self._stop_locked("Motor output failed")
                raise
            self.serial += 1
            self.wheels = (left, right)
            self.owner = owner
            self.camera_at = camera_at
            self.deadline = now + self.config.command_lease_s
            self.reason = "Moving" if left or right else "Stopped"
            return self.serial

    def check(self):
        with self.lock:
            if self.wheels == (0, 0):
                return
            now = self.clock()
            reason = clearance_reason(*self.wheels, self.ranges(), now, self.config)
            if now >= self.deadline:
                reason = "Command heartbeat expired"
            if self.mode == "autonomous" and (self.camera_at is None or now - self.camera_at > self.config.camera_max_age_s):
                reason = "Camera heartbeat expired"
            if self.estopped or self.closed or reason:
                self._stop_locked(reason or "Disabled")

    def _watch(self):
        while not self._done.wait(0.02):
            try:
                self.check()
            except Exception as exc:
                # A provider or output error is also a stop condition.
                try:
                    self.emergency_stop(f"Watchdog error: {exc}")
                except Exception:
                    pass

    def close(self):
        with self.lock:
            self.closed = True
            self.estopped = True
            self.generation += 1
            self._done.set()
            try:
                self._stop_locked("Shutdown")
            finally:
                self.output.close()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)


def wheels_for(action, speed):
    return {"FORWARD": (speed, speed), "BACKWARD": (-speed, -speed),
            "LEFT": (-speed, speed), "RIGHT": (speed, -speed), "STOP": (0, 0)}[action]
