"""Controller input and compatibility adapters for SARAH's voice/dance/game code."""
import math
import threading
import time

from .motion import wheels_for


class DriveFacade:
    def __init__(self, motion, mode="manual"):
        self.motion = motion
        self.mode = mode
        self.owner = "voice" if mode == "manual" else mode
        self.use_gpio = motion.output.use_gpio

    def stop(self):
        token = self.motion.token()
        if token[0] == self.mode:
            self.motion.stop(token=token, owner=self.owner)


def execute_timed(drive, command, stop_event, token=None):
    """Renew a bounded movement, but never resurrect work from another mode."""
    motion = drive.motion
    token = token or motion.token()
    if token[0] != drive.mode:
        return
    action = str(command.get("command", "STOP")).upper()
    if action == "STOP":
        motion.stop(token=token)
        return
    if action not in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
        return
    duration = float(command.get("duration", 0))
    speed = float(command.get("speed", motion.config.manual_speed))
    if not math.isfinite(duration) or not math.isfinite(speed) or duration <= 0 or speed < 0:
        motion.stop(token=token, reason="Invalid timed command")
        return
    end = time.monotonic() + min(duration, 2.0)
    receipt = None
    try:
        while not stop_event.is_set() and time.monotonic() < end:
            accepted = motion.command(*wheels_for(action, speed), token, owner=drive.owner)
            if accepted is None:
                break
            receipt = accepted
            stop_event.wait(0.05)
    finally:
        if receipt is not None:
            motion.stop(token=token, receipt=receipt)


def stick_wheels(x, y, speed, deadzone=0.18):
    def normalize(value):
        if abs(value) <= deadzone:
            return 0.0
        return math.copysign((min(abs(value), 1) - deadzone) / (1 - deadzone), value)
    steer, forward = normalize(x), -normalize(y)
    left, right = forward + steer, forward - steer
    scale = max(1.0, abs(left), abs(right))
    return left / scale * speed, right / scale * speed


class ControllerThread(threading.Thread):
    def __init__(self, motion, stop_event, set_mode, connected, game_dpad, config=None, pygame_module=None):
        super().__init__(name="controller", daemon=True)
        self.motion = motion
        self.stop_event = stop_event
        self.set_mode = set_mode
        self.connected = connected
        self.game_dpad = game_dpad
        self.config = config or {}
        self.pygame = pygame_module
        self.joystick = None
        self._previous = set()
        self._neutral_required = True
        self._token = None
        self._was_driving = False

    def lost(self):
        if self.joystick is not None:
            self.motion.emergency_stop("Controller disconnected; Start to resume")
            self.set_mode("manual", source="controller_disconnect")
            try:
                self.joystick.quit()
            except Exception:
                pass
        self.joystick = None
        self.connected(False)
        self.game_dpad(0, 0)
        self._neutral_required = True
        self._previous = set()
        self._was_driving = False

    def process(self, buttons, x, y, hat=(0, 0)):
        """One polled input frame; also used by hardware-free integration tests."""
        pressed = buttons - self._previous
        self._previous = set(buttons)
        stop_button = self.config.get("stop_button", 0)
        manual_button = self.config.get("manual_button", 7)
        auto_button = self.config.get("auto_button", 2)
        deadman = self.config.get("deadman_button", 4)
        deadzone = float(self.config.get("deadzone", 0.18))
        neutral = hat == (0, 0) and abs(x) <= deadzone and abs(y) <= deadzone
        if stop_button in buttons:
            if not self.motion.estopped:
                self.motion.emergency_stop("Controller A: emergency stop")
                self.set_mode("manual", source="controller_stop")
            self._neutral_required = True
            return
        if manual_button in pressed and neutral and deadman not in buttons:
            self.motion.resume_manual()
            self.set_mode("manual", source="controller_start")
        elif auto_button in pressed and neutral and not self.motion.estopped:
            self.set_mode("autonomous", source="controller_x")
        token = self.motion.token()
        if token != self._token:
            self._token = token
            self._neutral_required = True
            self._was_driving = False
        if token[0] == "game":
            self.game_dpad(*hat)
            return
        if token[0] != "manual" or self.motion.estopped:
            return
        if self._neutral_required:
            if neutral and deadman not in buttons:
                self._neutral_required = False
            return
        if deadman not in buttons:
            if self._was_driving:
                self.motion.stop(token=token, reason="Deadman released", owner="controller")
            self._was_driving = False
            return
        speed = self.motion.config.manual_speed
        if hat != (0, 0):
            x, y = float(hat[0]), float(-hat[1])
        left, right = stick_wheels(x, y, speed, deadzone)
        self.motion.command(left, right, token, owner="controller")
        self._was_driving = bool(left or right)

    def run(self):
        pg = self.pygame
        if pg is None:
            self.connected(False)
            return
        try:
            # Leave the avatar's display/mixer alive; never call pygame.quit().
            pg.joystick.init()
            if not pg.display.get_init():
                pg.display.init()
            while not self.stop_event.is_set():
                try:
                    pg.event.pump()
                    if pg.joystick.get_count() == 0:
                        self.lost()
                        self.stop_event.wait(0.2)
                        continue
                    if self.joystick is None:
                        self.joystick = pg.joystick.Joystick(0)
                        self.joystick.init()
                        self._neutral_required = True
                        self.connected(True)
                        print(f"[CTRL] {self.joystick.get_name()}: hold LB + D-pad/stick; A stop; Start manual/reset; X autonomous")
                    # Verify this instance still exists even if another pad was attached.
                    ids = {pg.joystick.Joystick(i).get_instance_id() for i in range(pg.joystick.get_count())}
                    if self.joystick.get_instance_id() not in ids:
                        self.lost()
                        continue
                    buttons = {i for i in range(self.joystick.get_numbuttons()) if self.joystick.get_button(i)}
                    x = self.joystick.get_axis(0) if self.joystick.get_numaxes() > 0 else 0
                    y = self.joystick.get_axis(1) if self.joystick.get_numaxes() > 1 else 0
                    hat = self.joystick.get_hat(0) if self.joystick.get_numhats() else (0, 0)
                    if not self.joystick.get_numhats() and self.joystick.get_numaxes() > 7:
                        hx, hy = self.joystick.get_axis(6), self.joystick.get_axis(7)
                        hat = ((1 if hx > .5 else -1 if hx < -.5 else 0), (-1 if hy > .5 else 1 if hy < -.5 else 0))
                    self.process(buttons, x, y, hat)
                except Exception as exc:
                    self.lost()
                    print(f"[CTRL] {exc}; reconnecting")
                    self.stop_event.wait(0.5)
                self.stop_event.wait(0.02)
        finally:
            self.lost()
