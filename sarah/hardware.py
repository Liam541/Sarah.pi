"""Raspberry Pi 5 adapters. Importing this module does not open hardware."""
import base64
import json
import math
from pathlib import Path
import threading
import time

from .motion import Sample, SafetyConfig


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"safety", "motor_pins", "motor_polarity", "ultrasonic_pins", "gpiochip", "camera", "ai", "controller"}
    if set(data) - allowed:
        raise ValueError(f"Unknown configuration keys: {set(data) - allowed}")
    config = SafetyConfig(**data.get("safety", {}))
    pins = data["motor_pins"]
    sensors = data["ultrasonic_pins"]
    if set(pins) != {"left_front", "left_rear", "right_front", "right_rear"}:
        raise ValueError("Configure all four motors")
    if set(sensors) != {"front_left", "rear", "front_right"}:
        raise ValueError("Configure front_left, rear, front_right sensors")
    all_pins = []
    for pair in [*pins.values(), *sensors.values()]:
        if not isinstance(pair, list) or len(pair) != 2 or any(type(p) is not int or not 0 <= p <= 27 for p in pair):
            raise ValueError("Each device needs two valid BCM pins")
        all_pins.extend(pair)
    if len(all_pins) != len(set(all_pins)):
        raise ValueError("GPIO pin assignments overlap")
    if set(data["motor_polarity"]) != set(pins) or any(type(v) is not int or v not in (-1, 1) for v in data["motor_polarity"].values()):
        raise ValueError("Each motor polarity must be +1 or -1")
    if data.get("gpiochip") is not None and (type(data["gpiochip"]) is not int or data["gpiochip"] < 0):
        raise ValueError("gpiochip must be null or a nonnegative chip number")
    camera = data.get("camera", {})
    if set(camera) - {"backend", "index", "size"}:
        raise ValueError("Unknown camera setting")
    if camera.get("backend", "picamera2") not in ("picamera2", "usb"):
        raise ValueError("camera.backend must be picamera2 or usb")
    size = camera.get("size", [320, 240])
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or not 64 <= v <= 1280 for v in size):
        raise ValueError("camera.size must contain two pixel dimensions in 64..1280")
    if type(camera.get("index", 0)) is not int or camera.get("index", 0) < 0:
        raise ValueError("camera.index must be a nonnegative integer")
    ai = data.get("ai", {})
    if set(ai) - {"url", "model", "timeout_s", "max_step_s"}:
        raise ValueError("Unknown AI setting")
    step = ai.get("max_step_s", 0.6)
    if type(step) not in (int, float) or not math.isfinite(step) or not 0.1 <= step <= 1:
        raise ValueError("ai.max_step_s must be 0.1..1")
    controller = data.get("controller", {})
    if set(controller) - {"stop_button", "manual_button", "auto_button", "deadman_button", "deadzone"}:
        raise ValueError("Unknown controller setting")
    buttons = [controller.get(key, default) for key, default in
               (("stop_button", 0), ("manual_button", 7), ("auto_button", 2), ("deadman_button", 4))]
    if any(type(v) is not int or not 0 <= v <= 31 for v in buttons) or len(set(buttons)) != len(buttons):
        raise ValueError("Controller buttons must be distinct indices in 0..31")
    zone = controller.get("deadzone", .18)
    if type(zone) not in (int, float) or not .05 <= zone <= .5:
        raise ValueError("controller.deadzone must be 0.05..0.5")
    return data, config


class MotorOutput:
    """PWM on the direction inputs; ENA/ENB jumpers stay installed.

    Forward polarity matches the original working four-motor wiring. Setting
    the inactive input low before enabling the other implements break-before-make.
    """
    def __init__(self, data, simulate=False):
        self.use_gpio = not simulate
        self.pin_factory = None
        self.devices = {}
        self.polarity = data["motor_polarity"]
        self.last = None
        if simulate:
            return
        from gpiozero import PWMOutputDevice
        from gpiozero.pins.lgpio import LGPIOFactory
        chip = data.get("gpiochip")
        self.pin_factory = LGPIOFactory() if chip is None else LGPIOFactory(chip=chip)
        try:
            for name, pair in data["motor_pins"].items():
                self.devices[name] = []
                for pin in pair:
                    self.devices[name].append(PWMOutputDevice(pin, frequency=100, initial_value=0, pin_factory=self.pin_factory))
        except Exception:
            self.close()
            raise

    def set_wheels(self, left, right):
        if self.last == (left, right):
            return
        errors = []
        for name, pair in self.devices.items():
            if len(pair) != 2:
                continue
            power = (left if name.startswith("left") else right) * self.polarity[name] / 100
            active, inactive = pair if power >= 0 else pair[::-1]
            try:
                inactive.value = 0
                active.value = abs(power)
            except Exception as exc:
                errors.append(exc)
        if errors:
            # Attempt EVERY output even if one device has failed.
            for pair in self.devices.values():
                for device in pair:
                    try:
                        device.value = 0
                    except Exception:
                        pass
            self.last = None
            raise RuntimeError("GPIO motor write failed; all outputs stopped where possible") from errors[0]
        self.last = (left, right)

    def close(self):
        errors = []
        for pair in self.devices.values():
            for device in pair:
                try:
                    device.value = 0
                except Exception as exc:
                    errors.append(exc)
                try:
                    device.close()
                except Exception as exc:
                    errors.append(exc)
        self.devices.clear()
        if self.pin_factory:
            self.pin_factory.close()
            self.pin_factory = None
        if errors:
            raise RuntimeError("Some GPIO outputs failed to close; all were attempted") from errors[0]


class EchoPulse:
    """Measure kernel-timestamped lgpio edges instead of busy-polling in Python."""
    def __init__(self, trigger, echo, pin_factory):
        self.trigger, self.echo, self.factory = trigger, echo, pin_factory
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.armed = False
        self.armed_at = 0.0
        self.rise = None
        self.duration = None
        echo.pin.edges = "both"
        # GPIO Zero's Pin API passes event ticks and level. LGPIO supplies the
        # kernel edge timestamp, so AI load / delayed Python callbacks don't
        # turn a short pulse into a falsely large obstacle distance.
        echo.pin.when_changed = self.edge

    def edge(self, ticks, level):
        with self.lock:
            if not self.armed or self.factory.ticks_diff(ticks, self.armed_at) < 0:
                return
            if level and self.rise is None:
                self.rise = ticks
            elif not level and self.rise is not None:
                self.duration = self.factory.ticks_diff(ticks, self.rise)
                self.armed = False
                self.done.set()

    def measure(self):
        if self.echo.value:
            return None
        with self.lock:
            self.done.clear()
            self.rise = self.duration = None
            self.armed_at = self.factory.ticks()
            self.armed = True
        try:
            self.trigger.on()
            time.sleep(0.00001)
            self.trigger.off()
            if not self.done.wait(0.04):
                return None
            with self.lock:
                cm = self.duration * 17150 if self.duration is not None else None
            # Preserve below-range readings as close obstacles.
            return cm if cm is not None and 0 <= cm <= 400 else None
        finally:
            with self.lock:
                self.armed = False
            self.trigger.off()


class RangeSensors(threading.Thread):
    """One ping at a time, >=65 ms apart across ALL sensors to reduce crosstalk."""
    def __init__(self, data, pin_factory, stop_event, simulate=False):
        super().__init__(name="ultrasonic", daemon=True)
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.samples = {name: Sample() for name in data["ultrasonic_pins"]}
        self.devices = {}
        self.pulses = {}
        self.simulate = simulate
        if not simulate:
            from gpiozero import DigitalInputDevice, DigitalOutputDevice
            try:
                for name, (trigger, echo) in data["ultrasonic_pins"].items():
                    self.devices[name] = []
                    self.devices[name].append(DigitalOutputDevice(trigger, initial_value=False, pin_factory=pin_factory))
                    self.devices[name].append(DigitalInputDevice(echo, pull_up=False, pin_factory=pin_factory))
                    self.pulses[name] = EchoPulse(*self.devices[name], pin_factory)
            except Exception:
                self.close()
                raise

    def snapshot(self):
        with self.lock:
            return dict(self.samples)

    def run(self):
        while not self.stop_event.is_set():
            for name in self.samples:
                if self.stop_event.is_set():
                    return
                at = time.monotonic()
                try:
                    cm = None if self.simulate else self.pulses[name].measure()
                except Exception:
                    cm = None
                with self.lock:
                    self.samples[name] = Sample(cm, at)
                self.stop_event.wait(max(0, 0.065 - (time.monotonic() - at)))

    def close(self):
        errors = []
        for pair in self.devices.values():
            for device in pair:
                try:
                    device.close()
                except Exception as exc:
                    errors.append(exc)
        self.devices.clear()
        if errors:
            raise RuntimeError("Some ultrasonic pins failed to close") from errors[0]


class CameraThread(threading.Thread):
    """A timestamp belongs to a successful capture, not to somebody reading it."""
    def __init__(self, stop_event, config=None):
        super().__init__(name="camera", daemon=True)
        self.stop_event = stop_event
        self.config = config or {}
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.frame_at = 0.0
        self.camera_enabled = False
        self.ready = threading.Event()
        self.error = "Camera not started"

    def snapshot(self):
        with self.frame_lock:
            return (None if self.latest_frame is None else self.latest_frame.copy(), self.frame_at)

    def get_latest_frame(self):
        return self.snapshot()[0]

    def wait_for_frame(self, timeout=5):
        return self.ready.wait(timeout)

    def get_frame_base64(self):
        frame, at = self.snapshot()
        if frame is None or time.monotonic() - at > 0.75:
            return None
        import cv2
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(encoded).decode("ascii") if ok else None

    def run(self):
        import cv2
        backend = self.config.get("backend", "picamera2")
        size = tuple(self.config.get("size", [320, 240]))
        index = self.config.get("index", 0)
        while not self.stop_event.is_set():
            camera = None
            try:
                if backend == "picamera2":
                    from picamera2 import Picamera2
                    camera = Picamera2(camera_num=index)
                    # Picamera2 RGB888 is BGR byte order, as expected by OpenCV.
                    camera.configure(camera.create_video_configuration(main={"size": size, "format": "RGB888"}, controls={"FrameRate": 15}, buffer_count=2))
                    camera.start()
                elif backend == "usb":
                    camera = cv2.VideoCapture(index)
                    if not camera.isOpened():
                        raise RuntimeError("USB camera did not open")
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    raise ValueError("camera.backend must be picamera2 or usb")
                while not self.stop_event.is_set():
                    # Timestamp before the blocking capture so slow reads go stale.
                    at = time.monotonic()
                    frame = camera.capture_array() if backend == "picamera2" else camera.read()[1]
                    if frame is None or frame.size == 0:
                        raise RuntimeError("Empty camera frame")
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_at = at
                        self.camera_enabled = True
                        self.error = ""
                    self.ready.set()
                    self.stop_event.wait(1 / 30)
            except Exception as exc:
                self.error = str(exc)
                print(f"[CAM] {exc}; retrying in 2 seconds")
            finally:
                self.camera_enabled = False
                with self.frame_lock:
                    self.latest_frame = None
                    self.frame_at = 0.0
                self.ready.clear()
                if camera is not None:
                    try:
                        if backend == "picamera2":
                            camera.stop()
                            camera.close()
                        else:
                            camera.release()
                    except Exception:
                        pass
            self.stop_event.wait(2)
