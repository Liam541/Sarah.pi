import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sarah.autonomy import AutonomousThread, LocalPlanner, parse_action, scene_signature
from sarah.controls import ControllerThread, DriveFacade, execute_timed, stick_wheels
from sarah.hardware import EchoPulse, RangeSensors, MotorOutput, CameraThread, load_config
from sarah.motion import MotionController, Sample, SafetyConfig, wheels_for

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Event:
    def __init__(self, clock):
        self.clock = clock
        self.stopped = False

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, duration):
        self.clock.now += duration
        return self.stopped


class Output:
    use_gpio = False

    def __init__(self):
        self.history = []

    def set_wheels(self, left, right):
        self.history.append((left, right))

    def close(self):
        pass


class MotionFixture:
    def setUp(self):
        self.clock = Clock()
        self.ranges = {n: Sample(100, self.clock()) for n in ("front_left", "rear", "front_right")}
        self.output = Output()
        self.motion = MotionController(self.output, lambda: self.ranges, clock=self.clock, start_watchdog=False)

    def command(self, action="FORWARD", speed=40):
        return self.motion.command(*wheels_for(action, speed), self.motion.token())


class MotionTests(MotionFixture, unittest.TestCase):
    def test_forward_requires_both_front_sensors(self):
        for name in ("front_left", "front_right"):
            for value in (None, float("nan"), float("inf"), -1, 401, 0, 1, 25):
                with self.subTest(name=name, value=value):
                    self.ranges[name] = Sample(value, self.clock())
                    self.assertIsNone(self.command())
                    self.assertEqual(self.motion.wheels, (0, 0))
            self.ranges[name] = Sample(100, self.clock())

    def test_directional_clearance_and_pivot(self):
        self.ranges["rear"] = Sample(5, self.clock())
        self.assertIsNotNone(self.command())
        self.assertIsNone(self.command("BACKWARD"))
        self.assertIsNone(self.command("LEFT"))
        self.ranges["rear"] = Sample(100, self.clock())
        self.ranges["front_left"] = Sample(5, self.clock())
        self.assertIsNotNone(self.command("BACKWARD"))
        self.assertIsNone(self.command("RIGHT"))

    def test_watchdog_stops_without_new_commands(self):
        self.command()
        self.clock.now += .26
        self.motion.check()
        self.assertEqual(self.motion.wheels, (0, 0))
        self.assertIn("heartbeat", self.motion.reason)

    def test_watchdog_rechecks_obstacles_independent_of_ai(self):
        self.command()
        self.ranges["front_right"] = Sample(3, self.clock())
        self.motion.check()
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_stale_or_future_sensor_does_not_authorize_motion(self):
        for stamp in (self.clock() - 1, self.clock() + 1):
            self.ranges["front_left"] = Sample(100, stamp)
            self.assertIsNone(self.command())

    def test_mode_transition_invalidates_old_result_even_after_returning(self):
        self.motion.set_mode("autonomous")
        old = self.motion.token()
        self.motion.set_mode("manual")
        self.motion.set_mode("autonomous")
        self.assertIsNone(self.motion.command(40, 40, old, camera_at=self.clock()))

    def test_stale_release_cannot_cancel_another_owner(self):
        first = self.command()
        self.command("BACKWARD")
        self.motion.stop(receipt=first)
        self.assertEqual(self.motion.wheels, (-40, -40))

    def test_stop_is_latched_across_modes(self):
        self.command()
        self.motion.emergency_stop()
        for mode in ("manual", "autonomous", "dance", "game"):
            self.motion.set_mode(mode)
            self.assertIsNone(self.command())
        self.motion.resume_manual()
        self.assertIsNotNone(self.command())

    def test_shutdown_never_allows_restart(self):
        self.motion.close()
        self.motion.resume_manual()
        self.assertIsNone(self.command())

    def test_speed_cap_and_nonfinite_commands(self):
        self.command(speed=100)
        self.assertEqual(self.motion.wheels, (55, 55))
        self.assertIsNone(self.motion.command(float("nan"), 1, self.motion.token()))
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_camera_freshness_is_required_only_for_autonomous(self):
        self.assertIsNotNone(self.command())
        self.motion.set_mode("autonomous")
        self.assertIsNone(self.command())
        self.assertIsNotNone(self.motion.command(40, 40, self.motion.token(), camera_at=self.clock()))
        self.clock.now += .8
        self.motion.check()
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_output_error_latches_emergency_stop(self):
        original = self.output.set_wheels
        self.output.set_wheels = lambda l, r: original(l, r) if l == r == 0 else (_ for _ in ()).throw(OSError("GPIO failed"))
        with self.assertRaises(OSError):
            self.command()
        self.assertTrue(self.motion.estopped)
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_chat_cannot_move(self):
        self.motion.set_mode("chat")
        self.assertIsNone(self.command())

    def test_range_provider_exception_stops_existing_motion(self):
        self.command()
        self.motion.ranges = Mock(side_effect=OSError("Sensor worker failed"))
        self.assertIsNone(self.command())
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_rc_preempts_voice_and_stale_voice_cannot_stop_rc(self):
        event = Event(self.clock)
        drive = DriveFacade(self.motion)
        def wait(duration):
            self.clock.now += duration
            self.motion.command(-40, -40, self.motion.token(), owner="controller")
            return False
        event.wait = wait
        with patch("sarah.controls.time.monotonic", self.clock):
            execute_timed(drive, {"command": "FORWARD", "duration": 1, "speed": 40}, event)
        self.assertEqual(self.motion.wheels, (-40, -40))
        self.assertEqual(self.motion.owner, "controller")

    def test_mode_switch_cancels_timed_dance_without_stopping_manual(self):
        self.motion.set_mode("dance")
        event = Event(self.clock)
        def wait(duration):
            self.clock.now += duration
            self.motion.set_mode("manual")
            self.motion.command(-40, -40, self.motion.token(), owner="controller")
            return False
        event.wait = wait
        with patch("sarah.controls.time.monotonic", self.clock):
            execute_timed(DriveFacade(self.motion, "dance"), {"command": "FORWARD", "duration": 1, "speed": 40}, event)
        self.assertEqual(self.motion.wheels, (-40, -40))

    def test_watchdog_thread_stops_when_command_producer_hangs(self):
        stopped = threading.Event()
        output = Output()
        original = output.set_wheels
        def write(left, right):
            original(left, right)
            if left == right == 0:
                stopped.set()
        output.set_wheels = write
        sensors = lambda: {n: Sample(100, time.monotonic()) for n in ("front_left", "rear", "front_right")}
        motion = MotionController(output, sensors)
        try:
            motion.command(40, 40, motion.token())
            stopped.clear()
            self.assertTrue(stopped.wait(0.8), "Independent watchdog did not stop an unrenewed command")
            self.assertEqual(motion.wheels, (0, 0))
        finally:
            motion.close()


class ControllerTests(MotionFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.controller = ControllerThread(self.motion, Event(self.clock),
                                           lambda mode, **kw: self.motion.set_mode(mode), Mock(), Mock())

    def test_deadman_and_neutral_after_mode_change(self):
        pad = self.controller
        pad.process({4}, 0, -1)  # held on connection
        self.assertEqual(self.motion.wheels, (0, 0))
        pad.process(set(), 0, 0)
        pad.process({4}, 0, -1)
        self.assertEqual(self.motion.wheels, (55, 55))
        pad.process(set(), 0, -1)
        self.assertEqual(self.motion.wheels, (0, 0))
        self.motion.set_mode("autonomous")
        self.motion.set_mode("manual")
        pad.process({4}, 0, -1)
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_held_direction_renews_heartbeat(self):
        self.controller.process(set(), 0, 0)
        for _ in range(15):
            self.clock.now += .02
            self.controller.process({4}, 0, -1)
            self.motion.check()
        self.assertEqual(self.motion.wheels, (55, 55))

    def test_a_stop_in_auto_and_x_cannot_clear_it(self):
        self.motion.set_mode("autonomous")
        self.controller.process({0}, 0, 0)
        self.controller.process({2}, 0, 0)
        self.assertTrue(self.motion.estopped)
        self.assertEqual(self.motion.mode, "manual")
        self.controller.process({7}, 0, 0)
        self.assertFalse(self.motion.estopped)

    def test_disconnect_stops_and_never_switches_to_auto(self):
        self.controller.joystick = Mock()
        self.command()
        self.controller.lost()
        self.assertEqual(self.motion.wheels, (0, 0))
        self.assertEqual(self.motion.mode, "manual")
        self.assertTrue(self.motion.estopped)

    def test_x_requires_neutral(self):
        self.controller.process({2}, 0, -1)
        self.assertEqual(self.motion.mode, "manual")
        self.controller.process(set(), 0, 0)
        self.controller.process({2}, 0, 0)
        self.assertEqual(self.motion.mode, "autonomous")

    def test_diagonal_and_stick_direction(self):
        left, right = stick_wheels(1, -1, 50)
        self.assertEqual((left, right), (50, 0))
        self.assertEqual(stick_wheels(0.1, -.1, 50), (0, 0))


class PlannerTests(unittest.TestCase):
    def test_config_is_valid(self):
        data, safety = load_config(ROOT / "robot_config.json")
        self.assertEqual(data["ultrasonic_pins"]["rear"], [22, 25])
        self.assertLess(safety.auto_speed, 100)

    def test_remote_or_redirectable_urls_are_rejected(self):
        for url in ("http://192.168.1.2:11434", "https://localhost:11434", "http://localhost:11434/path", "http://user@localhost:11434"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LocalPlanner({"url": url})

    def test_strict_json_and_action_validation(self):
        for text in ('hello', '[]', '{}', '{"action":"JUMP"}', '```json\n{"action":"FORWARD"}\n```', '{"action":"FORWARD"}junk'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_action(text)
        self.assertEqual(parse_action('{"action":"LEFT","reason":"open floor"}')[0], "LEFT")

    def test_local_model_receives_real_image_ranges_and_history(self):
        response = Mock(status_code=200)
        response.json.return_value = {"done": True, "response": '{"action":"LEFT"}'}
        post = Mock(return_value=response)
        planner = LocalPlanner({}, post=post)
        frame = np.full((24, 32, 3), 100, dtype=np.uint8)
        self.assertEqual(planner.decide(frame, {"rear": Sample(80, 1)}, ["LEFT", "STOP"], ["FORWARD"])[0], "LEFT")
        payload = post.call_args.kwargs["json"]
        self.assertTrue(payload["images"][0])
        self.assertIn('"rear": 80', payload["prompt"])
        self.assertIn('"FORWARD"', payload["prompt"])
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        response.json.return_value = {"done": True, "response": '{"action":"BACKWARD"}'}
        with self.assertRaises(ValueError):
            planner.decide(frame, {}, ["STOP"], [])
        response.json.return_value = {"done": False, "response": '{"action":"STOP"}'}
        with self.assertRaises(ValueError):
            planner.decide(frame, {}, ["STOP"], [])

    def test_dark_camera_rejected(self):
        with self.assertRaises(ValueError):
            scene_signature(np.zeros((24, 32, 3), dtype=np.uint8))

    def test_bad_safety_settings_rejected(self):
        for kwargs in ({"stop_cm": 1}, {"manual_speed": 100}, {"command_lease_s": 5}, {"stop_cm": float("nan")}, {"auto_speed": True}):
            with self.assertRaises(ValueError):
                SafetyConfig(**kwargs)


class AutonomyTests(MotionFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.event = Event(self.clock)
        self.camera = Mock()
        self.frame = np.tile(np.arange(64, dtype=np.uint8) * 3, (48, 1))
        self.frame = np.repeat(self.frame[:, :, None], 3, axis=2)
        self.camera.snapshot.side_effect = lambda: (self.frame, self.clock())
        # Keep synthetic sensor samples fresh throughout each maneuver.
        self.motion.ranges = lambda: {n: Sample(s.cm, self.clock()) for n, s in self.ranges.items()}
        self.planner = Mock(timeout=45)
        self.planner.decide.return_value = ("FORWARD", "Open floor")
        self.auto = AutonomousThread(self.motion, self.event, self.camera, {"max_step_s": .3}, self.planner)
        self.motion.set_mode("autonomous")
        self.patch = patch("sarah.autonomy.time.monotonic", self.clock)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_model_choice_drives_bounded_step(self):
        self.planner.decide.return_value = ("LEFT", "Left is clear")
        self.auto.run_step()
        self.assertTrue(any(l < 0 < r for l, r in self.output.history))
        self.assertEqual(self.motion.wheels, (0, 0))
        self.assertLess(self.clock() - 100, .5)

    def test_ai_failure_does_not_move(self):
        self.planner.decide.side_effect = TimeoutError("Local model unavailable")
        with self.assertRaises(TimeoutError):
            self.auto.run_step()
        self.assertFalse(any(l or r for l, r in self.output.history))

    def test_ai_reply_after_mode_switch_is_discarded(self):
        def decide(*args):
            self.motion.set_mode("manual")
            self.motion.set_mode("autonomous")
            return "FORWARD", "Late answer"
        self.planner.decide.side_effect = decide
        self.auto.run_step()
        self.assertFalse(any(l or r for l, r in self.output.history))

    def test_obstacle_appearing_during_inference_vetoes_reply(self):
        def decide(*args):
            self.ranges["front_left"] = Sample(5, self.clock())
            return "FORWARD", "Old view"
        self.planner.decide.side_effect = decide
        with self.assertRaises(ValueError):
            self.auto.run_step()
        self.assertFalse(any(l or r for l, r in self.output.history))

    def test_unknown_sensor_prevents_ai_and_motion(self):
        self.ranges["rear"] = Sample()
        with self.assertRaises(ValueError):
            self.auto.run_step()
        self.planner.decide.assert_not_called()

    def test_frozen_camera_prevents_ai_and_motion(self):
        self.camera.snapshot.side_effect = lambda: (self.frame, self.clock() - 2)
        with self.assertRaises(ValueError):
            self.auto.run_step()
        self.planner.decide.assert_not_called()

    def test_repeated_avoidance_requires_operator_reset(self):
        self.planner.decide.return_value = ("LEFT", "Avoid")
        for _ in range(8):
            self.auto.run_step()
        self.assertEqual(self.planner.decide.call_count, 6)
        self.assertEqual(self.motion.wheels, (0, 0))

    def test_new_obstacle_during_movement_stops_before_end(self):
        def wait(duration):
            self.clock.now += duration
            self.ranges["front_right"] = Sample(5, self.clock())
            return False
        self.event.wait = wait
        self.auto.run_step()
        self.assertEqual(self.motion.wheels, (0, 0))
        self.assertLess(self.clock() - 100, .2)

    def test_frame_failure_during_movement_stops_in_finally(self):
        observations = 0
        def snapshot():
            nonlocal observations
            observations += 1
            return (self.frame if observations <= 3 else None, self.clock())
        self.camera.snapshot.side_effect = snapshot
        with self.assertRaises(ValueError):
            self.auto.run_step()
        self.assertTrue(any(l or r for l, r in self.output.history))
        self.assertEqual(self.motion.wheels, (0, 0))


class HardwareTests(unittest.TestCase):
    def echo(self, high=False):
        factory = Mock()
        factory.ticks.return_value = 100
        factory.ticks_diff.side_effect = lambda newer, older: newer - older
        return EchoPulse(Mock(), Mock(value=high), factory)

    def test_echo_stuck_high_is_unknown_not_clear(self):
        pulse = self.echo(high=True)
        self.assertIsNone(pulse.measure())
        pulse.trigger.on.assert_not_called()

    def test_echo_never_arrives_is_unknown(self):
        pulse = self.echo()
        pulse.done = Mock()
        pulse.done.wait.return_value = False
        self.assertIsNone(pulse.measure())
        self.assertFalse(pulse.armed)
        pulse.trigger.off.assert_called()

    def test_echo_uses_edge_timestamps_not_callback_arrival_time(self):
        pulse = self.echo()
        def trigger():
            pulse.edge(99, 1)  # stale edge ignored
            pulse.edge(100.001, 1)
            pulse.edge(100.002, 0)
        pulse.trigger.on.side_effect = trigger
        self.assertAlmostEqual(pulse.measure(), 17.15)
        self.assertFalse(pulse.armed)

    def test_out_of_range_pulse_is_unknown(self):
        pulse = self.echo()
        pulse.trigger.on.side_effect = lambda: (pulse.edge(100.001, 1), pulse.edge(100.031, 0))
        self.assertIsNone(pulse.measure())

    def test_snapshot_does_not_make_an_old_camera_frame_new(self):
        camera = CameraThread(threading.Event())
        camera.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        camera.frame_at = 123
        frame, stamp = camera.snapshot()
        frame[:] = 255
        self.assertEqual(camera.snapshot()[1], 123)
        self.assertEqual(camera.latest_frame.sum(), 0)

    def test_pwm_polarity_duty_cycle_and_stop_with_mock_gpio(self):
        import sys
        import types
        from gpiozero.pins.mock import MockFactory, MockPWMPin
        factory = MockFactory(pin_class=MockPWMPin)
        fake_lgpio = types.ModuleType("gpiozero.pins.lgpio")
        fake_lgpio.LGPIOFactory = lambda: factory
        data, _ = load_config(ROOT / "robot_config.json")
        with patch.dict(sys.modules, {"gpiozero.pins.lgpio": fake_lgpio}):
            output = MotorOutput(data)
        try:
            output.set_wheels(40, 40)
            for name, pair in output.devices.items():
                expected = [0.4, 0] if data["motor_polarity"][name] == 1 else [0, 0.4]
                self.assertEqual([device.value for device in pair], expected)
            output.set_wheels(-30, -30)
            for name, pair in output.devices.items():
                expected = [0, .3] if data["motor_polarity"][name] == 1 else [.3, 0]
                self.assertEqual([device.value for device in pair], expected)
            output.set_wheels(0, 0)
            self.assertTrue(all(device.value == 0 for pair in output.devices.values() for device in pair))
        finally:
            output.close()


if __name__ == "__main__":
    unittest.main()
