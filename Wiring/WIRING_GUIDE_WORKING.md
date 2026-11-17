# SARAH Raspberry Pi 5 wiring

BCM numbers below match `robot_config.json`. The confirmed sensors face
**front-left, rear, front-right**. The old ?center? sensor is the rear sensor.

## Motors: two L298N boards, four motors

| Motor | IN pair (BCM) | Physical header pins | Forward input levels |
| --- | --- | --- | --- |
| Left front | 17 / 18 | 11 / 12 | LOW / PWM |
| Left rear | 15 / 27 | 10 / 13 | PWM / LOW |
| Right front | 12 / 19 | 32 / 35 | PWM / LOW |
| Right rear | 26 / 6 | 37 / 31 | LOW / PWM |

These polarities preserve the original working mirrored motor arrangement.
Reverse swaps the active input. A left pivot reverses the left side and drives
the right side forward; a right pivot does the opposite.

Keep **ENA and ENB jumpers installed** on both motor boards. Speed is now PWM on
the direction inputs at 100 Hz. The inactive input is LOW before the opposite
input is energized; STOP sets all eight inputs LOW. Do not connect a separate
GPIO enable wire to a jumpered ENA/ENB terminal.

Default speed caps are 55% manual and 40% autonomous, without a full-power kick.
Change motor polarity in `robot_config.json` only if physical wheel orientation
requires it. Test with wheels raised first. PWM duty is not a voltage regulator;
verify the battery and driver are suitable for the motors' ratings.

Power motor drivers from a suitable motor battery and the Pi from its own
appropriate USB-C supply. Connect the Pi, both driver grounds and sensor grounds
together. Do not power the Pi from an L298N module's 5 V regulator output. Check
your particular module's regulator jumper instructions before connecting power.

## HC-SR04 ultrasonic sensors

| Position | Trigger BCM (physical) | Echo BCM (physical) |
| --- | --- | --- |
| Front-left | 23 (16) | 24 (18) |
| Rear | 22 (15) | 25 (22) |
| Front-right | 9 (21) | 10 (19) |

HC-SR04 modules normally use 5 V power and emit a 5 V ECHO signal. **Do not connect
that ECHO directly to a Pi input.** Use an appropriate level shifter or a divider
on EACH echo line, for example 1 kohm from ECHO to the GPIO node and 2 kohm from
that node to ground (approximately 3.3 V). Verify the actual sensor/module type.
Trigger connects to the corresponding output and all grounds are shared.

Sensors are triggered sequentially at least 65 ms apart across the array.
Distance uses GPIO edge timestamps; a timeout, stuck-high echo or reading over
400 cm is unknown. A very short echo remains an obstacle. No smoothing averages
away a newly close obstacle.

GPIO9/10 overlap SPI functions and GPIO15 overlaps UART RX. Disable peripheral
functions that conflict with these assignments. GPIO ownership/pinmux conflicts
must be resolved; the app rejects duplicate pin assignments. The earlier claim
that GPIO5/16/20 cannot be driven LOW on all Pi 5 boards was incorrect and has
been removed.

## Camera and validation

Use the appropriate Pi 5 camera cable, connect it while powered down, and verify
CSI detection with `rpicam-hello --list-cameras`. Configuration defaults to
Picamera2 camera index 0 at 320x240. `RGB888` gives the BGR byte order expected by
OpenCV. A USB camera can be selected in `robot_config.json`.

Follow the raised-wheel and sensor tests in [README](../README.md). The preserved
images and L298N document in this directory are historical references; the table
above and `robot_config.json` describe the current software wiring.

References: [Raspberry Pi GPIO documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#gpio),
[GPIO Zero output devices](https://gpiozero.readthedocs.io/en/latest/api_output.html),
and [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).
