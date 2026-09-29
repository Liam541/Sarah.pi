# SARAH Robot - Working Wiring Configuration
**Last Updated:** December 31, 2025  
**Status:** ✅ All 4 motors operational

---

## Hardware Overview
- **Platform:** Raspberry Pi 5
- **Motor Drivers:** 2× L298N Dual H-Bridge
- **Motors:** 4× TT DC Motors
- **Power:** 8V Battery (2× 18650 cells in series)

---

## GPIO Pin Assignments

### Left L298N Motor Driver
| Function | GPIO Pin | Physical Pin | Description |
|----------|----------|--------------|-------------|
| IN1      | GPIO17   | Pin 11       | Motor A (front-left) control |
| IN2      | GPIO18   | Pin 12       | Motor A (front-left) control |
| IN3      | GPIO15   | Pin 10       | Motor B (back-left) control |
| IN4      | GPIO27   | Pin 13       | Motor B (back-left) control |

### Right L298N Motor Driver
| Function | GPIO Pin | Physical Pin | Description |
|----------|----------|--------------|-------------|
| IN1      | GPIO12   | Pin 32       | Motor A (front-right) control |
| IN2      | GPIO19   | Pin 35       | Motor A (front-right) control |
| IN3      | GPIO26   | Pin 37       | Motor B (back-right) control |
| IN4      | GPIO6    | Pin 31       | Motor B (back-right) control |

---

## Motor Output Wiring

### Both L298N Drivers (Left & Right)
| Terminal | Wire Color | Motor Connection |
|----------|------------|------------------|
| OUT1     | **Red**    | Motor positive   |
| OUT2     | **Black**  | Motor negative   |
| OUT3     | **Black**  | Motor negative   |
| OUT4     | **Red**    | Motor positive   |

**Connection Pattern:**
- **Motor A** (Front): Red → OUT1, Black → OUT2
- **Motor B** (Back): Red → OUT4, Black → OUT3

---

## Power Connections

### L298N Power Inputs
| Terminal | Connection | Voltage | Notes |
|----------|------------|---------|-------|
| +12V     | Battery +  | 8V      | Main motor power |
| GND      | Battery -  | 0V      | Common ground |
| +5V      | NOT USED   | -       | Internal regulator disabled |

### Raspberry Pi Power
- **Power:** Separate USB-C power supply (5V/3A recommended)
- **DO NOT** power Pi from L298N +5V output

### Ground Connection
- Connect L298N GND to Raspberry Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)

---

## Enable Pins (ENA/ENB)

**Configuration:** Jumpers installed on both drivers

| Jumper | Position | Function |
|--------|----------|----------|
| ENA    | ✅ ON    | Motor A always enabled (full battery voltage) |
| ENB    | ✅ ON    | Motor B always enabled (full battery voltage) |

**Note:** With jumpers ON, motors run at full 8V battery power. No PWM speed control is used.

---

## Complete Wiring Summary

### Left L298N Driver
```
Raspberry Pi 5          L298N (Left)          Motors
─────────────────       ─────────────         ──────────
Pin 11 (GPIO17) ──────> IN1                   
Pin 12 (GPIO18) ──────> IN2                   OUT1 (Red) ──┐
Pin 10 (GPIO15) ──────> IN3                   OUT2 (Black)─┤─> Front-Left Motor
Pin 13 (GPIO27) ──────> IN4                   OUT3 (Black)─┤─> Back-Left Motor
Pin 6  (GND)    ──────> GND                   OUT4 (Red) ──┘
                        +12V <───── 8V Battery (+)
                        GND  <───── 8V Battery (-)
                        ENA  [Jumper ON]
                        ENB  [Jumper ON]
```

### Right L298N Driver
```
Raspberry Pi 5          L298N (Right)         Motors
─────────────────       ─────────────         ──────────
Pin 32 (GPIO12) ──────> IN1                   
Pin 35 (GPIO19) ──────> IN2                   OUT1 (Red) ──┐
Pin 37 (GPIO26) ──────> IN3                   OUT2 (Black)─┤─> Front-Right Motor
Pin 31 (GPIO6)  ──────> IN4                   OUT3 (Black)─┤─> Back-Right Motor
Pin 6  (GND)    ──────> GND                   OUT4 (Red) ──┘
                        +12V <───── 8V Battery (+)
                        GND  <───── 8V Battery (-)
                        ENA  [Jumper ON]
                        ENB  [Jumper ON]
```

---

## Important Notes

### ✅ Working Configuration
- All 4 motors operational and synchronized
- Right side uses GPIO12/19/26/6 (avoiding GPIO5/16/20 which had pull-up issues on Pi 5)
- Software handles right-side motor inversion for mirrored chassis mounting

### ⚠️ GPIO Pins to Avoid on Raspberry Pi 5
The following GPIO pins have hardware pull-up resistors and cannot be reliably driven LOW:
- **GPIO5** (I2C SDA)
- **GPIO16** (SPI CE2)
- **GPIO20** (PCM DIN)

These pins were initially used but caused the right side to malfunction. They have been replaced with GPIO12/19/26.

### 🔧 Troubleshooting
If motors don't run:
1. Verify ENA/ENB jumpers are installed
2. Check battery voltage (should be 7-8V under load)
3. Confirm ground connection between Pi and L298N
4. Run `python3 sarah_pi.py` and check for `[DRIVE-GPIO] ✓ All 4 motors set to FORWARD`
5. Use `gpio readall` or `pinctrl` to verify GPIO pin states

### 📋 Motor Direction Logic
- **Forward:** All wheels spin to move robot forward
- **Backward:** All wheels spin to move robot backward
- **Left Turn:** Right side forward, left side stopped
- **Right Turn:** Left side forward, right side stopped

Software automatically handles polarity inversion for mirrored right-side motors.

---

## Testing Commands

After wiring, test each direction:

```bash
python3 sarah_pi.py
# Say "Sarah" then "move forward"
# Say "Sarah" then "move backward"
# Say "Sarah" then "turn left"
# Say "Sarah" then "turn right"
```

Expected behavior:
- ✅ Forward: Robot moves forward
- ✅ Backward: Robot moves backward
- ✅ Left: Robot rotates left
- ✅ Right: Robot rotates right
- ✅ All 4 motors should run during forward/backward commands

---

## Configuration Files

This wiring is implemented in:
- **File:** `sarah_pi.py`
- **Class:** `Drive` (lines ~3311-3820)
- **Pin Definitions:** Lines ~3470-3478

No environment variables or external configuration files are used. All pin assignments are hardcoded in the Drive class.

---

**End of Wiring Guide**
