#!/usr/bin/env python3
"""
sarah_autonomous_pi.py

Raspberry Pi 5 autonomous-capable SARAH robot.

Behavior summary:
- Uses the same Option A pinout and PWM speeds as before.
- If an Xbox controller is detected (pygame joystick), controller mode is active and controller D-Pad controls robot.
- If no controller is detected, the robot switches to AUTONOMOUS mode where a local Llama 3.1 (Ollama) model is used to generate driving commands periodically.
- Voice activation word: "Sarah" (case-insensitive).
- Dance mode activation phrase: "Sarah dance".
- Dance song MP3 is expected to be in the SAME FOLDER as this script on the Pi (e.g., your Documents folder if that's where you place `sarah_pi.py`).
- When the assistant hears a phrase containing "Sarah" it will forward the phrase to Llama 3.1 and execute the movement command the model returns (unless it's a simple local command like mode switching).
- The model is instructed to output a strict JSON command format so commands are machine-parseable.
- Emergency stop available via KeyboardInterrupt or controller A button when controller is present.

Safety: test with wheels raised or disconnected until you verify behavior.
"""

import cv2
import base64
import threading
import time
import signal
import sys
import re
import requests
import wave
import io
import json
import hashlib
import os
import queue
import gc  # Garbage collection for memory management
import subprocess  # For Piper TTS
from tempfile import gettempdir
import platform
from contextlib import contextmanager
from datetime import datetime
import glob
import logging
import shutil
from typing import Optional
import math
import random
from collections import deque

# Human-readable build stamp to help confirm which script copy is running on the Pi.
SARAH_BUILD = "2025-12-27.4"

# Suppress Jack audio daemon warnings on Raspberry Pi
logging.getLogger('pulsectl').setLevel(logging.ERROR)
os.environ['JACK_NO_AUDIO_SYNC'] = '1'
if platform.system() == 'Linux':
    # Suppress Jack audio messages - they're harmless
    os.environ['PULSE_PROP_media.role'] = 'multimedia'
    # Prevent PortAudio/Pulse from trying to auto-start jackd
    os.environ.setdefault('JACK_NO_START_SERVER', '1')
    os.environ.setdefault('JACK_NO_AUDIO_RESERVATION', '1')

try:
    import ollama
except ImportError:
    print("[FATAL] 'ollama' module not found. Install it with:")
    print("        pip install ollama")
    sys.exit(1)

import speech_recognition as sr


@contextmanager
def _silence_stderr_fd(enabled: bool):
    """Silence noisy C-library stderr output (ALSA/JACK/PortAudio) on Linux.

    PortAudio and JACK often print directly to file descriptor 2, which Python's
    logging/redirect_stderr cannot reliably capture. We only silence when
    enabled and not in debug mode.
    """
    if not enabled:
        yield
        return

    try:
        if platform.system() != "Linux":
            yield
            return
    except Exception:
        yield
        return

    try:
        old_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)
    except Exception:
        # If anything goes wrong, don't block audio; just proceed.
        yield

# Memory management (for 4GB RPi5 constraint)
try:
    import psutil  # type: ignore
    MEMORY_MONITORING_ENABLED = True
except ImportError:
    print("[WARN] psutil not installed for memory monitoring. Install with: pip install psutil")
    psutil = None  # type: ignore
    MEMORY_MONITORING_ENABLED = False

import numpy as np

# Detect if running on Raspberry Pi FIRST - needed for other imports
# Check for both 'arm' and 'aarch64' (Raspberry Pi 5 uses aarch64)
IS_RASPBERRY_PI = platform.system() == "Linux" and ("arm" in platform.machine().lower() or "aarch64" in platform.machine().lower())
PLATFORM_NAME = "Raspberry Pi 5" if IS_RASPBERRY_PI else f"{platform.system()} {platform.machine()}"

if IS_RASPBERRY_PI:
    print("[PLATFORM] [OK] Raspberry Pi detected - GPIO control enabled")
else:
    print(f"[PLATFORM] [INFO] Running on {PLATFORM_NAME} - GPIO control disabled (testing/Windows mode)")

# Suppress ALSA and Jack audio warnings on Linux (they're harmless)
try:
    from ctypes import *
    if platform.system() == "Linux":
        # Suppress ALSA lib warnings
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        try:
            asound = cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
        except (OSError, AttributeError):
            pass
except Exception:
    pass

# Optional: picamera2 for Raspberry Pi ribbon cable camera
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
    print("[INIT] picamera2 available - CSI ribbon cable camera support enabled")
except ImportError:
    PICAMERA2_AVAILABLE = False
    if IS_RASPBERRY_PI:
        print("[WARN] picamera2 not available on Raspberry Pi")
        # On Raspberry Pi OS, picamera2 is typically installed via apt.
        print("[WARN]   Install (recommended): sudo apt-get install -y python3-picamera2")
        print("[WARN]   If you must use venv/pip: pip install picamera2 (may fail on some Pi OS builds)")
        print("[WARN]   Falling back to rpicam-jpeg...")


# Piper TTS - Neural text-to-speech (offline, high quality)
try:
    import subprocess
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("[WARN] Piper TTS will be used if installed. Install with: pip install piper-tts")

# Controller (optional on RPi5)
try:
    import pygame
except Exception as e:
    pygame = None
    print("[WARN] pygame not available; controller disabled.", e)


# ==================== PI DISPLAY AVATAR (OPTIONAL) ====================
# This is intentionally enabled only on Raspberry Pi so it will NOT open a
# window on a Windows development machine.
AVATAR_ENABLED = IS_RASPBERRY_PI and (os.getenv("SARAH_AVATAR", "1").strip().lower() not in ("0", "false", "no"))
AVATAR_FULLSCREEN = os.getenv("SARAH_AVATAR_FULLSCREEN", "1").strip().lower() not in ("0", "false", "no")
AVATAR_PREFER_WAYLAND = os.getenv("SARAH_AVATAR_PREFER_WAYLAND", "1").strip().lower() not in ("0", "false", "no")


class AvatarDisplay:
    """Simple blue eyes avatar on black background - moves up/down with emotions.

    Features:
    - Eyes move up when AI is working/speaking
    - Eyes at center when idle
    - Eyes can express different emotions (happy, sad, surprised, etc.)
    - Smooth vertical animation
    
    SSH/HDMI Compatibility:
    - Maintains full SSH compatibility (works when SSH'd into Pi)
    - Supports Wayland, X11, and direct framebuffer (kmsdrm/fbcon)
    - Automatically detects and uses the correct video driver
    - Works with HDMI displays connected to Raspberry Pi
    - Handles sudo/root environments correctly
    
    See standalone test: sarah_simple_eyes_test.py
    """

    def __init__(self, width: int = 0, height: int = 0, fps: int = 30):
        self.width = int(width)
        self.height = int(height)
        env_fps = os.getenv("SARAH_AVATAR_FPS", "").strip()
        self.fps = int(env_fps) if env_fps.isdigit() else int(fps)
        # Keep FPS sane; also prevents divide-by-zero in blink timing.
        self.fps = max(30, self.fps)

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state_lock = threading.Lock()
        self._ai_busy_count = 0
        self._speaking = False
        self._emotion = "neutral"
        # Battery mood override (when low battery) should persist even if other parts of the system
        # temporarily set emotions for movement/speech.
        self._battery_override_emotion: Optional[str] = None
        self._eye_y_offset = 0.0  # Vertical position offset (-1.0 to 1.0)
        self._target_y_offset = 0.0
        self._eye_x_offset = 0.0  # Horizontal position offset (-1.0 to 1.0)
        self._target_x_offset = 0.0
        self._animation_time = 0.0  # For animated effects like bouncing

        # Optional manual gaze override (used for dance mode rhythmic eye movement).
        self._manual_gaze_x: Optional[float] = None
        self._manual_gaze_y: Optional[float] = None
        self._manual_gaze_until = 0.0

        # Optional manual eye color override (used for dance mode color flashes).
        self._manual_eye_color: Optional[tuple[int, int, int]] = None
        self._manual_eye_color_until = 0.0

        # Optional per-eye openness override (used for dance winks/partial winks).
        # 1.0 = fully open, 0.0 = closed.
        self._manual_eye_open_left: Optional[float] = None
        self._manual_eye_open_right: Optional[float] = None
        self._manual_eye_open_until = 0.0
        
        # Blinking
        self._is_blinking = False
        self._blink_timer = 0.0
        self._next_blink_time = random.uniform(2.0, 5.0)  # Random blink interval
        self._blink_duration = 0.15  # How long the blink lasts
        
        # Idle floating animation
        self._idle_float_x = 0.0
        self._idle_float_y = 0.0
        self._idle_animation_time = 0.0

        # Game mode (blue Dino runner) state
        self._last_render_mode: Optional[str] = None
        self._game_inited = False
        self._game_over = False
        self._game_score = 0.0
        self._game_high_score = 0.0  # Track high score across games
        self._game_speed = 240.0  # px/s
        self._game_dino_y = 0.0
        self._game_dino_vy = 0.0
        self._game_duck = False
        self._game_obstacles: deque[dict] = deque()
        self._game_spawn_timer = 0.0
        self._game_jump_latch = False
        self._game_duck_latch = False  # Prevent duck event spam

        # Matrix-style rain transition into game scene
        self._game_transition_t = 0.0
        self._game_transition_active = False
        self._game_rain_columns: list[dict] = []

    def _game_reset(self):
        self._game_inited = True
        self._game_over = False
        self._game_score = 0.0
        self._game_speed = 240.0
        self._game_dino_y = 0.0
        self._game_dino_vy = 0.0
        self._game_duck = False
        self._game_obstacles.clear()
        self._game_spawn_timer = 0.0
        self._game_jump_latch = False
        self._game_duck_latch = False

    def _start_game_transition(self):
        """Initialize eye-pixel rain transition - pixels originate from eyes and fade down."""
        self._game_transition_t = 0.0
        self._game_transition_active = True
        self._game_rain_columns = []

        w = max(1, int(self.width))
        h = max(1, int(self.height))
        
        # Eye positions (same as normal eye rendering)
        eye_w = int(w * 0.35)
        eye_h = int(h * 0.35)
        eye_spacing = int(w * 0.08)
        base_x = (w - (2 * eye_w + eye_spacing)) // 2
        base_y = int(h * 0.35)
        
        # Define eye regions
        left_eye_region = (base_x, base_y, eye_w, eye_h)
        right_eye_region = (base_x + eye_w + eye_spacing, base_y, eye_w, eye_h)
        
        # Spawn particles from within the eye areas
        rng = random.Random()
        for eye_region in [left_eye_region, right_eye_region]:
            ex, ey, ew, eh = eye_region
            # Create more particles per eye for fuller effect
            num_particles = 120
            for _ in range(num_particles):
                px = rng.uniform(ex, ex + ew)
                py = rng.uniform(ey, ey + eh)
                size = rng.randint(3, 7)
                speed = rng.uniform(200.0, 500.0)
                brightness = rng.uniform(0.4, 1.0)
                
                self._game_rain_columns.append({
                    "x": px,
                    "y": py,
                    "size": size,
                    "speed": speed,
                    "brightness": brightness
                })

    def _draw_game_scene(self, screen, dt: float, transition_only: bool = False):
        w = max(1, int(self.width))
        h = max(1, int(self.height))
        ground_y = int(h * 0.78)

        # Background
        screen.fill((0, 0, 0))

        # Ground line
        try:
            pygame.draw.line(screen, (70, 70, 70), (0, ground_y), (w, ground_y), 3)
        except Exception:
            pass

        # Dino (simple blue block)
        dino_w = int(w * 0.06)
        dino_h = int(h * 0.12)
        standing_dino_h = int(dino_h)  # Store original standing height
        dino_x = int(w * 0.18)
        dino_base_y = ground_y - dino_h - int(self._game_dino_y)
        
        dino_color = (0, 140, 255)
        
        # Draw dino (simple rectangle)
        if self._game_duck and not self._game_over:
            # Ducking: shorter, wider block
            duck_h = int(dino_h * 0.5)
            duck_w = int(dino_w * 1.4)
            duck_y = ground_y - duck_h - int(self._game_dino_y)
            pygame.draw.rect(screen, dino_color, pygame.Rect(dino_x, duck_y, duck_w, duck_h))
            # Store dino rect for collision
            dino_rect = pygame.Rect(dino_x, duck_y, duck_w, duck_h)
        else:
            # Standing: normal block
            pygame.draw.rect(screen, dino_color, pygame.Rect(dino_x, dino_base_y, dino_w, dino_h))
            # Store dino rect for collision
            dino_rect = pygame.Rect(dino_x, dino_base_y, dino_w, dino_h)

        # Obstacles (simple blocks)
        for obs in list(self._game_obstacles):
            ox = int(obs.get("x", 0))
            ow = int(obs.get("w", 10))
            oh = int(obs.get("h", 10))
            is_flying = obs.get("flying", False)
            
            if is_flying:
                # Flying obstacle (red block) - positioned to hit standing dino body (must jump)
                fly_height = int(standing_dino_h * 0.5)  # Mid-body height
                oy = ground_y - fly_height - oh
                obs_color = (255, 80, 80)  # Red
            else:
                # Ground obstacle (green block)
                oy = ground_y - oh
                obs_color = (80, 255, 80)  # Green
            
            pygame.draw.rect(screen, obs_color, pygame.Rect(ox, oy, ow, oh))

        if transition_only:
            return

        # Minimal HUD (score) - only if font is available
        try:
            if hasattr(pygame, "font"):
                pygame.font.init()
                font = pygame.font.Font(None, max(18, int(h * 0.05)))
                txt = f"Score {int(self._game_score)}" + ("  GAME OVER" if self._game_over else "")
                if self._game_high_score > 0:
                    txt += f"  High: {int(self._game_high_score)}"
                surf = font.render(txt, True, (180, 180, 180))
                screen.blit(surf, (int(w * 0.03), int(h * 0.03)))
        except Exception:
            pass

    def _update_game(self, dt: float):
        # Controls via D-pad
        dx, dy = get_game_dpad()
        jump_pressed = (dy == 1)
        duck_pressed = (dy == -1)
        restart_pressed = (dx == 1)

        if self._game_over:
            if restart_pressed or (jump_pressed and not self._game_jump_latch):
                self._game_reset()
            self._game_jump_latch = jump_pressed
            return

        self._game_duck = bool(duck_pressed)

        # Jump (edge-trigger)
        if jump_pressed and not self._game_jump_latch:
            if self._game_dino_y <= 0.01:
                self._game_dino_vy = 520.0
                # Notify robot to do forward burst
                try:
                    notify_game_jump()
                except Exception:
                    pass
        self._game_jump_latch = jump_pressed

        # Duck (edge-trigger to prevent spam)
        if duck_pressed and not self._game_duck_latch:
            try:
                notify_game_duck()
            except Exception:
                pass
        self._game_duck_latch = duck_pressed

        # Physics
        gravity = 1400.0
        self._game_dino_vy -= gravity * dt
        self._game_dino_y += self._game_dino_vy * dt
        if self._game_dino_y < 0.0:
            self._game_dino_y = 0.0
            self._game_dino_vy = 0.0

        # Speed + score
        self._game_speed = min(520.0, self._game_speed + 14.0 * dt)
        self._game_score += (self._game_speed * dt) / 10.0

        # Spawn obstacles
        self._game_spawn_timer -= dt
        if self._game_spawn_timer <= 0.0:
            # Dynamic obstacle scaling based on game progression
            w = max(1, int(self.width))
            h = max(1, int(self.height))
            ground_y = int(h * 0.78)
            
            # Calculate progression factor (0.0 to 1.0 based on speed)
            progress = min(1.0, (self._game_speed - 100.0) / 420.0)

            # Physics caps so obstacles are always passable
            jump_v = 520.0
            max_jump_px = (jump_v * jump_v) / (2.0 * gravity)
            air_time = (2.0 * jump_v) / gravity
            max_passable_w = int(max(12.0, min(float(w) * 0.25, self._game_speed * air_time * 0.85)))
            dino_h_px = int(h * 0.12)
            fly_height_px = int(dino_h_px * 0.5)
            max_flying_h = max(8, int(max_jump_px - fly_height_px - 6))
            # Ground cactus height: keep a little extra margin for discrete timestep physics
            max_ground_h = max(8, int(max_jump_px - 20))
            
            # 30% chance of flying obstacle
            is_flying = random.random() < 0.3
            
            if is_flying:
                # Flying obstacles: positioned to hit standing dino (must jump to avoid)
                # Size increases with progression
                min_h = max(15, int(h * (0.04 + progress * 0.02)))
                max_h = max(30, int(h * (0.08 + progress * 0.04)))
                max_h = min(max_h, max_flying_h)
                min_h = min(min_h, max_h)
                obs_h = random.randint(min_h, max_h)
                min_w = max(10, int(w * (0.02 + progress * 0.01)))
                max_w = max(14, int(w * (0.05 + progress * 0.03)))
                obs_w = random.randint(min_w, max_w)
            else:
                # Ground obstacles (cacti): taller and wider as game progresses
                min_h = max(18, int(h * (0.05 + progress * 0.03)))
                max_h = max(42, int(h * (0.12 + progress * 0.06)))
                max_h = min(max_h, max_ground_h)
                min_h = min(min_h, max_h)
                obs_h = random.randint(min_h, max_h)
                min_w = max(10, int(w * (0.02 + progress * 0.02)))
                max_w = max(20, int(w * (0.06 + progress * 0.04)))
                obs_w = random.randint(min_w, max_w)

            obs_w = max(8, min(int(obs_w), max_passable_w))
            
            self._game_obstacles.append({
                "x": float(w + obs_w + 10), 
                "w": int(obs_w), 
                "h": int(obs_h), 
                "gy": ground_y,
                "flying": is_flying
            })
            self._game_spawn_timer = random.uniform(0.8, 1.35)

        # Move obstacles and cull
        w = max(1, int(self.width))
        for obs in list(self._game_obstacles):
            obs["x"] = float(obs.get("x", 0.0)) - self._game_speed * dt
        while self._game_obstacles and float(self._game_obstacles[0].get("x", 0.0)) < -200:
            self._game_obstacles.popleft()

        # Collision
        h = max(1, int(self.height))
        ground_y = int(h * 0.78)
        dino_w = int(w * 0.06)
        dino_h = int(h * 0.12)
        standing_dino_h = int(dino_h)
        dino_x = int(w * 0.18)
        dino_y = ground_y - dino_h - int(self._game_dino_y)
        
        # Adjust dino hitbox for ducking
        if self._game_duck:
            dino_h = int(dino_h * 0.5)
            dino_w = int(dino_w * 1.4)
            dino_y = ground_y - dino_h - int(self._game_dino_y)
        
        dino_rect = pygame.Rect(dino_x, dino_y, max(8, dino_w), max(8, dino_h))
        
        for obs in list(self._game_obstacles):
            ox = int(obs.get("x", 0))
            ow = int(obs.get("w", 10))
            oh = int(obs.get("h", 10))
            is_flying = obs.get("flying", False)
            
            if is_flying:
                # Flying obstacle collision - positioned to hit standing dino body
                fly_height = int(standing_dino_h * 0.5)  # Mid-body height
                oy = ground_y - fly_height - oh
            else:
                # Ground obstacle collision
                oy = ground_y - oh
            
            if dino_rect.colliderect(pygame.Rect(ox, oy, ow, oh)):
                self._game_over = True
                # Update high score
                if self._game_score > self._game_high_score:
                    self._game_high_score = self._game_score
                # Notify robot about game over
                try:
                    notify_game_over()
                except Exception:
                    pass
                break

        # Update global score
        try:
            update_game_score(self._game_score)
        except Exception:
            pass

    def start(self):
        if not AVATAR_ENABLED or not pygame:
            return
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="AvatarDisplay", daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def ai_busy_enter(self):
        with self._state_lock:
            self._ai_busy_count += 1
            self._recompute_targets_locked()

    def ai_busy_exit(self):
        with self._state_lock:
            self._ai_busy_count = max(0, self._ai_busy_count - 1)
            self._recompute_targets_locked()

    def set_speaking(self, speaking: bool):
        with self._state_lock:
            self._speaking = bool(speaking)
            self._recompute_targets_locked()

    def set_battery_override(self, emotion: Optional[str]):
        """If set, forces the avatar to show a battery mood (e.g., tired/exhausted)."""
        with self._state_lock:
            if emotion is None:
                self._battery_override_emotion = None
            else:
                self._battery_override_emotion = str(emotion)
            self._recompute_targets_locked()

    def set_emotion(self, emotion: str):
        """Set avatar emotion: neutral, happy, excited, thinking, surprised, concerned, sad, listening, proud, tired, exhausted."""
        valid_emotions = ["neutral", "happy", "excited", "thinking", "surprised", "concerned", "sad", "listening", "proud", "tired", "exhausted"]
        with self._state_lock:
            if emotion in valid_emotions:
                self._emotion = emotion
            else:
                self._emotion = "neutral"
            self._recompute_targets_locked()

    def set_manual_gaze(self, x: Optional[float], y: Optional[float], ttl_s: float = 0.6):
        """Temporarily override eye gaze (x/y offsets in [-1,1]) for a short period.

        Used by dance mode to create rhythmic eye motion that matches robot movement.
        Set x/y to None to clear.
        """
        try:
            ttl = float(ttl_s)
        except Exception:
            ttl = 0.6
        ttl = max(0.05, min(3.0, ttl))
        now = time.time()
        with self._state_lock:
            self._manual_gaze_x = None if x is None else float(max(-1.0, min(1.0, float(x))))
            self._manual_gaze_y = None if y is None else float(max(-1.0, min(1.0, float(y))))
            self._manual_gaze_until = now + ttl
            self._recompute_targets_locked()

    def set_manual_eye_color(self, rgb: Optional[tuple[int, int, int]], ttl_s: float = 0.6):
        """Temporarily override eye color (RGB tuple) for a short period.

        Used by dance mode to flash/cycle eye colors.
        Set rgb to None to clear.
        """
        try:
            ttl = float(ttl_s)
        except Exception:
            ttl = 0.6
        ttl = max(0.05, min(3.0, ttl))
        now = time.time()
        with self._state_lock:
            if rgb is None:
                self._manual_eye_color = None
                self._manual_eye_color_until = 0.0
            else:
                try:
                    r, g, b = rgb
                    r = int(max(0, min(255, int(r))))
                    g = int(max(0, min(255, int(g))))
                    b = int(max(0, min(255, int(b))))
                    self._manual_eye_color = (r, g, b)
                    self._manual_eye_color_until = now + ttl
                except Exception:
                    self._manual_eye_color = None
                    self._manual_eye_color_until = 0.0

    def set_manual_eye_open(self, left_open: Optional[float], right_open: Optional[float], ttl_s: float = 0.6):
        """Temporarily override per-eye openness (1.0=open, 0.0=closed).

        Used by dance mode to create winks / partial winks.
        Pass None for both to clear.
        """
        try:
            ttl = float(ttl_s)
        except Exception:
            ttl = 0.6
        ttl = max(0.05, min(3.0, ttl))
        now = time.time()
        with self._state_lock:
            if left_open is None and right_open is None:
                self._manual_eye_open_left = None
                self._manual_eye_open_right = None
                self._manual_eye_open_until = 0.0
            else:
                lo = 1.0 if left_open is None else float(left_open)
                ro = 1.0 if right_open is None else float(right_open)
                lo = max(0.0, min(1.0, lo))
                ro = max(0.0, min(1.0, ro))
                self._manual_eye_open_left = lo
                self._manual_eye_open_right = ro
                self._manual_eye_open_until = now + ttl

    def _emotion_base_targets(self, emotion: str) -> tuple[float, float]:
        if emotion == "excited":
            return 0.0, -0.4
        if emotion == "happy":
            return 0.0, -0.2
        if emotion == "surprised":
            return 0.0, -0.35
        if emotion == "thinking":
            return 0.3, -0.15
        if emotion == "concerned":
            return 0.0, 0.1
        if emotion == "sad":
            return 0.0, 0.2
        if emotion == "tired":
            return 0.0, 0.25
        if emotion == "exhausted":
            return 0.0, 0.35
        if emotion == "listening":
            return -0.25, 0.0
        if emotion == "proud":
            return 0.0, -0.3
        return 0.0, 0.0

    def _recompute_targets_locked(self):
        effective = str(self._battery_override_emotion or self._emotion)
        base_x, base_y = self._emotion_base_targets(effective)
        target_x, target_y = base_x, base_y
        # Speaking/AI-busy are overlays: move up, but don't override a stronger "up" emotion.
        if self._ai_busy_count > 0:
            target_y = min(target_y, -0.3)
        if self._speaking:
            target_y = min(target_y, -0.2)

        # Manual gaze override (e.g., dance mode) takes precedence while active.
        now = time.time()
        if self._manual_gaze_until > now:
            if self._manual_gaze_x is not None:
                target_x = float(self._manual_gaze_x)
            if self._manual_gaze_y is not None:
                target_y = float(self._manual_gaze_y)
        else:
            self._manual_gaze_x = None
            self._manual_gaze_y = None

        self._target_x_offset = target_x
        self._target_y_offset = target_y

    def begin_mouth_sync_from_wav(self, wav_path: str):
        """Simplified - just mark as speaking."""
        self.set_speaking(True)

    def end_speaking(self):
        self.set_speaking(False)

    def _choose_video_driver(self):
        if platform.system() != "Linux":
            return
        display = (os.getenv("DISPLAY") or "").strip()
        is_ssh = bool(os.getenv("SSH_CONNECTION") or os.getenv("SSH_TTY"))

        def _uid_home_from_user(user: str):
            try:
                import pwd
                pw = pwd.getpwnam(user)
                return int(pw.pw_uid), str(pw.pw_dir)
            except Exception:
                return None, None

        sudo_user = (os.getenv("SUDO_USER") or "").strip()
        sudo_uid, sudo_home = (None, None)
        if sudo_user:
            sudo_uid, sudo_home = _uid_home_from_user(sudo_user)

        display_is_remote = bool(display) and (":" in display) and (not display.startswith(":"))

        if is_ssh and AVATAR_PREFER_WAYLAND and not os.getenv("SDL_VIDEODRIVER"):
            try:
                candidate_runtime_dirs = []
                xdg_env = (os.getenv("XDG_RUNTIME_DIR") or "").strip()
                if xdg_env:
                    candidate_runtime_dirs.append(xdg_env)
                if sudo_uid is not None:
                    candidate_runtime_dirs.append(f"/run/user/{sudo_uid}")
                for d in sorted(glob.glob("/run/user/[0-9]*")):
                    candidate_runtime_dirs.append(d)

                for runtime_dir in candidate_runtime_dirs:
                    if not runtime_dir or not os.path.isdir(runtime_dir):
                        continue
                    for candidate in ("wayland-0", "wayland-1"):
                        socket_path = os.path.join(runtime_dir, candidate)
                        if os.path.exists(socket_path):
                            os.environ["XDG_RUNTIME_DIR"] = runtime_dir
                            os.environ.setdefault("WAYLAND_DISPLAY", candidate)
                            os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
                            return
            except Exception:
                pass

        # If we're SSH'd in, try to attach to the local HDMI session.
        # On many Pi setups (including Wayland desktops), Xwayland still exposes :0.
        # Forcing SDL to use x11 here is typically the most reliable path.
        if is_ssh and (not display or display_is_remote):
            try:
                if os.path.exists("/tmp/.X11-unix/X0"):
                    os.environ["DISPLAY"] = ":0"
                    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
                    if not (os.getenv("XAUTHORITY") or "").strip():
                        # If running over SSH (and especially under sudo), XAUTHORITY may be unset.
                        # Try sudo user's .Xauthority first, then current user's, then any /home/* candidate.
                        if sudo_home and os.path.exists(os.path.join(sudo_home, ".Xauthority")):
                            os.environ["XAUTHORITY"] = os.path.join(sudo_home, ".Xauthority")
                        elif os.path.exists(os.path.expanduser("~/.Xauthority")):
                            os.environ["XAUTHORITY"] = os.path.expanduser("~/.Xauthority")
                        else:
                            for cand in sorted(glob.glob("/home/*/.Xauthority")):
                                if os.path.exists(cand):
                                    os.environ["XAUTHORITY"] = cand
                                    break
                    return
            except Exception:
                pass

        display = (os.getenv("DISPLAY") or "").strip()
        if display and (display.startswith(":0") or (not is_ssh)):
            return
        if os.getenv("SDL_VIDEODRIVER"):
            return

        for drv in ("kmsdrm", "fbcon", "directfb"):
            os.environ["SDL_VIDEODRIVER"] = drv
            try:
                pygame.display.init()
                pygame.display.quit()
                return
            except Exception:
                continue
        os.environ.pop("SDL_VIDEODRIVER", None)

    def _run(self):
        try:
            self._choose_video_driver()
            pygame.init()
            pygame.display.set_caption("SARAH")
            flags = 0
            if AVATAR_FULLSCREEN:
                flags |= pygame.FULLSCREEN
                if hasattr(pygame, "SCALED"):
                    flags |= pygame.SCALED

            # Some Pi display backends (and some SDL drivers) cannot use SCALED with a 0x0 mode.
            # Also, framebuffer-style drivers often don't support SCALED reliably.
            sdl_driver = (os.getenv("SDL_VIDEODRIVER") or "").strip().lower()
            if sdl_driver in ("kmsdrm", "fbcon", "directfb") and hasattr(pygame, "SCALED"):
                flags &= ~pygame.SCALED

            if hasattr(pygame, "DOUBLEBUF"):
                flags |= pygame.DOUBLEBUF

            if self.width <= 0 or self.height <= 0:
                # Prefer an explicit size to avoid (0,0)+SCALED failures.
                target_w, target_h = (0, 0)
                try:
                    if hasattr(pygame.display, "get_desktop_sizes"):
                        sizes = pygame.display.get_desktop_sizes() or []
                        if sizes:
                            target_w, target_h = map(int, sizes[0])
                except Exception:
                    pass

                if target_w > 0 and target_h > 0:
                    screen = pygame.display.set_mode((target_w, target_h), flags)
                else:
                    # Last resort: if we couldn't determine size, drop SCALED and let SDL pick.
                    if hasattr(pygame, "SCALED"):
                        flags &= ~pygame.SCALED
                    screen = pygame.display.set_mode((0, 0), flags)

                try:
                    self.width, self.height = screen.get_size()
                except Exception:
                    pass
            else:
                screen = pygame.display.set_mode((self.width, self.height), flags)

            clock = pygame.time.Clock()

            while self._running.is_set():
                for event in pygame.event.get():
                    if event.type == getattr(pygame, 'QUIT', None):
                        self._running.clear()
                        break

                self._draw(screen)
                pygame.display.flip()
                clock.tick(max(10, self.fps))
        except Exception as e:
            try:
                Logger.log(
                    "AVATAR",
                    "Avatar renderer failed: "
                    f"{type(e).__name__}: {e} | "
                    f"SDL_VIDEODRIVER={(os.getenv('SDL_VIDEODRIVER') or '').strip()} "
                    f"DISPLAY={(os.getenv('DISPLAY') or '').strip()} "
                    f"WAYLAND_DISPLAY={(os.getenv('WAYLAND_DISPLAY') or '').strip()} "
                    f"XDG_RUNTIME_DIR={(os.getenv('XDG_RUNTIME_DIR') or '').strip()} "
                    f"SSH={'1' if (os.getenv('SSH_CONNECTION') or os.getenv('SSH_TTY')) else '0'}",
                    "WARN",
                )
            except Exception:
                pass
            try:
                print(f"[AVATAR] Avatar renderer failed: {type(e).__name__}: {e}", file=sys.stderr)
            except Exception:
                pass
        finally:
            try:
                pygame.display.quit()
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass

    def _draw(self, screen):
        """Draw simple blue eyes on black background with dynamic movement, blinking, and idle floating."""

        # Game mode render path
        try:
            current_mode = get_current_mode()
        except Exception:
            current_mode = "manual"

        dt = 1.0 / max(10, int(self.fps))
        if current_mode == "game":
            if self._last_render_mode != "game":
                # Entering game mode
                self._game_reset()
                self._start_game_transition()
                # Ensure game starts from a neutral D-pad state
                set_game_dpad(0, 0)
            self._last_render_mode = "game"

            if self._game_transition_active:
                self._game_transition_t += dt
                
                # Black background
                screen.fill((0, 0, 0))
                
                # Update and draw eye particles falling and fading
                h = max(1, int(self.height))
                base_blue = (0, 140, 255)
                
                for particle in self._game_rain_columns:
                    # Update particle position
                    particle["y"] += particle["speed"] * dt
                    
                    # Fade out as it falls and over time
                    time_fade = 1.0 - (self._game_transition_t / 1.0)  # Fade over 1 second
                    fall_fade = 1.0 - max(0.0, min(1.0, particle["y"] / h))  # Fade as it falls
                    combined_fade = time_fade * fall_fade
                    
                    # Draw particle if still visible
                    if combined_fade > 0.05 and 0 <= particle["y"] < h:
                        intensity = particle["brightness"] * combined_fade
                        color = (
                            int(base_blue[0] * intensity),
                            int(base_blue[1] * intensity),
                            int(base_blue[2] * intensity)
                        )
                        px = int(particle["x"])
                        py = int(particle["y"])
                        sz = particle["size"]
                        pygame.draw.rect(screen, color, pygame.Rect(px, py, sz, sz))
                
                # Fade in game scene behind rain as transition progresses
                if self._game_transition_t > 0.5:
                    # Create a semi-transparent surface for the game scene
                    fade_alpha = min(255, int((self._game_transition_t - 0.5) * 510))
                    temp_surface = pygame.Surface((max(1, int(self.width)), h))
                    temp_surface.set_alpha(fade_alpha)
                    self._draw_game_scene(temp_surface, dt, transition_only=True)
                    screen.blit(temp_surface, (0, 0))
                
                # End transition after 1.0s
                if self._game_transition_t >= 1.0:
                    self._game_transition_active = False
                return

            # Normal game loop
            self._update_game(dt)
            self._draw_game_scene(screen, dt, transition_only=False)
            return
        else:
            # Leaving game mode
            if self._last_render_mode == "game":
                set_game_dpad(0, 0)
            self._last_render_mode = current_mode
        
        # Black background
        screen.fill((0, 0, 0))

        with self._state_lock:
            emotion = str(self._battery_override_emotion or self._emotion)
            current_y_offset = self._eye_y_offset
            target_y_offset = self._target_y_offset
            current_x_offset = self._eye_x_offset
            target_x_offset = self._target_x_offset
            self._animation_time += 0.05  # Increment for animations
            self._idle_animation_time += 0.03  # Slower idle animation
            
            # Update blink timer
            dt = 1.0 / max(1, int(self.fps))
            self._blink_timer += dt
            if not self._is_blinking and self._blink_timer >= self._next_blink_time:
                # Start a blink
                self._is_blinking = True
                self._blink_timer = 0.0
            elif self._is_blinking and self._blink_timer >= self._blink_duration:
                # End the blink
                self._is_blinking = False
                self._blink_timer = 0.0
                self._next_blink_time = random.uniform(2.0, 5.0)  # Next blink in 2-5 seconds
            
            is_blinking = self._is_blinking
            
            # Idle floating animation (subtle drift when not actively doing something)
            # Only apply when not speaking and emotion is neutral/listening
            if not self._speaking and emotion in ["neutral", "listening"]:
                self._idle_float_x = math.sin(self._idle_animation_time * 0.5) * 0.15
                self._idle_float_y = math.cos(self._idle_animation_time * 0.3) * 0.1
            else:
                # Fade out idle float when active
                self._idle_float_x *= 0.9
                self._idle_float_y *= 0.9

        # Smooth eye movement (faster interpolation for more dynamic feel)
        if abs(current_y_offset - target_y_offset) > 0.01:
            with self._state_lock:
                self._eye_y_offset += (target_y_offset - current_y_offset) * 0.25
        
        if abs(current_x_offset - target_x_offset) > 0.01:
            with self._state_lock:
                self._eye_x_offset += (target_x_offset - current_x_offset) * 0.25

        # Eye dimensions (like the image - rounded rectangles)
        eye_width = int(self.width * 0.35)
        eye_height = int(self.height * 0.35)
        eye_spacing = int(self.width * 0.08)
        corner_radius = int(min(eye_width, eye_height) * 0.3)

        # Base eye color (similar to image)
        eye_color = (40, 160, 255)

        # Dance/manual override: temporarily replace eye color.
        try:
            with self._state_lock:
                now = time.time()
                if self._manual_eye_color is not None and self._manual_eye_color_until > now:
                    eye_color = self._manual_eye_color
                elif self._manual_eye_color_until <= now:
                    self._manual_eye_color = None
                    self._manual_eye_color_until = 0.0
        except Exception:
            pass

        # Calculate position with offsets (including idle float)
        center_y = self.height // 2
        center_x = self.width // 2
        vertical_range = int(self.height * 0.28)  # Increased range for dance movement
        horizontal_range = int(self.width * 0.20)  # Increased range for dance movement
        
        # Apply idle floating animation
        with self._state_lock:
            idle_x = self._idle_float_x
            idle_y = self._idle_float_y
        
        # Base position with offsets and idle float
        eye_y = int(center_y + (current_y_offset * vertical_range) + (idle_y * vertical_range) - eye_height // 2)
        base_x_offset = int((current_x_offset + idle_x) * horizontal_range)

        # Emotion-based modifications
        current_eye_width = eye_width
        current_eye_height = eye_height
        additional_y_offset = 0
        additional_x_offset = 0
        
        if emotion == "happy":
            # Squinted happy eyes
            current_eye_height = int(eye_height * 0.7)
            additional_y_offset = int(eye_height * 0.15)
        elif emotion == "excited":
            # Squinted eyes with bouncing animation
            current_eye_height = int(eye_height * 0.65)
            additional_y_offset = int(eye_height * 0.15)
            # Add slight bounce effect
            bounce = math.sin(self._animation_time * 3) * 8  # Fast bouncing
            additional_y_offset += int(bounce)
            # Slight horizontal wiggle for energy
            wiggle = math.sin(self._animation_time * 4) * 5
            additional_x_offset = int(wiggle)
        elif emotion == "surprised":
            # Wider eyes
            current_eye_width = int(eye_width * 1.15)
            current_eye_height = int(eye_height * 1.15)
        elif emotion == "sad":
            # Droopy eyes
            current_eye_height = int(eye_height * 0.8)
            additional_y_offset = int(eye_height * 0.1)
        elif emotion == "tired":
            # Sleepy, half-lidded eyes
            current_eye_height = int(eye_height * 0.55)
            additional_y_offset = int(eye_height * 0.18)
        elif emotion == "exhausted":
            # Very sleepy, more closed eyes
            current_eye_height = int(eye_height * 0.4)
            additional_y_offset = int(eye_height * 0.24)
        elif emotion == "concerned":
            # Slightly droopy, subtle side-to-side
            current_eye_height = int(eye_height * 0.85)
            additional_y_offset = int(eye_height * 0.08)
            sway = math.sin(self._animation_time * 1.5) * 3
            additional_x_offset = int(sway)
        elif emotion == "thinking":
            # Eyes looking to the side with occasional shifts
            think_shift = math.sin(self._animation_time * 0.8) * 8
            additional_x_offset = int(think_shift)
        elif emotion == "proud":
            # Slightly narrowed, looking up
            current_eye_height = int(eye_height * 0.85)
        
        # Apply blink - override eye height to create closed eyes
        if is_blinking:
            current_eye_height = int(eye_height * 0.05)  # Very thin horizontal line
            additional_y_offset += int(eye_height * 0.4)  # Shift down slightly during blink

        # Dance/manual override: per-eye openness (winks/partial winks)
        left_open = 1.0
        right_open = 1.0
        try:
            with self._state_lock:
                now = time.time()
                if self._manual_eye_open_until > now:
                    if self._manual_eye_open_left is not None:
                        left_open = float(self._manual_eye_open_left)
                    if self._manual_eye_open_right is not None:
                        right_open = float(self._manual_eye_open_right)
                else:
                    self._manual_eye_open_left = None
                    self._manual_eye_open_right = None
                    self._manual_eye_open_until = 0.0
        except Exception:
            pass

        eye_y += additional_y_offset

        # Left eye
        left_eye_x = center_x - eye_spacing - current_eye_width + base_x_offset + additional_x_offset
        left_h = max(2, int(current_eye_height * left_open))
        left_y = eye_y + int((current_eye_height - left_h) * 0.5)
        left_eye_rect = pygame.Rect(left_eye_x, left_y, current_eye_width, left_h)
        pygame.draw.rect(screen, eye_color, left_eye_rect, border_radius=corner_radius)

        # Right eye
        right_eye_x = center_x + eye_spacing + base_x_offset + additional_x_offset
        right_h = max(2, int(current_eye_height * right_open))
        right_y = eye_y + int((current_eye_height - right_h) * 0.5)
        right_eye_rect = pygame.Rect(right_eye_x, right_y, current_eye_width, right_h)
        pygame.draw.rect(screen, eye_color, right_eye_rect, border_radius=corner_radius)




# Global avatar instance, started in main() on Raspberry Pi.
_AVATAR: Optional[AvatarDisplay] = None

# Active ultrasonic sensors reference, set in main().
# This avoids fragile `from __main__ import sensors` imports (which can break under
# debugpy, when imported as a module, or in other execution contexts).
_ACTIVE_ULTRASONIC_SENSORS = None


def set_active_ultrasonic_sensors(sensors_obj) -> None:
    global _ACTIVE_ULTRASONIC_SENSORS
    _ACTIVE_ULTRASONIC_SENSORS = sensors_obj


@contextmanager
def avatar_ai_activity():
    """Marks a region where an AI model is actively working."""
    global _AVATAR
    if _AVATAR is not None:
        try:
            _AVATAR.ai_busy_enter()
        except Exception:
            pass
    try:
        yield
    finally:
        if _AVATAR is not None:
            try:
                _AVATAR.ai_busy_exit()
            except Exception:
                pass

def set_avatar_emotion(emotion: str):
    """Helper to set avatar emotion safely."""
    global _AVATAR
    if _AVATAR is not None:
        try:
            _AVATAR.set_emotion(emotion)
        except Exception:
            pass


def set_avatar_gaze(x: Optional[float], y: Optional[float], ttl_s: float = 0.6) -> None:
    """Temporarily override the avatar gaze (eye offsets) if the avatar is running."""
    global _AVATAR
    if _AVATAR is None:
        return
    try:
        if hasattr(_AVATAR, "set_manual_gaze"):
            _AVATAR.set_manual_gaze(x, y, ttl_s=ttl_s)
    except Exception:
        pass


def set_avatar_eye_color(rgb: Optional[tuple[int, int, int]], ttl_s: float = 0.6) -> None:
    """Temporarily override the avatar eye color (RGB) if the avatar is running."""
    global _AVATAR
    if _AVATAR is None:
        return
    try:
        if hasattr(_AVATAR, "set_manual_eye_color"):
            _AVATAR.set_manual_eye_color(rgb, ttl_s=ttl_s)
    except Exception:
        pass


def set_avatar_eye_open(left_open: Optional[float], right_open: Optional[float], ttl_s: float = 0.6) -> None:
    """Temporarily override per-eye openness (1.0=open, 0.0=closed) if the avatar is running."""
    global _AVATAR
    if _AVATAR is None:
        return
    try:
        if hasattr(_AVATAR, "set_manual_eye_open"):
            _AVATAR.set_manual_eye_open(left_open, right_open, ttl_s=ttl_s)
    except Exception:
        pass


# ==================== GAME MODE INPUT (D-PAD) ====================
_GAME_INPUT_LOCK = threading.Lock()
_GAME_DPAD = (0, 0)  # (x,y) normalized to -1/0/1
_GAME_STATE_LOCK = threading.Lock()
_GAME_JUMP_EVENT = False
_GAME_DUCK_EVENT = False
_GAME_OVER_EVENT = False
_GAME_SCORE = 0.0


def set_game_dpad(x: int, y: int) -> None:
    global _GAME_DPAD
    try:
        x = int(max(-1, min(1, int(x))))
        y = int(max(-1, min(1, int(y))))
    except Exception:
        x, y = (0, 0)
    with _GAME_INPUT_LOCK:
        _GAME_DPAD = (x, y)


def get_game_dpad() -> tuple[int, int]:
    with _GAME_INPUT_LOCK:
        return int(_GAME_DPAD[0]), int(_GAME_DPAD[1])


def notify_game_jump() -> None:
    """Called by avatar when dino jumps in game - triggers robot forward burst."""
    global _GAME_JUMP_EVENT
    with _GAME_STATE_LOCK:
        _GAME_JUMP_EVENT = True


def notify_game_duck() -> None:
    """Called by avatar when dino ducks - triggers robot backward movement."""
    global _GAME_DUCK_EVENT
    with _GAME_STATE_LOCK:
        _GAME_DUCK_EVENT = True


def notify_game_over() -> None:
    """Called by avatar when game over occurs."""
    global _GAME_OVER_EVENT
    with _GAME_STATE_LOCK:
        _GAME_OVER_EVENT = True


def update_game_score(score: float) -> None:
    """Update global game score for tracking."""
    global _GAME_SCORE
    with _GAME_STATE_LOCK:
        _GAME_SCORE = float(score)


def get_game_events() -> dict:
    """Get and clear game events for robot movement."""
    global _GAME_JUMP_EVENT, _GAME_DUCK_EVENT, _GAME_OVER_EVENT, _GAME_SCORE
    with _GAME_STATE_LOCK:
        events = {
            'jump': bool(_GAME_JUMP_EVENT),
            'duck': bool(_GAME_DUCK_EVENT),
            'game_over': bool(_GAME_OVER_EVENT),
            'score': float(_GAME_SCORE)
        }
        _GAME_JUMP_EVENT = False
        _GAME_DUCK_EVENT = False
        _GAME_OVER_EVENT = False
        return events


def set_avatar_battery_mood(emotion: Optional[str]):
    """Sets (or clears) a persistent battery mood overlay on the avatar."""
    global _AVATAR
    if _AVATAR is None:
        return
    try:
        if hasattr(_AVATAR, "set_battery_override"):
            _AVATAR.set_battery_override(emotion)
        else:
            if emotion is None:
                _AVATAR.set_emotion("neutral")
            else:
                _AVATAR.set_emotion(str(emotion))
    except Exception:
        pass


def detect_emotion_from_text(text: str) -> str:
    """
    Analyze text content and return appropriate emotion for avatar.
    Used to make eyes respond dynamically to what's being said.
    """
    text_lower = text.lower()
    
    # Excited/Happy words
    if any(word in text_lower for word in [
        "excited", "amazing", "wonderful", "great", "awesome",
        "yes!", "found", "perfect", "excellent", "yay", "fantastic"
    ]):
        return "excited"
    
    # Happy words
    if any(word in text_lower for word in [
        "happy", "glad", "nice", "good", "pleasure", "lovely", "smile"
    ]):
        return "happy"
    
    # Surprised words
    if any(word in text_lower for word in [
        "wow", "whoa", "oh!", "surprising", "unexpected", "incredible", "really?"
    ]):
        return "surprised"
    
    # Concerned/Worried words
    if any(word in text_lower for word in [
        "careful", "watch out", "danger", "problem", "obstacle",
        "stuck", "blocked", "warning", "caution"
    ]):
        return "concerned"
    
    # Sad words
    if any(word in text_lower for word in [
        "sorry", "unfortunately", "failed", "can't", "unable", "disappointed"
    ]):
        return "sad"
    
    # Thinking words
    if any(word in text_lower for word in [
        "hmm", "thinking", "considering", "analyzing", "processing",
        "let me", "i'm analyzing", "calculating"
    ]):
        return "thinking"
    
    # Proud/Confident words
    if any(word in text_lower for word in [
        "accomplished", "completed", "done", "successfully", "achieved",
        "success", "mission"
    ]):
        return "proud"
    
    # Listening words
    if any(word in text_lower for word in [
        "listening", "hearing", "waiting", "ready", "awaiting"
    ]):
        return "listening"
    
    # Default to neutral
    return "neutral"


def set_emotion_for_movement(command: str):
    """
    Map movement commands to emotions automatically.
    Makes eyes respond to robot movements.
    """
    global _AVATAR
    if not _AVATAR:
        return
    
    emotion_map = {
        "FORWARD": "excited",      # Determined/excited look when moving forward
        "BACKWARD": "concerned",   # Cautious look when backing up
        "LEFT": "thinking",        # Thoughtful when turning left
        "RIGHT": "thinking",       # Thoughtful when turning right
        "STOP": "neutral",         # Calm when stopped
        "SPEED": "excited",        # Excited about speed changes
    }
    
    emotion = emotion_map.get(command, "neutral")
    _AVATAR.set_emotion(emotion)
    # Silent: emotion set for command (don't print to avoid confusion)
    dprint(SARAH_DEBUG, f"[AVATAR] Emotion set to '{emotion}' for command '{command}'")


# GPIO Zero for Raspberry Pi (motor control and sensors)
#
# Raspberry Pi 5 note:
# - gpiozero can use different "pin factories".
# - On Pi 5, the recommended backend is typically `lgpio`.
# - In a venv, you'll often need `rpi-lgpio` (pip) or `python3-lgpio` (apt).
try:
    from gpiozero import Device, DigitalOutputDevice, PWMOutputDevice, DigitalInputDevice  # type: ignore

    GPIOZERO_AVAILABLE = True

    if IS_RASPBERRY_PI:
        # Prefer lgpio on Pi 5 (and generally on modern Raspberry Pi OS).
        # However, Python 3.13 venvs often don't have a usable `lgpio` module wheel.
        # So we try: lgpio -> pigpio -> native (only if accessible).
        # Only force it if the user hasn't explicitly chosen a pin factory.
        forced_factory: str | None = None
        try:
            existing_factory = os.getenv("GPIOZERO_PIN_FACTORY", "").strip().lower()
        except Exception:
            existing_factory = ""

        def _pigpio_host_port() -> tuple[str, int]:
            host = (
                os.environ.get("SARAH_PIGPIO_HOST")
                or os.environ.get("PIGPIO_ADDR")
                or "localhost"
            ).strip() or "localhost"
            try:
                port = int(
                    (
                        os.environ.get("SARAH_PIGPIO_PORT")
                        or os.environ.get("PIGPIO_PORT")
                        or "8888"
                    ).strip()
                )
            except Exception:
                port = 8888
            return host, port

        def _can_connect(host: str, port: int, timeout_s: float = 0.5) -> bool:
            try:
                import socket

                with socket.create_connection((host, port), timeout=timeout_s):
                    return True
            except Exception:
                return False

        def _try_start_pigpiod() -> bool:
            # Best-effort: start daemon via systemd when available; otherwise launch pigpiod directly.
            try:
                import subprocess
                import shutil

                if shutil.which("systemctl"):
                    # If already active, nothing to do.
                    for unit in ("pigpiod", "pigpio"):
                        check = subprocess.run(
                            ["systemctl", "is-active", unit],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            text=True,
                        )
                        if check.returncode == 0:
                            return True

                    # Try starting without prompting for a password (works if sudoers allows it).
                    for unit in ("pigpiod", "pigpio"):
                        start = subprocess.run(
                            ["sudo", "-n", "systemctl", "start", unit],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            text=True,
                        )
                        if start.returncode == 0:
                            return True

                # Fall back: launch pigpiod directly (common on minimal installs without the unit file).
                if shutil.which("pigpiod"):
                    start = subprocess.run(
                        ["sudo", "-n", "pigpiod"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                    if start.returncode == 0:
                        return True
            except Exception:
                pass
            return False

        # If the user explicitly selected pigpio, do a preflight now (before Drive init)
        # so we fail fast with actionable output (and can try to start pigpiod).
        if existing_factory == "pigpio":
            try:
                from gpiozero.pins.pigpio import PiGPIOFactory  # type: ignore

                host, port = _pigpio_host_port()
                if not _can_connect(host, port) and host in ("localhost", "127.0.0.1"):
                    _try_start_pigpiod()
                if not _can_connect(host, port):
                    raise OSError(
                        f"pigpio daemon not reachable at {host}:{port} "
                        f"(start with: sudo pigpiod OR sudo systemctl enable --now pigpiod)"
                    )
                Device.pin_factory = PiGPIOFactory(host=host, port=port)
                forced_factory = "pigpio"
            except Exception as e:
                GPIOZERO_AVAILABLE = False
                DigitalOutputDevice = None  # type: ignore
                PWMOutputDevice = None  # type: ignore
                DigitalInputDevice = None  # type: ignore
                print(f"[WARN] GPIOZERO_PIN_FACTORY=pigpio but pigpio is not usable: {type(e).__name__}: {e}")
                print("[WARN] Start the daemon and retry:")
                try:
                    import shutil

                    if not shutil.which("pigpiod"):
                        print("[WARN]   pigpiod is not installed (command not found)")
                        print("[WARN]   sudo apt-get update")
                        print("[WARN]   sudo apt-get install -y pigpio")
                    else:
                        print("[WARN]   sudo pigpiod")
                except Exception:
                    print("[WARN]   sudo apt-get update")
                    print("[WARN]   sudo apt-get install -y pigpio")
                    print("[WARN]   sudo pigpiod")

                print("[WARN] Verify:")
                print("[WARN]   which pigpiod")
                print("[WARN]   ss -ltnp | grep 8888")
                print("[WARN]   python -c \"import pigpio; pi=pigpio.pi('localhost',8888); print('connected=',pi.connected); pi.stop()\"")

        if not existing_factory and GPIOZERO_AVAILABLE:

            last_err: Exception | None = None
            # On some Pi images, pigpio packages are not available (no pigpiod),
            # so prefer native before pigpio when auto-selecting.
            for factory_name in ("lgpio", "native", "pigpio"):
                try:
                    if factory_name == "lgpio":
                        from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore

                        Device.pin_factory = LGPIOFactory()
                    else:
                        from gpiozero.pins.native import NativeFactory  # type: ignore

                        Device.pin_factory = NativeFactory()

                    if factory_name == "pigpio":
                        from gpiozero.pins.pigpio import PiGPIOFactory  # type: ignore
                        import shutil

                        host, port = _pigpio_host_port()
                        # If pigpiod isn't installed, skip pigpio (it will never connect).
                        if host in ("localhost", "127.0.0.1") and not shutil.which("pigpiod"):
                            raise OSError("pigpiod not installed (command not found)")

                        # Preflight the daemon so we can give a better error (and optionally start it).
                        if not _can_connect(host, port):
                            if host in ("localhost", "127.0.0.1"):
                                _try_start_pigpiod()
                            if not _can_connect(host, port):
                                raise OSError(f"pigpio daemon not reachable at {host}:{port}")

                        Device.pin_factory = PiGPIOFactory(host=host, port=port)

                    os.environ["GPIOZERO_PIN_FACTORY"] = factory_name
                    forced_factory = factory_name
                    break
                except Exception as e:
                    last_err = e
                    continue

            # If none succeeded, leave gpiozero to its default discovery; we'll fail fast below.
            if forced_factory is None and last_err is not None:
                print(f"[WARN] gpiozero pin factory auto-select failed: {type(last_err).__name__}: {last_err}")

        # Touch the factory to trigger any BadPinFactory / permission issues early.
        try:
            _ = Device.pin_factory
        except Exception as e:
            GPIOZERO_AVAILABLE = False
            DigitalOutputDevice = None  # type: ignore
            PWMOutputDevice = None  # type: ignore
            DigitalInputDevice = None  # type: ignore
            print(f"[WARN] gpiozero present but pin factory failed: {type(e).__name__}: {e}")
            print("[WARN] GPIO control disabled.")

            # Helpful diagnostics: on some OS images / kernels, /dev/gpiomem is not present.
            # In that case, gpiozero's 'native' backend cannot work; prefer lgpio (/dev/gpiochip*).
            try:
                import glob as _glob
                import getpass as _getpass

                user = _getpass.getuser()
                euid = os.geteuid() if hasattr(os, "geteuid") else None
                gpiomem_exists = os.path.exists("/dev/gpiomem")
                gpiochips = sorted(_glob.glob("/dev/gpiochip*"))

                print(
                    f"[WARN] GPIO diagnostics: user={user} euid={euid} "
                    f"/dev/gpiomem={gpiomem_exists} /dev/gpiochip*={len(gpiochips)}"
                )
                if gpiochips:
                    sample = ", ".join(gpiochips[:3])
                    more = "..." if len(gpiochips) > 3 else ""
                    print(f"[WARN] GPIO diagnostics: found {sample}{more}")
            except Exception:
                pass

            print("[WARN] Fix options (Pi 5):")
            print("[WARN]   Option A (recommended): use Raspberry Pi OS system Python + apt packages")
            print("[WARN]     sudo apt-get install -y python3-lgpio python3-gpiozero")
            print("[WARN]     # then run using system python/venv (often 3.11), not Python 3.13")
            print("[WARN]")
            print("[WARN]   Option B (works well in Python 3.13 venv): use pigpio daemon + pigpio pin factory")
            print("[WARN]     sudo apt-get install -y pigpio")
            print("[WARN]     sudo systemctl enable --now pigpiod")
            print("[WARN]     # verify: systemctl status pigpiod --no-pager")
            print("[WARN]     pip install pigpio")
            print("[WARN]     export GPIOZERO_PIN_FACTORY=pigpio")
            print("[WARN]     # optional: export PIGPIO_ADDR=localhost ; export PIGPIO_PORT=8888")
            print("[WARN]")
            print("[WARN]   Option C (native): ensure permissions for /dev/gpiomem")
            print("[WARN]     sudo usermod -aG gpio $USER && sudo reboot")
            print("[WARN]     # NOTE: if /dev/gpiomem does not exist on your OS/kernel, native cannot work")

        if GPIOZERO_AVAILABLE:
            factory_label = forced_factory or existing_factory
            extra = f" (pin factory={factory_label})" if factory_label else ""
            print(f"[INIT] gpiozero available - GPIO control enabled{extra}")

except ImportError:
    GPIOZERO_AVAILABLE = False
    DigitalOutputDevice = None  # type: ignore
    PWMOutputDevice = None  # type: ignore
    DigitalInputDevice = None  # type: ignore
    if IS_RASPBERRY_PI:
        print("[WARN] gpiozero not installed. GPIO control disabled.")
        print("[WARN]   Install (venv): pip install gpiozero rpi-lgpio")

except Exception as e:
    # gpiozero can raise non-RuntimeError exceptions on import if a pin backend is unavailable.
    GPIOZERO_AVAILABLE = False
    DigitalOutputDevice = None  # type: ignore
    PWMOutputDevice = None  # type: ignore
    DigitalInputDevice = None  # type: ignore
    if IS_RASPBERRY_PI:
        print(f"[WARN] GPIO initialization failed: {type(e).__name__}: {e}")
        print("[WARN] GPIO control disabled.")
        print("[WARN] Fix on Pi 5 (venv): pip install gpiozero rpi-lgpio")
        print("[WARN] Fix on Pi OS (system): sudo apt-get install -y python3-gpiozero python3-lgpio")

# ==================== UNIFIED LOGGER ====================
class Logger:
    """Unified logging with optional timestamps and color coding."""
    ENABLE_COLORS = IS_RASPBERRY_PI  # Disable colors on RPi for log file compatibility
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'MAGENTA': '\033[95m',
    }
    
    @staticmethod
    def format_message(module, level, message, add_timestamp=True):
        """Format log message with optional timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S") if add_timestamp else ""
        time_str = f"[{timestamp}] " if timestamp else ""
        return f"{time_str}[{module}] {message}"
    
    @staticmethod
    def log(module, message, level="INFO"):
        """Log message with module prefix (INFO, WARN, ERROR, SUCCESS)."""
        color = {
            'INFO': Logger.COLORS['CYAN'],
            'WARN': Logger.COLORS['YELLOW'],
            'ERROR': Logger.COLORS['RED'],
            'SUCCESS': Logger.COLORS['GREEN'],
        }.get(level, Logger.COLORS['RESET'])
        
        formatted = Logger.format_message(module, level, message)
        if Logger.ENABLE_COLORS:
            print(f"{color}{formatted}{Logger.COLORS['RESET']}")
        else:
            print(formatted)

# ==================== RPi5 UTILITIES ====================
def get_cpu_temperature():
    """Get RPi5 CPU temperature in Celsius. Returns None if unavailable."""
    if not IS_RASPBERRY_PI:
        return None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_c = int(f.read().strip()) / 1000.0
            return round(temp_c, 1)
    except (FileNotFoundError, ValueError, OSError):
        return None

def cleanup_temp_files(pattern: str = "tts_temp_*.wav"):
    """Remove temporary files to save RPi disk space."""
    temp_dir = gettempdir()
    temp_files = glob.glob(f"{temp_dir}/{pattern}")
    removed_count = 0
    for filepath in temp_files:
        try:
            os.remove(filepath)
            removed_count += 1
        except (OSError, PermissionError):
            pass
    if removed_count > 0:
        Logger.log("CLEANUP", f"Removed {removed_count} temp files")

# Speed constants optimized for TT Motor (200RPM, 1:48 gearbox, 3-6V DC)
# PWM <40% = Motor stalls (insufficient voltage). PWM >80% = Near maximum RPM.
# Safe operating range: 40-100% PWM - BUT we'll use 100% with startup kick for reliability
MIN_SPEED = 0       # No minimum - let motors run at any speed (kick handles startup)
DEFAULT_SPEED = 100 # Default to 100% for maximum torque and reliability
MAX_SPEED = 100     # Maximum speed (100% = full battery voltage)
CONTROLLER_POLL_INTERVAL = 0.2  # INCREASED: Was 0.05, now 0.2s to give TTS time to complete
AUTONOMOUS_LOOP_INTERVAL = 1.0  # REDUCED: Was 5.0s, now 1.0s for faster decision cycles and continuous motor control
LLAMA_RESPONSE_TIMEOUT = 60  # seconds - max time to wait for Llama response (INCREASED for remote server processing)
AUTONOMOUS_DECISION_TIMEOUT = 15  # OPTIMIZED: Reduced from 25s to 15s for faster decisions
AUTONOMOUS_VISION_TIMEOUT = 12  # OPTIMIZED: Vision analysis timeout (faster than decision)
LLAMA_CACHE_SIZE = 20  # cache size for recent queries

# Token budgets
# - Movement/command generation needs short outputs for speed.
# - Chat should be allowed to respond fully.
try:
    # Back-compat env name: SARAH_AUTO_MAX_OUTPUT_TOKENS
    LLAMA_MAX_OUTPUT_TOKENS_MOVE = int(os.getenv("SARAH_MOVE_MAX_OUTPUT_TOKENS", os.getenv("SARAH_AUTO_MAX_OUTPUT_TOKENS", "100")) or "100")
except Exception:
    LLAMA_MAX_OUTPUT_TOKENS_MOVE = 100
try:
    LLAMA_MAX_OUTPUT_TOKENS_CHAT = int(os.getenv("SARAH_CHAT_MAX_OUTPUT_TOKENS", "250") or "250")
except Exception:
    LLAMA_MAX_OUTPUT_TOKENS_CHAT = 250

# Backward-compatible default used by non-autonomous chat-style calls.
LLAMA_MAX_OUTPUT_TOKENS = LLAMA_MAX_OUTPUT_TOKENS_CHAT

# Controller disconnect grace period (seconds). If the controller drops briefly
# (e.g., Bluetooth hiccup), we avoid immediately forcing a mode change.
# Manual mode still requires a controller; this only delays the automatic
# MANUAL->AUTONOMOUS fallback triggered by disconnect events.
try:
    CONTROLLER_DISCONNECT_GRACE_S = float(os.getenv("SARAH_CONTROLLER_DISCONNECT_GRACE_S", "2.5") or "2.5")
except Exception:
    CONTROLLER_DISCONNECT_GRACE_S = 2.5
if CONTROLLER_DISCONNECT_GRACE_S < 0:
    CONTROLLER_DISCONNECT_GRACE_S = 0.0

# Global mode state for dynamic mode switching.
# IMPORTANT: This is intentionally simple and thread-safe. Threads check CURRENT_MODE
# before commanding motors. We do NOT set the global stop_event for mode switches.
CURRENT_MODE = "manual"  # 'manual', 'autonomous', 'dance', or 'chat'

# When a forward move triggers a mid-movement stop, remember it briefly so
# the next autonomous cycle doesn't immediately re-issue FORWARD.
_LAST_AUTO_FORWARD_MIDSTOP_TS = 0.0
_LAST_AUTO_FORWARD_MIDSTOP_CM = None

# Recent evasive direction memory to prevent oscillating left/right spins near obstacles.
_LAST_EVADE_TS = 0.0
_LAST_EVADE_DIR = None

# Controller presence/connection state.
# We treat MANUAL as requiring an active controller; if none is connected,
# we fall back to AUTONOMOUS to avoid a "stuck" manual mode.
_CONTROLLER_LOCK = threading.RLock()
_CONTROLLER_CONNECTED = False


def set_controller_connected(is_connected: bool) -> None:
    global _CONTROLLER_CONNECTED
    with _CONTROLLER_LOCK:
        _CONTROLLER_CONNECTED = bool(is_connected)


def is_controller_connected() -> bool:
    with _CONTROLLER_LOCK:
        return bool(_CONTROLLER_CONNECTED)

# When entering GAME/DANCE from MANUAL/AUTONOMOUS, remember where we came from
# so "end game mode" / "stop dancing" can return to the prior control mode.
_MODE_BEFORE_GAME: Optional[str] = None
_MODE_BEFORE_DANCE: Optional[str] = None
# Condition-based mode switching so threads can wait without polling.
_MODE_LOCK = threading.RLock()
_MODE_COND = threading.Condition(_MODE_LOCK)
_MODE_SEQ = 0

# Autonomous navigation goals
# Lightweight: uses command-based dead-reckoning (no wheel encoders).
RETURN_TO_START_EVENT = threading.Event()
EXPLORE_REQUEST_EVENT = threading.Event()
RESET_MAP_MEMORY_EVENT = threading.Event()

# Autonomous safety tuning
# Faster, room-scale exploration needs longer bursts than 1.0s, but we still keep a hard cap.
# Forward motion remains additionally bounded by the adaptive safety clamp + mid-movement ultrasonic monitoring.
AUTONOMOUS_MAX_MOVE_DURATION = 5.0  # seconds (hard cap for FORWARD/BACKWARD/LEFT/RIGHT)
AUTONOMOUS_OBSTACLE_PAUSE_SECONDS = 0.25  # seconds to pause after an obstacle-triggered STOP
# Loosened slightly for quicker exploration (robot will approach a bit closer before stopping).
ULTRASONIC_STOP_DISTANCE_CM = 15.0  # stop if any valid sensor reading is <= this
ULTRASONIC_READ_TIMEOUT_S = 0.1  # FIXED: HC-SR04 needs up to 38ms for max range (400cm), using 100ms for safety
ULTRASONIC_ECHO_HIGH_TIMEOUT_S = 0.03  # FIXED: Max echo pulse width is ~23ms at 400cm, using 30ms for safety
ULTRASONIC_MIN_VALID_CM = 2.0  # ignore <2cm readings for decisions (often unreliable at very close range)


def get_current_mode() -> str:
    with _MODE_LOCK:
        return str(CURRENT_MODE)


def _get_mode_seq() -> int:
    with _MODE_LOCK:
        return int(_MODE_SEQ)


def _wait_for_mode_change(last_seq: int, timeout: float = 0.5) -> int:
    """Block until CURRENT_MODE changes (or timeout), returning the new seq."""
    try:
        timeout = float(timeout)
    except Exception:
        timeout = 0.5
    if timeout < 0:
        timeout = 0.0

    with _MODE_COND:
        if _MODE_SEQ == last_seq:
            _MODE_COND.wait(timeout=timeout)
        return int(_MODE_SEQ)


def set_current_mode(mode: str, source: str = "") -> None:
    global CURRENT_MODE, _MODE_SEQ, _MODE_BEFORE_GAME, _MODE_BEFORE_DANCE
    mode = (mode or "").strip().lower()
    # Manual requires a controller. If none is connected, fall back to autonomous.
    if mode == "manual" and (not is_controller_connected()):
        mode = "autonomous"
        source = (source + "_no_controller").strip("_") if source else "no_controller"
    if mode not in ("manual", "autonomous", "dance", "chat", "game"):
        return
    
    mode_changed = False
    with _MODE_COND:
        # Record the last control mode before entering special modes.
        try:
            if mode == "game" and CURRENT_MODE != "game":
                if str(CURRENT_MODE) in ("manual", "autonomous"):
                    _MODE_BEFORE_GAME = str(CURRENT_MODE)
            elif mode == "dance" and CURRENT_MODE != "dance":
                if str(CURRENT_MODE) in ("manual", "autonomous"):
                    _MODE_BEFORE_DANCE = str(CURRENT_MODE)
        except Exception:
            pass

        if CURRENT_MODE != mode:
            CURRENT_MODE = mode
            _MODE_SEQ += 1
            mode_changed = True
            _MODE_COND.notify_all()
    
    # Print outside lock to avoid blocking threads waiting on condition
    if mode_changed:
        if source:
            print(f"[MODE] Switched to {mode.upper()} (source={source})")
        else:
            print(f"[MODE] Switched to {mode.upper()}")


def request_return_to_start(source: str = "") -> None:
    """Request autonomous mode to retrace its path back to the start."""
    try:
        RETURN_TO_START_EVENT.set()
    except Exception:
        pass
    try:
        # Returning home is an autonomous behavior.
        set_current_mode("autonomous", source=source or "return_to_start")
    except Exception:
        pass


def request_explore(source: str = "") -> None:
    """Request autonomous exploration (clears return-to-start if set)."""
    try:
        RETURN_TO_START_EVENT.clear()
    except Exception:
        pass
    try:
        EXPLORE_REQUEST_EVENT.set()
    except Exception:
        pass
    try:
        set_current_mode("autonomous", source=source or "explore")
    except Exception:
        pass


def request_reset_map_memory(source: str = "") -> None:
    """Request the autonomous thread to reset its persistent exploration/map memory.

    Signals the autonomous thread to wipe its in-memory map (visited/blocked/pose/history)
    and delete the persistence file. The thread handles the actual deletion to avoid
    race conditions and ensure the correct path is used.
    """

    try:
        RESET_MAP_MEMORY_EVENT.set()
    except Exception:
        pass

    # Do not force a mode change here; reset should work from any mode.
    if source:
        dprint(True, f"[MAP] Reset requested (source={source})")
    else:
        dprint(True, "[MAP] Reset requested")


def request_dance(source: str = "") -> None:
    """Request dance mode (music + rhythmic motion)."""
    try:
        set_current_mode("dance", source=source or "dance")
    except Exception:
        pass


def request_game(source: str = "") -> None:
    """Request game mode (local blue Dino runner on the Pi display)."""
    try:
        set_current_mode("game", source=source or "game")
    except Exception:
        pass


def _restore_mode_after_special(prev: Optional[str]) -> str:
    """Return the safe mode to restore to after GAME/DANCE ends."""
    prev = (prev or "").strip().lower()
    if prev == "manual" and (not is_controller_connected()):
        return "autonomous"
    if prev in ("manual", "autonomous"):
        return prev
    return "autonomous" if (not is_controller_connected()) else "manual"


def exit_game_mode(source: str = "") -> None:
    """Exit game mode and return to the previous manual/autonomous mode (defaults to manual)."""
    global _MODE_BEFORE_GAME
    # Read and clear _MODE_BEFORE_GAME atomically to avoid races.
    with _MODE_LOCK:
        prev_mode = _MODE_BEFORE_GAME
        _MODE_BEFORE_GAME = None
    target = _restore_mode_after_special(prev_mode)
    set_current_mode(target, source=source or "game_end")


def exit_dance_mode(source: str = "") -> None:
    """Exit dance mode and return to the previous manual/autonomous mode (defaults to manual)."""
    global _MODE_BEFORE_DANCE
    # Read and clear _MODE_BEFORE_DANCE atomically to avoid races.
    with _MODE_LOCK:
        prev_mode = _MODE_BEFORE_DANCE
        _MODE_BEFORE_DANCE = None
    target = _restore_mode_after_special(prev_mode)
    set_current_mode(target, source=source or "dance_end")

# Ultrasonic sensor default pins (trigger, echo) for left, center, right
# FIXED: Avoid GPIO16 and GPIO8 which have hardware pull-ups/SPI conflicts on Raspberry Pi 5
# Using GPIO pins that are safe and don't conflict with motor driver
ULTRASONIC_PINS = [
    (23, 24),  # left sensor: trigger=GPIO23(pin16), echo=GPIO24(pin18) - SAFE PINS
    (22, 25),  # center sensor: trigger=GPIO22(pin15), echo=GPIO25(pin22) - SAFE PINS
    (9, 10),   # right sensor: trigger=GPIO9(pin21), echo=GPIO10(pin19) - WORKING
]
# Audio / model config
# ==================== DISTRIBUTED SYSTEM CONFIGURATION ====================
# Choose how to run the AI model:
# - "local": Run Ollama locally on the Raspberry Pi (original behavior)
# - "remote": Use remote AI server on Windows/other computer (recommended for RPi)
#
# AI MODE SELECTION:
# - "remote": Raspberry Pi + Windows PC with ai_server.py (recommended for Pi)
#   Requires: ai_server.py running on Windows, Ollama on Windows
# - "local": Single machine with Ollama (use on Windows for testing)
#   Requires: Ollama installed and running locally (ollama serve)
#
# Auto-detection: Remote mode on Pi, Local on Windows
# Manual override: Uncomment/change AI_MODE = "remote" below to force a mode

# Auto-select mode based on detected platform
if IS_RASPBERRY_PI:
    AI_MODE = "remote"  # Force remote on Raspberry Pi
else:
    AI_MODE = "local"  # Default to local on Windows (can override below)

# MANUAL OVERRIDE: Uncomment to force remote mode on Windows (for testing)
# AI_MODE = "remote"

# Remote Ollama server URL (Windows box running `ollama serve`).
# Override without editing code:
#   export SARAH_REMOTE_OLLAMA_URL="http://100.119.188.18:11434"
REMOTE_AI_SERVER_URL = os.environ.get("SARAH_REMOTE_OLLAMA_URL", "http://100.119.188.18:11434").strip()  # Windows PC IP - Ollama server

# Ollama Configuration - supports both local and remote servers
# Windows uses localhost, Raspberry Pi will connect to Windows IP
# DEPRECATED: Dynamic detection - now using explicit IP for reliability
# OLLAMA_SERVER_URL = "http://localhost:11434" if not IS_RASPBERRY_PI else "http://10.0.0.212:11434"

# Memory monitoring functions (for 4GB constraint)
def get_memory_status():
    """Get current memory usage. Works on Windows and Raspberry Pi with graceful fallback."""
    if not MEMORY_MONITORING_ENABLED:
        return {'percent': 0, 'used_gb': 0, 'available_gb': 0, 'total_gb': 4, 'platform': 'Unknown'}
    try:
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': round(mem.used / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
            'total_gb': round(mem.total / (1024**3), 2),
            'platform': 'Raspberry Pi' if IS_RASPBERRY_PI else 'Windows/Linux'
        }
    except Exception as e:
        # Fallback for Raspberry Pi if psutil fails - read from /proc/meminfo
        if IS_RASPBERRY_PI:
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        key, val = line.split(':')
                        meminfo[key.strip()] = int(val.split()[0]) * 1024  # Convert KB to bytes
                    total = meminfo.get('MemTotal', 4 * 1024**3)
                    available = meminfo.get('MemAvailable', 2 * 1024**3)
                    used = total - available
                    return {
                        'percent': round((used / total) * 100, 1) if total > 0 else 0,
                        'used_gb': round(used / (1024**3), 2),
                        'available_gb': round(available / (1024**3), 2),
                        'total_gb': round(total / (1024**3), 2),
                        'platform': 'Raspberry Pi (fallback)'
                    }
            except Exception:
                pass
        # Final fallback if all methods fail
        return {'percent': 0, 'used_gb': 0, 'available_gb': 0, 'total_gb': 4, 'platform': 'Unknown (fallback)'}

def print_memory_status(label=""):
    """Log current memory usage with optional CPU temp on RPi."""
    status = get_memory_status()
    cpu_temp = get_cpu_temperature()
    temp_str = f" | CPU: {cpu_temp}°C" if cpu_temp else ""
    label_str = f" [{label}]" if label else ""
    Logger.log("MEMORY", f"{status['percent']}% ({status['used_gb']}GB/{status['total_gb']}GB){temp_str}", "INFO")

def check_memory_critical():
    """Check if memory is getting critically low. Returns True if critical."""
    status = get_memory_status()
    cpu_temp = get_cpu_temperature()
    
    if status['percent'] > 85:
        Logger.log("MEMORY", f"[WARNING] Memory at {status['percent']}% - approaching limit!", "WARN")
        return True
    
    if cpu_temp and cpu_temp > 80:
        Logger.log("THERMAL", f"[WARNING] CPU at {cpu_temp}°C - approaching throttle threshold!", "WARN")
        return True
    
    return False

def print_startup_banner():
    """Display startup information clearly with platform and mode."""
    print("\n" + "="*70)
    print("  SARAH Robot - Autonomous Navigation System")
    print("="*70)
    cpu_temp = get_cpu_temperature()
    temp_info = f"| CPU: {cpu_temp}°C" if cpu_temp else ""
    mem = get_memory_status()
    
    # Show detailed mode and platform info
    platform_indicator = "[Pi]" if IS_RASPBERRY_PI else "[Windows]"
    mode_indicator = "[REMOTE]" if AI_MODE == "remote" else "[LOCAL]"
    ai_info = f"Mode: {AI_MODE.upper()} {mode_indicator}"
    if AI_MODE == "remote":
        ai_info += f" → {REMOTE_AI_SERVER_URL}"
    
    try:
        script_path = os.path.abspath(__file__)
    except Exception:
        script_path = "(unknown)"

    print(f"  Script: {script_path} | Build: {SARAH_BUILD}")
    print(f"  Platform: {PLATFORM_NAME} {platform_indicator} | Memory: {mem['percent']}% {temp_info}")


# ==================== BATTERY MONITORING (OPTIONAL) ====================
# This robot platform may not expose a standard battery percentage API.
# We support a few safe, opt-in methods:
# - SARAH_BATTERY_PERCENT=<0-100>
# - SARAH_BATTERY_PERCENT_FILE=/path/to/file   (file contains an int/float percent)
# - Linux laptops: /sys/class/power_supply/*/capacity (best-effort)
def _read_battery_percent() -> Optional[float]:
    try:
        v = (os.getenv("SARAH_BATTERY_PERCENT") or "").strip()
        if v:
            p = float(v)
            if 0.0 <= p <= 100.0:
                return p
    except Exception:
        pass

    try:
        fp = (os.getenv("SARAH_BATTERY_PERCENT_FILE") or "").strip()
        if fp and os.path.exists(fp):
            raw = (open(fp, "r", encoding="utf-8", errors="ignore").read() or "").strip()
            p = float(raw)
            if 0.0 <= p <= 100.0:
                return p
    except Exception:
        pass

    # Best-effort Linux battery capacity
    try:
        if platform.system() == "Linux":
            for cand in glob.glob("/sys/class/power_supply/*/capacity"):
                try:
                    raw = (open(cand, "r", encoding="utf-8", errors="ignore").read() or "").strip()
                    p = float(raw)
                    if 0.0 <= p <= 100.0:
                        return p
                except Exception:
                    continue
    except Exception:
        pass

    return None


def _battery_level_from_percent(percent: Optional[float]) -> str:
    if percent is None:
        return "unknown"
    low = float(os.getenv("SARAH_BATTERY_LOW_PCT", "25") or "25")
    critical = float(os.getenv("SARAH_BATTERY_CRITICAL_PCT", "10") or "10")
    if percent <= critical:
        return "critical"
    if percent <= low:
        return "low"
    return "ok"


def _which(cmd: str) -> bool:
    """Return True if an executable is available on PATH."""
    try:
        return shutil.which(cmd) is not None
    except Exception:
        return False


def check_pi_runtime_dependencies():
    """Best-effort checks for common missing Pi packages that cause runtime failures."""
    if not IS_RASPBERRY_PI:
        return

    missing_cmds = [cmd for cmd in ("aplay", "arecord", "amixer") if not _which(cmd)]
    if missing_cmds:
        print(f"[INIT] [WARN] Missing ALSA tools: {', '.join(missing_cmds)}")
        print("[INIT]        Install: sudo apt-get update && sudo apt-get install -y alsa-utils")

    if pyaudio is None:
        print("[INIT] [WARN] PyAudio not available; microphone input will not work.")
        print("[INIT]        Install: sudo apt-get install -y python3-pyaudio")

    if not _which("flac"):
        print("[INIT] [WARN] 'flac' not found; speech_recognition may fail with 'FLAC conversion utility not available'.")
        print("[INIT]        Install: sudo apt-get install -y flac")

    # TTS (Piper / fallback)
    if TTS_ENABLED:
        if not (_which("piper")):
            print("[INIT] [WARN] 'piper' command not found; Piper TTS may not work.")
            print("[INIT]        Install (venv): pip install piper-tts")
        if not (_which("espeak-ng") or _which("espeak")):
            print("[INIT] [INFO] espeak not found; Linux TTS fallback unavailable.")
            print("[INIT]        Install: sudo apt-get install -y espeak-ng")

    if not (_which("rpicam-still") or _which("rpicam-jpeg") or _which("libcamera-still")):
        print("[INIT] [WARN] No libcamera capture tool found (rpicam-still/rpicam-jpeg/libcamera-still).")
        print("[INIT]        Install: sudo apt-get install -y libcamera-tools")

    if not GPIOZERO_AVAILABLE:
        print("[INIT] [WARN] gpiozero not available; GPIO motor control will be disabled.")
        print("[INIT]        Install: sudo apt-get install -y python3-gpiozero")


def gpio_backend_diagnostics() -> None:
    """Print detailed GPIO backend diagnostics (Pi-only).

    Enable by setting SARAH_GPIO_DIAG=1. This is intended to make GPIO issues
    (missing modules, missing device nodes, permissions) obvious in one run.
    """
    if not IS_RASPBERRY_PI:
        print("[GPIO-DIAG] Not a Raspberry Pi; skipping.")
        return

    print("\n" + "-" * 70)
    print("[GPIO-DIAG] GPIO backend diagnostics")
    print("-" * 70)
    try:
        print(f"[GPIO-DIAG] Python: {sys.executable}")
        print(f"[GPIO-DIAG] Version: {sys.version.split()[0]}")
    except Exception:
        pass

    try:
        euid = os.geteuid() if hasattr(os, "geteuid") else None
        print(f"[GPIO-DIAG] EUID: {euid}")
    except Exception:
        pass

    try:
        for path in ("/dev/gpiomem", "/dev/mem"):
            print(f"[GPIO-DIAG] exists {path}: {os.path.exists(path)}")
        chips = sorted(glob.glob("/dev/gpiochip*"))
        print(f"[GPIO-DIAG] gpiochips: {len(chips)}")
        if chips:
            print(f"[GPIO-DIAG] sample: {', '.join(chips[:4])}{'...' if len(chips) > 4 else ''}")
    except Exception:
        pass

    try:
        print(f"[GPIO-DIAG] GPIOZERO_PIN_FACTORY env: {os.environ.get('GPIOZERO_PIN_FACTORY','').strip() or '(unset)'}")
        print(f"[GPIO-DIAG] PIGPIO_ADDR env: {os.environ.get('PIGPIO_ADDR','').strip() or '(unset)'}")
        print(f"[GPIO-DIAG] PIGPIO_PORT env: {os.environ.get('PIGPIO_PORT','').strip() or '(unset)'}")
    except Exception:
        pass

    # Import checks
    def _try_import(mod: str) -> str:
        try:
            __import__(mod)
            return "OK"
        except Exception as ex:
            return f"FAIL ({type(ex).__name__}: {ex})"

    print(f"[GPIO-DIAG] import gpiozero: {_try_import('gpiozero')}")
    print(f"[GPIO-DIAG] import lgpio: {_try_import('lgpio')}")
    print(f"[GPIO-DIAG] import pigpio: {_try_import('pigpio')}")
    print(f"[GPIO-DIAG] import RPi.GPIO: {_try_import('RPi.GPIO')}")

    # Suggest next action
    print("[GPIO-DIAG] Next actions:")
    print("[GPIO-DIAG] - If /dev/gpiochip* exists but 'import lgpio' fails: use system python + apt python3-lgpio (Pi OS).")
    print("[GPIO-DIAG] - If /dev/gpiomem is missing and you are not root: try 'sudo -E python sarah_pi.py' (native via /dev/mem).")
    print("[GPIO-DIAG] - If pigpio is selected: you need pigpiod running (often unavailable on non-Pi-OS repos).")
    print("-" * 70 + "\n")


# Optional: allow selecting a specific CSI camera index when multiple are present.
# This is helpful for Arducam ribbon cameras on some setups.
try:
    CSI_CAMERA_INDEX = int(os.environ.get("CSI_CAMERA_INDEX", "-1"))
except Exception:
    CSI_CAMERA_INDEX = -1

# Configure GC threshold once at startup for memory-constrained systems (4GB RPi5)
gc.set_threshold(500, 10, 10)  # More aggressive collection strategy

def optimize_memory():
    """Aggressive garbage collection for memory-constrained systems."""
    gc.collect()  # Force garbage collection immediately

# OLLAMA_SERVER_URL: Set based on platform, but allow env override.
#   export SARAH_OLLAMA_URL="http://localhost:11434"
env_ollama_url = os.environ.get("SARAH_OLLAMA_URL", "").strip()
if env_ollama_url:
    OLLAMA_SERVER_URL = env_ollama_url
else:
    # Windows: connect to localhost, Raspberry Pi: connect to Windows PC IP
    if IS_RASPBERRY_PI:
        OLLAMA_SERVER_URL = REMOTE_AI_SERVER_URL  # Pi connects to Windows PC IP (remote mode default)
    else:
        OLLAMA_SERVER_URL = "http://localhost:11434"  # Windows connects to local Ollama
OLLAMA_API_KEY = None  # Set your API key here if using a remote server (e.g., "sk-your-key-here")
OLLAMA_CONNECTION_TIMEOUT = 15  # seconds to wait for Ollama to respond (increased for network latency)
model = "llama3.1"  # Model name - MUST be defined before verify_ollama_available

# Auto-start Ollama service on Windows
def try_auto_start_ollama():
    """Attempt to auto-start Ollama on Windows if it's not running."""
    if not platform.system() == "Windows":
        return False  # Only auto-start on Windows
    
    print("[INIT] Checking Ollama status...")
    
    # Check if Ollama is already running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("[INIT] [OK] Ollama is already running")
            return True
    except Exception:
        pass
    
    # Try to start Ollama
    try:
        print("[INIT] Attempting to auto-start Ollama...")
        import subprocess
        
        # Check if ollama.exe exists in common locations
        ollama_paths = [
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe",
            os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
        ]
        
        for ollama_path in ollama_paths:
            if os.path.exists(ollama_path):
                print(f"[INIT] Starting Ollama from: {ollama_path}")
                env = os.environ.copy()
                env['OLLAMA_HOST'] = '0.0.0.0:11434'
                subprocess.Popen([ollama_path, "serve"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL,
                                env=env)
                print("[INIT] Waiting for Ollama to start (10 seconds)...")
                time.sleep(10)
                
                # Check if Ollama is now running
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("[INIT] [OK] Ollama successfully started!")
                        return True
                except Exception:
                    pass
                break
        
        # If ollama.exe not found, try shell command
        if not any(os.path.exists(p) for p in ollama_paths):
            print("[INIT] Ollama executable not found in standard locations")
            print("[INIT] Attempting to launch via shell command...")
            try:
                env = os.environ.copy()
                env['OLLAMA_HOST'] = '0.0.0.0:11434'
                subprocess.Popen("ollama serve", shell=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                env=env)
                print("[INIT] Waiting for Ollama to start (10 seconds)...")
                time.sleep(10)
                
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("[INIT] [OK] Ollama successfully started!")
                        return True
                except Exception:
                    pass
            except Exception:
                pass
    except Exception as e:
        print(f"[INIT] [WARN] Could not auto-start Ollama: {e}")
    
    return False

# Auto-start AI server if in remote mode and Windows
def try_auto_start_ai_server():
    """Attempt to auto-start the AI server on Windows if it's not running."""
    if not platform.system() == "Windows":
        return False  # Only auto-start on Windows
    
    # Check localhost first (Flask dev server binds here)
    print("[INIT] Checking if AI server is running...")
    localhost_urls = [
        "http://localhost:5555/health",
        "http://127.0.0.1:5555/health",
    ]
    
    for url in localhost_urls:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"[INIT] [OK] AI server already running at {url}")
                return True
        except Exception:
            pass
    
    # Try remote IP as fallback
    try:
        response = requests.get(f"{REMOTE_AI_SERVER_URL}/health", timeout=2)
        if response.status_code == 200:
            print(f"[INIT] [OK] AI server already running at {REMOTE_AI_SERVER_URL}")
            return True
    except Exception:
        pass
    
    # Try to start the server
    try:
        print(f"[INIT] Attempting to auto-start AI server...")
        import subprocess
        
        # Look for ai_server.py in current and parent directories
        server_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_server.py"),
            "ai_server.py",
            os.path.join(os.getcwd(), "ai_server.py"),
        ]
        
        # Also try the hardcoded path
        server_paths.insert(0, r"c:\Users\dusan\OneDrive - Jacksonville University\SARAH\Code\Use in Pi\ai_server.py")
        
        python_executable = sys.executable
        
        for server_path in server_paths:
            if os.path.exists(server_path):
                print(f"[INIT] Starting AI server from: {server_path}")
                # Start ai_server.py in background
                subprocess.Popen(
                    [python_executable, server_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if platform.system() == "Windows" else 0
                )
                print("[INIT] Waiting for AI server to start (1 second)...")
                time.sleep(1)
                
                # Check if server is now running
                for url in localhost_urls:
                    try:
                        response = requests.get(url, timeout=2)
                        if response.status_code == 200:
                            print(f"[INIT] [OK] AI server successfully started at {url}!")
                            return True
                    except Exception:
                        pass
                break
    except Exception as e:
        print(f"[INIT] Could not auto-start AI server: {e}")
    
    return False

# Initialize AI client (local Ollama or remote HTTP server)
client = None
CLIENT_TYPE = None  # Track which client type we're using

if AI_MODE == "remote":
    print(f"[INIT-REMOTE] Initializing remote AI client on {PLATFORM_NAME}...")
    print(f"[INIT-REMOTE] Server: {REMOTE_AI_SERVER_URL} (Ollama)")
    
    # Remote mode uses Ollama directly (recommended for Pi→Windows):
    # - No ai_server.py required
    # - No ai_client.py required
    # If you *do* have a separate AI server, wire it explicitly elsewhere.

    # Ensure the rest of the code uses the same server URL.
    OLLAMA_SERVER_URL = REMOTE_AI_SERVER_URL

    # Try to auto-start Ollama on Windows when running remote mode locally for testing.
    if platform.system() == "Windows":
        print("[INIT] Checking/starting Ollama on Windows...")
        try_auto_start_ollama()

    try:
        if OLLAMA_API_KEY:
            client = ollama.Client(host=REMOTE_AI_SERVER_URL, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
            CLIENT_TYPE = "OllamaClient (remote with auth)"
        else:
            client = ollama.Client(host=REMOTE_AI_SERVER_URL)
            CLIENT_TYPE = "OllamaClient (remote)"
        print(f"[INIT-REMOTE] [OK] Using remote Ollama: {REMOTE_AI_SERVER_URL}")
        print(f"[INIT-REMOTE] Status: READY")
    except Exception as e:
        print(f"[INIT-REMOTE] [ERROR] Failed to initialize remote Ollama client: {type(e).__name__}: {e}")
        print("[INIT-REMOTE] Falling back to local mode...")
        AI_MODE = "local"
else:
    print(f"[INIT-LOCAL] Initializing local Ollama client on {PLATFORM_NAME}...")
    print(f"[INIT-LOCAL] Server: {OLLAMA_SERVER_URL}")
    
    # Try to auto-start Ollama on Windows
    if platform.system() == "Windows":
        try_auto_start_ollama()
    
    if OLLAMA_API_KEY:
        # Remote server with API key authentication
        client = ollama.Client(host=OLLAMA_SERVER_URL, headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"})
        CLIENT_TYPE = "OllamaClient (remote with auth)"
        print(f"[INIT-LOCAL] Using authenticated remote Ollama: {OLLAMA_SERVER_URL}")
    else:
        # Local Ollama instance (default: localhost:11434 from 'ollama serve')
        client = ollama.Client(host=OLLAMA_SERVER_URL)
        CLIENT_TYPE = "OllamaClient (local)"
        print(f"[INIT-LOCAL] [OK] Using local Ollama: {OLLAMA_SERVER_URL}")
        print(f"[INIT-LOCAL] Status: READY (verify with 'ollama serve' in another terminal)")

# Response cache for frequently repeated queries
class ResponseCache:
    """True LRU cache for model responses to reduce redundant queries."""
    def __init__(self, max_size: int = LLAMA_CACHE_SIZE):
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str):
        """Get cached response, or None if not found. Updates LRU order."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        """Cache a response, evicting least recently used if cache is full."""
        if key in self.cache:
            # Update existing and move to end
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """Clear all cached responses."""
        self.cache.clear()

response_cache = ResponseCache()

def verify_ollama_available():
    """
    Check if Ollama server is available (local or remote).
    Connects directly to the Ollama API endpoint with comprehensive diagnostics.
    """
    # Determine which server to check based on platform and mode
    if IS_RASPBERRY_PI:
        # On Raspberry Pi, connect to remote Ollama on Windows
        server_url = OLLAMA_SERVER_URL
        mode_label = "[REMOTE]"
    else:
        # On Windows, connect to local Ollama
        server_url = OLLAMA_SERVER_URL
        mode_label = "[LOCAL]"
    
    try:
        print(f"[HEALTH] Testing connection to {server_url}...")
        
        # Try with a longer timeout for network diagnostics
        response = requests.get(
            f"{server_url}/api/tags",
            timeout=OLLAMA_CONNECTION_TIMEOUT  # Use the configured timeout for consistency
        )
        
        if response.status_code == 200:
            Logger.log("HEALTH", f"{mode_label}-{PLATFORM_NAME} [OK] Ollama responsive at {server_url}", "SUCCESS")
            if IS_RASPBERRY_PI:
                Logger.log("MODE", f"[REMOTE] Running on Raspberry Pi with Windows Ollama server", "INFO")
            else:
                Logger.log("MODE", f"[LOCAL] Running on {PLATFORM_NAME} with local Ollama", "INFO")
            return True
        else:
            Logger.log("HEALTH", f"{mode_label} [ERROR] Ollama returned status {response.status_code}", "ERROR")
            return False
            
    except requests.exceptions.Timeout:
        Logger.log("HEALTH", f"{mode_label}-{PLATFORM_NAME} [ERROR] Connection timeout to {server_url}", "ERROR")
        if IS_RASPBERRY_PI:
            Logger.log("DIAGNOSTIC", "Timeout connecting to Windows Ollama server. Possible causes:", "WARN")
            Logger.log("DIAGNOSTIC", "  1. Windows firewall blocking port 11434", "WARN")
            Logger.log("DIAGNOSTIC", "  2. Windows PC IP address is incorrect (currently: 100.119.188.18)", "WARN")
            Logger.log("DIAGNOSTIC", "  3. Ollama not running on Windows - START WITH: $env:OLLAMA_HOST='0.0.0.0:11434'; ollama serve", "WARN")
            Logger.log("DIAGNOSTIC", "  4. Ollama not listening on all interfaces (needs OLLAMA_HOST=0.0.0.0:11434)", "WARN")
            Logger.log("DIAGNOSTIC", "  5. Network connectivity issue between Pi and Windows", "WARN")
            Logger.log("DIAGNOSTIC", "Test from Pi: ping 100.119.188.18 && curl http://100.119.188.18:11434/api/tags", "WARN")
            Logger.log("DIAGNOSTIC", "Test from Windows: ipconfig | findstr IPv4", "WARN")
        else:
            Logger.log("FIX", f"[LOCAL] Start Ollama with: ollama serve", "WARN")
        return False
        
    except requests.exceptions.ConnectionError as e:
        Logger.log("HEALTH", f"{mode_label}-{PLATFORM_NAME} [ERROR] Connection refused to {server_url}: {e}", "ERROR")
        if IS_RASPBERRY_PI:
            Logger.log("DIAGNOSTIC", "Cannot connect to Windows Ollama server:", "WARN")
            Logger.log("DIAGNOSTIC", "  1. Start Windows Ollama with: $env:OLLAMA_HOST='0.0.0.0:11434'; ollama serve", "WARN")
            Logger.log("DIAGNOSTIC", "  2. Verify Windows IP is 100.119.188.18 - ipconfig | findstr IPv4", "WARN")
            Logger.log("DIAGNOSTIC", "  3. Check Windows firewall allows port 11434 (inbound)", "WARN")
            Logger.log("DIAGNOSTIC", "  4. Verify Ollama is listening on 0.0.0.0 not 127.0.0.1 - netstat -an | findstr :11434", "WARN")
            Logger.log("DIAGNOSTIC", "  5. Test from Pi: curl http://100.119.188.18:11434/api/tags", "WARN")
        else:
            Logger.log("FIX", f"[LOCAL] Start Ollama with: ollama serve", "WARN")
        return False
        
    except Exception as e:
        Logger.log("HEALTH", f"{mode_label} [ERROR] Connection error: {type(e).__name__}: {e}", "ERROR")
        Logger.log("FIX", f"Verify Ollama is running at {server_url}", "WARN")
        return False

def warmup_model():
    """Pre-load model to avoid first-query delay (or warm up remote connection)."""
    Logger.log("INIT", "Verifying AI connection...", "INFO")
    try:
        if AI_MODE == "remote":
            # Quick connectivity check without loading model
            print(f"[INIT] Testing remote server connection...")
            response = requests.get(f"{REMOTE_AI_SERVER_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                Logger.log("INIT", "[OK] Remote AI server is responsive", "SUCCESS")
                return True
            else:
                Logger.log("INIT", f"Remote server returned status {response.status_code}", "WARN")
                return False
        else:
            # Quick connectivity check for local Ollama
            print(f"[INIT] Testing local Ollama connection...")
            response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                Logger.log("INIT", "[OK] Local Ollama is responsive (models will load on first use)", "SUCCESS")
                return True
            else:
                Logger.log("INIT", f"Ollama returned status {response.status_code}", "WARN")
                return False
    except requests.exceptions.Timeout:
        Logger.log("INIT", "Connection check timed out (may still work when needed)", "WARN")
        return False
    except Exception as e:
        Logger.log("INIT", f"Connection check failed: {e}", "WARN")
        return False

# PyAudio - might not be available on all systems
# NOTE: This must be imported BEFORE USB audio auto-detection below.
try:
    import pyaudio
except ImportError:
    print("[WARN] pyaudio not installed. Install with: pip install pyaudio")
    print("[WARN] On RPi5, use: sudo apt-get install python3-pyaudio")
    pyaudio = None

# Debug toggles
# Set SARAH_DEBUG=1 to enable verbose device listings and camera debug logs.
SARAH_DEBUG = os.environ.get("SARAH_DEBUG", "0").strip() in ("1", "true", "TRUE", "yes", "YES")
AUDIO_DEBUG = SARAH_DEBUG
CAMERA_DEBUG = SARAH_DEBUG
VISION_DEBUG = SARAH_DEBUG


def dprint(enabled: bool, message: str):
    if enabled:
        print(message)

# Voice Debug and Microphone Configuration (must be before MIC detection)
VOICE_DEBUG = True

# Auto-detect USB audio devices on Raspberry Pi
def find_usb_audio_devices():
    """Find USB microphone and speaker indices automatically.
    
    Hardware Configuration:
    - Microphone: USB (returns device index for explicit selection)
    - Speaker: USB (manually configured via PyAudio)
    
    Two-stage detection:
    1. Try PyAudio (fast, but may fail on some systems)
    2. Fall back to ALSA direct detection (slow, but more reliable)
    """
    # IMPORTANT:
    # - Microphone selection needs a PyAudio *device index* (speech_recognition uses PyAudio).
    # - Speaker selection on Linux is more reliable via ALSA *card number* used by `aplay -D hw:<card>,0`.
    usb_mic_index_pyaudio = None
    usb_speaker_index_pyaudio = None
    usb_mic_card_alsa = None
    usb_speaker_alsa = None
    
    # STAGE 1: Try PyAudio first (faster, works on most systems)
    if pyaudio:
        try:
            dprint(AUDIO_DEBUG, "[AUDIO] [Stage 1] Scanning with PyAudio...")
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            dprint(AUDIO_DEBUG, f"[AUDIO] PyAudio found {device_count} devices")
            
            if device_count == 0:
                dprint(AUDIO_DEBUG, "[AUDIO] [WARN] PyAudio found 0 devices! This is unusual.")
                dprint(AUDIO_DEBUG, "[AUDIO] Trying ALSA fallback detection...")
            else:
                usb_mic_index_pyaudio, usb_speaker_index_pyaudio = _detect_devices_pyaudio(p, device_count)
            
            p.terminate()
            
            # If PyAudio found something, return it
            # (Speaker on Linux is still preferably ALSA, so we keep going for speaker.)
            if usb_mic_index_pyaudio is not None:
                dprint(AUDIO_DEBUG, f"[AUDIO] [OK] Selected USB mic via PyAudio index: {usb_mic_index_pyaudio}")
                
        except Exception as e:
            print(f"[AUDIO] PyAudio detection failed: {e}")
            dprint(AUDIO_DEBUG, "[AUDIO] This happens because Jack daemon or other audio issues")
            dprint(AUDIO_DEBUG, "[AUDIO] Switching to ALSA detection (more reliable on Raspberry Pi)...")
    
    # STAGE 2: Fall back to ALSA detection (more reliable)
    dprint(AUDIO_DEBUG, "[AUDIO] [Stage 2] Scanning with ALSA directly...")
    usb_mic_card_alsa, usb_speaker_alsa = _detect_devices_alsa()

    # Final selection rules:
    # - Mic: only use PyAudio device index (ALSA card numbers are NOT valid for speech_recognition).
    # - Speaker: on Linux prefer ALSA card; elsewhere fall back to PyAudio output index.
    final_mic = usb_mic_index_pyaudio

    try:
        system_os = platform.system()
    except Exception:
        system_os = ""

    if system_os == "Linux":
        # IMPORTANT: For Raspberry Pi, forcing hw/plughw devices is often fragile (can be busy/nonexistent).
        # Prefer the system default unless explicitly overridden.
        alsa_override = os.getenv("SARAH_ALSA_SPEAKER_DEVICE", "").strip()
        final_speaker = alsa_override if alsa_override else "default"
    else:
        final_speaker = usb_speaker_index_pyaudio

    speaker_label = "ALSA" if system_os == "Linux" else "PyAudio"
    print(f"[AUDIO] Selected mic={final_mic} (PyAudio index), speaker={final_speaker} ({speaker_label})")
    return final_mic, final_speaker


def _detect_devices_pyaudio(p, device_count):
    """Detect USB devices using PyAudio."""
    usb_mic_index = None
    usb_speaker_index = None
    fallback_mic = None
    fallback_speaker = None
    
    dprint(AUDIO_DEBUG, "[AUDIO] Enumerating PyAudio devices...")
    for i in range(device_count):
        try:
            info = p.get_device_info_by_index(i)
            device_name = info['name'].lower()
            is_usb = 'usb' in device_name or 'uac' in device_name or 'audio' in device_name
            max_input = info['maxInputChannels']
            max_output = info['maxOutputChannels']
            
            usb_label = " [USB]" if is_usb else ""
            dprint(AUDIO_DEBUG, f"  [{i}] {info['name']}{usb_label} | In: {max_input}ch | Out: {max_output}ch")
            
            # Find USB speaker (output device)
            if max_output > 0 and is_usb and usb_speaker_index is None:
                usb_speaker_index = i
                dprint(AUDIO_DEBUG, "       -> Selected for USB speaker output")
            elif max_output > 0 and fallback_speaker is None:
                fallback_speaker = i
            
            # Find USB microphone (input device)
            if max_input > 0 and is_usb and usb_mic_index is None:
                usb_mic_index = i
                dprint(AUDIO_DEBUG, "       -> Selected for USB microphone input")
            elif max_input > 0 and fallback_mic is None:
                fallback_mic = i
                
        except Exception as e:
            dprint(AUDIO_DEBUG, f"[AUDIO] Error checking device {i}: {e}")
            continue
    
    # Use USB devices if available, otherwise use fallback
    final_mic = usb_mic_index if usb_mic_index is not None else fallback_mic
    final_speaker = usb_speaker_index if usb_speaker_index is not None else fallback_speaker
    
    dprint(AUDIO_DEBUG, f"[AUDIO] PyAudio selection - Microphone: {final_mic}, Speaker: {final_speaker}")
    return final_mic, final_speaker


def _detect_devices_alsa():
    """Detect USB devices using ALSA directly (fallback when PyAudio fails)."""
    import subprocess
    
    usb_mic_index = None
    usb_speaker_index = None
    usb_speaker_hw = None
    
    dprint(AUDIO_DEBUG, "[AUDIO] Using ALSA detection (more reliable on Raspberry Pi)...")
    
    try:
        # List speaker devices
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            dprint(AUDIO_DEBUG, "[AUDIO] ALSA Speaker Devices:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    dprint(AUDIO_DEBUG, f"  {line}")
                    # Look for USB/UAC devices - match "card" lines with USB markers
                    line_lower = line.lower()
                    if 'card' in line_lower:
                        has_usb_marker = any(marker in line_lower for marker in ['usb', 'uac', 'audio', 'soundcard'])
                        no_hdmi = 'hdmi' not in line_lower
                        
                        if has_usb_marker and no_hdmi:
                            # Extract card number: "card 2: UACDemoV10"
                            try:
                                card_part = line.split('card')[1].split(':')[0].strip()
                                if card_part.isdigit():
                                    usb_speaker_index = int(card_part)
                                    # Try to extract device number too if present: "device 0:"
                                    try:
                                        dev_num = 0
                                        if 'device' in line_lower:
                                            dev_part = line_lower.split('device', 1)[1].split(':', 1)[0].strip()
                                            dev_num = int(''.join(ch for ch in dev_part if ch.isdigit()) or '0')
                                        usb_speaker_hw = f"hw:{usb_speaker_index},{dev_num}"
                                    except Exception:
                                        usb_speaker_hw = f"hw:{usb_speaker_index},0"
                                    dprint(AUDIO_DEBUG, f"    -> Selected USB speaker device: card {usb_speaker_index}")
                            except (IndexError, ValueError):
                                pass
    except Exception as e:
        print(f"[AUDIO] ALSA speaker detection failed: {e}")
    
    try:
        # List microphone devices
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            dprint(AUDIO_DEBUG, "[AUDIO] ALSA Microphone Devices:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    dprint(AUDIO_DEBUG, f"  {line}")
                    # Look for USB/UAC devices
                    line_lower = line.lower()
                    if 'card' in line_lower:
                        has_usb_marker = any(marker in line_lower for marker in ['usb', 'uac', 'audio', 'soundcard'])
                        no_hdmi = 'hdmi' not in line_lower
                        
                        if has_usb_marker and no_hdmi:
                            # Extract card number
                            try:
                                card_part = line.split('card')[1].split(':')[0].strip()
                                if card_part.isdigit():
                                    usb_mic_index = int(card_part)
                                    dprint(AUDIO_DEBUG, f"    -> Selected USB mic device: card {usb_mic_index}")
                            except (IndexError, ValueError):
                                pass
    except Exception as e:
        print(f"[AUDIO] ALSA microphone detection failed: {e}")
    
    if usb_mic_index is None and usb_speaker_index is None:
        print("[AUDIO] [WARN] No USB audio devices detected by ALSA.")
        print("[AUDIO] This may mean:")
        print("[AUDIO]   - USB device not connected or powered off")
        print("[AUDIO]   - Device not recognized by OS (try: lsusb)")
        print("[AUDIO]   - Falling back to system defaults")
    else:
        if usb_speaker_hw is not None:
            print(f"[AUDIO] [OK] USB speaker detected: {usb_speaker_hw}")
        elif usb_speaker_index is not None:
            print(f"[AUDIO] [OK] USB speaker detected: card {usb_speaker_index}")
        if usb_mic_index is not None:
            print(f"[AUDIO] [OK] USB microphone detected: card {usb_mic_index}")

    speaker_sel = usb_speaker_hw if usb_speaker_hw is not None else usb_speaker_index
    return usb_mic_index, speaker_sel

# Auto-detect USB devices
try:
    MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX = find_usb_audio_devices()
except Exception as e:
    print(f"[AUDIO] Auto-detection failed: {e}")
    MIC_DEVICE_INDEX = None  # Let speech_recognition auto-detect
    SPEAKER_DEVICE_INDEX = None

AUDIO_OUTPUT_FILE = "output.wav"

# Activation word
ACTIVATION_WORD = "sarah"
TTS_ENABLED = True
TTS_NEEDS_ACTIVATION = True  # NEW: TTS only responds after activation word is detected for each prompt

# UX tuning (can be overridden via env vars)
MODE_SELECT_LISTEN_SECONDS = float(os.getenv("SARAH_MODE_SELECT_LISTEN_SECONDS", "4"))
CAMERA_FIRST_FRAME_TIMEOUT_SECONDS = int(os.getenv("SARAH_CAMERA_FIRST_FRAME_TIMEOUT_SECONDS", "15"))

# TTS Configuration (Piper - Neural offline TTS)
TTS_PROVIDER = "piper"  # Neural offline TTS provider
PIPER_VOICE = "en_US-amy-medium"  # High-quality friendly female voice (LibriTTS alternative)
PIPER_MODELS_DIR = os.path.expanduser("~/.local/share/piper/models")  # Where Piper downloads models
TTS_LOCK = threading.Lock()  # Ensure only one response speaks at a time

# Safety: maximum duration allowed for a single movement (seconds)
MAX_MOVE_DURATION = 30

###############################################
# Audio Functions
###############################################

def play_audio_on_device(wav_file: str, device_index):
    """
    Play a WAV file through USB speaker using PyAudio with ALSA fallback.
    
    Tries: PyAudio -> aplay directly -> speaker-test
    """
    import platform
    
    if not os.path.exists(wav_file):
        print(f"[AUDIO] WAV file not found: {wav_file}")
        return False
    
    system_os = platform.system()
    
    if device_index is None or (isinstance(device_index, int) and device_index < 0):
        print(f"[AUDIO] [WARN] No device specified, using system default")
    else:
        print(f"[AUDIO] Playing {wav_file} on device {device_index}")
    
    def _wav_duration_seconds(path: str):
        try:
            with wave.open(path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate() or 0
                if rate <= 0:
                    return None
                return frames / float(rate)
        except Exception:
            return None

    expected_duration = _wav_duration_seconds(wav_file)
    try:
        if expected_duration is not None:
            file_size = os.path.getsize(wav_file)
            print(f"[AUDIO] WAV duration≈{expected_duration:.2f}s size={file_size} bytes")
    except Exception:
        pass

    # On Linux/Raspberry Pi, prefer ALSA (aplay).
    # IMPORTANT: the system `default` device is the most reliable choice on Pi.
    # Use SARAH_ALSA_SPEAKER_DEVICE to explicitly force hw/plughw/etc.
    if system_os == "Linux":
        alsa_candidates = []

        alsa_override = os.getenv("SARAH_ALSA_SPEAKER_DEVICE", "").strip()
        if alsa_override:
            # Explicit override: try exactly what the user requested first.
            alsa_candidates.append(alsa_override)

        # Always try system default first (or right after explicit override).
        alsa_candidates.append("default")

        # Only after default do we try whatever the caller passed (legacy behavior).
        if device_index is not None:
            if isinstance(device_index, str):
                dev = device_index.strip()
                if dev and dev != "default" and dev != alsa_override:
                    if dev.startswith("hw:"):
                        alsa_candidates.append("plughw:" + dev[len("hw:"):])
                    alsa_candidates.append(dev)
            elif isinstance(device_index, int) and device_index >= 0:
                alsa_candidates.append(f"plughw:{device_index},0")
                alsa_candidates.append(f"hw:{device_index},0")

        tried = set()
        for alsa_dev in alsa_candidates:
            if alsa_dev in tried:
                continue
            tried.add(alsa_dev)
            try:
                if alsa_dev == "default":
                    print("[AUDIO] Trying ALSA playback via aplay (system default)...")
                    cmd = ["aplay", "--buffer-time", "500000", "--period-time", "100000", wav_file]
                else:
                    print(f"[AUDIO] Attempting direct ALSA playback via aplay -D {alsa_dev}...")
                    cmd = ["aplay", "--buffer-time", "500000", "--period-time", "100000", "-D", alsa_dev, wav_file]

                start_t = time.time()
                result = subprocess.run(
                    cmd,
                    timeout=120,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                elapsed = time.time() - start_t

                # If aplay exits much earlier than the WAV duration, treat it as a failure and try fallbacks.
                suspicious_short = False
                if expected_duration is not None and expected_duration >= 1.5:
                    # On Pi this tends to manifest as ~1s playback for multi-second WAVs.
                    if elapsed < max(1.0, expected_duration * 0.90):
                        suspicious_short = True

                if result.returncode == 0 and not suspicious_short:
                    if alsa_dev == "default":
                        print("[AUDIO] [OK] Default ALSA playback succeeded")
                    else:
                        print(f"[AUDIO] [OK] ALSA playback succeeded on {alsa_dev}")
                    return True
                else:
                    stderr_snip = (result.stderr or "").strip().replace("\n", " ")
                    if suspicious_short:
                        print(f"[AUDIO] [WARN] aplay exited early on {alsa_dev} (elapsed={elapsed:.2f}s, expected≈{expected_duration:.2f}s); trying fallback...")
                    if stderr_snip:
                        # Keep this concise even when not debugging; it's useful when audio cuts off.
                        print(f"[AUDIO] aplay stderr on {alsa_dev}: {stderr_snip[:180]}")
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                dprint(AUDIO_DEBUG, f"[AUDIO] aplay unavailable/timeout on {alsa_dev}: {e}")
            except Exception as e:
                dprint(AUDIO_DEBUG, f"[AUDIO] ALSA playback exception on {alsa_dev}: {e}")

        if IS_RASPBERRY_PI:
            print("[AUDIO] [ERROR] ALSA playback failed on Raspberry Pi (skipping PyAudio fallback)")
            return False
    
    # Fallback: Try PyAudio on any available device
    if pyaudio and not IS_RASPBERRY_PI:
        try:
            print(f"[AUDIO] Falling back to PyAudio...")
            with wave.open(wav_file, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                
                # Collect output devices
                output_devices = []
                for i in range(device_count):
                    try:
                        info = p.get_device_info_by_index(i)
                        if info['maxOutputChannels'] > 0:
                            output_devices.append((i, info['name']))
                    except (OSError, IndexError):
                        pass
                
                # Try each device
                for try_device_id, try_device_name in output_devices:
                    try:
                        print(f"[AUDIO] Trying PyAudio device [{try_device_id}] {try_device_name}...")
                        wf.rewind()
                        
                        stream = p.open(
                            format=p.get_format_from_width(sample_width),
                            channels=channels,
                            rate=frame_rate,
                            output=True,
                            output_device_index=try_device_id,
                            frames_per_buffer=2048
                        )
                        
                        # Write audio
                        chunk_size = 4096
                        data = wf.readframes(chunk_size)
                        while data:
                            try:
                                stream.write(data, num_frames=len(data)//sample_width//channels)
                            except Exception:
                                stream.write(data)
                            data = wf.readframes(chunk_size)
                        
                        stream.stop_stream()
                        stream.close()
                        print(f"[AUDIO] [OK] PyAudio playback succeeded on device [{try_device_id}]")
                        
                        p.terminate()
                        return True
                        
                    except Exception as e:
                        print(f"[AUDIO] Device [{try_device_id}] failed: {str(e)[:80]}")
                        continue
                
                p.terminate()
        except Exception as e:
            print(f"[AUDIO] PyAudio fallback failed: {e}")
    
    # Last resort: aplay with default device
    if system_os == "Linux":
        try:
            print(f"[AUDIO] Last resort: aplay with system default...")
            result = subprocess.run(
                ["aplay", wav_file],
                timeout=30,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"[AUDIO] [OK] Default ALSA playback succeeded")
                return True
        except Exception as e:
            print(f"[AUDIO] Default ALSA failed: {e}")
    
    print(f"[AUDIO] All playback methods failed")
    return False


def _resolve_dance_song_path() -> str:
    """Resolve the MP3 path for dance mode.

    Default: MP3 sits next to this script on the Pi.
    Override with SARAH_DANCE_SONG (relative to script dir or absolute path).
    """
    default_name = "Yeah Yeah Yeahs - Heads Will Roll (Official Music Video).mp3"
    raw = os.getenv("SARAH_DANCE_SONG", default_name).strip() or default_name
    try:
        if os.path.isabs(raw):
            return raw
    except Exception:
        pass
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()
    return os.path.join(base_dir, raw)


def _start_mp3_playback(mp3_path: str) -> Optional[subprocess.Popen]:
    """Start MP3 playback in the background (best-effort).

    Prefers mpg123 on Linux/Pi. Falls back to ffplay when available.
    Returns a Popen handle or None.
    """
    if not mp3_path:
        return None
    if not os.path.exists(mp3_path):
        print(f"[DANCE] [WARN] MP3 not found: {mp3_path}")
        return None

    candidates: list[list[str]] = []
    try:
        if platform.system() == "Linux":
            if shutil.which("mpg123"):
                candidates.append(["mpg123", "-q", mp3_path])
            if shutil.which("ffplay"):
                candidates.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", mp3_path])
        else:
            if shutil.which("ffplay"):
                candidates.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", mp3_path])
    except Exception:
        pass

    for cmd in candidates:
        try:
            print(f"[DANCE] Starting audio player: {cmd[0]}")
            return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue

    print("[DANCE] [WARN] No MP3 player available. Install `mpg123` (recommended) or `ffmpeg` (ffplay).")
    return None


def _stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()
    except Exception:
        pass


def play_audio_on_device_realtime_mouth_sync(wav_file: str, device_index, avatar: Optional[AvatarDisplay]) -> bool:
    """Play WAV via PyAudio while driving avatar mouth from the exact audio chunks.

    This yields tighter start/stop timing than precomputed envelopes because the
    mouth updates are derived from the same PCM frames sent to the output stream.

    Falls back by returning False (caller can use ALSA/aplay + envelope).
    """
    if avatar is None:
        return False
    if pyaudio is None:
        return False
    if not wav_file or not os.path.exists(wav_file):
        return False

    def _maybe_print_pyaudio_output_devices(p) -> None:
        if os.getenv("SARAH_PYAUDIO_LIST_DEVICES", "").strip() not in ("1", "true", "True", "YES", "yes"):
            return
        try:
            print("[AUDIO] PyAudio output devices (set SARAH_PYAUDIO_SPEAKER_INDEX to choose):")
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    out_ch = int(info.get('maxOutputChannels', 0) or 0)
                    if out_ch <= 0:
                        continue
                    name = str(info.get('name', '') or '')
                    host = str(info.get('hostApi', '') or '')
                    rate = info.get('defaultSampleRate', '')
                    print(f"[AUDIO]   index={i} out_ch={out_ch} rate={rate} name={name} hostApi={host}")
                except Exception:
                    continue
        except Exception as e:
            print(f"[AUDIO] Failed to list PyAudio devices: {type(e).__name__}: {e}")

    # Allow explicit override for PyAudio output device index.
    # This is the most reliable way to route to the correct device.
    env_out = os.getenv("SARAH_PYAUDIO_SPEAKER_INDEX", "").strip()
    out_index_override: Optional[int] = None
    if env_out.isdigit():
        try:
            out_index_override = int(env_out)
        except Exception:
            out_index_override = None

    def _pick_output_device_index(p) -> Optional[int]:
        if out_index_override is not None:
            return out_index_override

        # If caller provided an integer, treat it as a PyAudio output index.
        if isinstance(device_index, int) and device_index >= 0:
            return device_index

        # Otherwise try to pick a reasonable USB-ish output device.
        try:
            preferred = ("usb", "speaker", "audio", "uac")
            avoid = ("hdmi", "bcm", "vc4")
            best = None
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    if int(info.get('maxOutputChannels', 0) or 0) <= 0:
                        continue
                    name = str(info.get('name', '') or '').lower()
                    if any(a in name for a in avoid):
                        continue
                    if any(k in name for k in preferred):
                        return int(i)
                    if best is None:
                        best = int(i)
                except Exception:
                    continue
            return best
        except Exception:
            return None

    try:
        with wave.open(wav_file, 'rb') as wf:
            channels = wf.getnchannels() or 1
            sample_width = wf.getsampwidth() or 2
            rate = wf.getframerate() or 16000

            p = pyaudio.PyAudio()
            try:
                _maybe_print_pyaudio_output_devices(p)
                out_index = _pick_output_device_index(p)

                # Smaller buffers improve A/V sync at the cost of CPU.
                frames_per_chunk = int(os.getenv("SARAH_MOUTH_CHUNK_FRAMES", "512"))
                frames_per_chunk = max(128, min(4096, frames_per_chunk))

                fmt = p.get_format_from_width(sample_width)

                # Real-time RMS normalization with gentle smoothing.
                peak = 1e-6
                mouth = 0.0

                # Tuning knobs (env-overridable)
                gate = float(os.getenv("SARAH_MOUTH_GATE", "0.035"))
                gamma = float(os.getenv("SARAH_MOUTH_GAMMA", "0.60"))
                attack = float(os.getenv("SARAH_MOUTH_ATTACK", "0.65"))
                release = float(os.getenv("SARAH_MOUTH_RELEASE", "0.45"))
                peak_decay = float(os.getenv("SARAH_MOUTH_PEAK_DECAY", "0.995"))

                finished = False

                def _callback(in_data, frame_count, time_info, status_flags):
                    nonlocal peak, mouth, finished
                    try:
                        data = wf.readframes(frame_count)
                    except Exception:
                        data = b""

                    expected_bytes = int(frame_count) * int(channels) * int(sample_width)

                    if not data:
                        finished = True
                        return (b"\x00" * expected_bytes, getattr(pyaudio, "paComplete", 2))

                    # If we got a partial buffer (end-of-file), pad to exact size and mark complete.
                    status = getattr(pyaudio, "paContinue", 0)
                    if len(data) < expected_bytes:
                        finished = True
                        data = data + (b"\x00" * (expected_bytes - len(data)))
                        status = getattr(pyaudio, "paComplete", 2)

                    # Mouth update from exactly what will be played.
                    try:
                        if sample_width == 1:
                            samples = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                            samples = (samples - 128.0) / 128.0
                        elif sample_width == 2:
                            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:
                            samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            samples = None

                        if samples is not None and samples.size:
                            if channels > 1:
                                samples = samples.reshape(-1, channels).mean(axis=1)
                            rms = float(np.sqrt(np.mean(samples * samples)))
                            peak = max(peak * peak_decay, rms, 1e-6)
                            v = (rms / peak) if peak > 0 else 0.0
                            v = max(0.0, min(1.0, v))
                            v = (v ** gamma)
                            if v < gate:
                                v = 0.0

                            if v > mouth:
                                mouth = mouth + attack * (v - mouth)
                            else:
                                mouth = mouth + release * (v - mouth)

                            # Simplified avatar doesn't need mouth animation
                            # Just mark as speaking (handled by set_speaking)
                    except Exception:
                        pass

                    return (data, status)

                stream = p.open(
                    format=fmt,
                    channels=channels,
                    rate=rate,
                    output=True,
                    output_device_index=out_index,
                    frames_per_buffer=frames_per_chunk,
                    stream_callback=_callback,
                )

                try:
                    stream.start_stream()
                    # Block until done.
                    while stream.is_active() and not finished:
                        time.sleep(0.01)
                finally:
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
            finally:
                try:
                    p.terminate()
                except Exception:
                    pass

        # Simplified avatar doesn't need mouth cleanup (handled by end_speaking)
        return True
    except Exception as e:
        dprint(AUDIO_DEBUG, f"[AUDIO] Real-time PyAudio playback failed: {type(e).__name__}: {e}")
        return False


###############################################
# Piper TTS Voice Synthesis
###############################################

def check_system_audio():
    """
    Diagnose system audio configuration and maximize volume.
    Provides platform-specific diagnostics for Windows and Linux.
    USB speaker volume configuration for Raspberry Pi.
    """
    import platform
    import subprocess
    
    print("[AUDIO] Initializing audio (volume + device sanity checks)...")
    system_os = platform.system()
    dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] Operating System: {system_os}")
    
    if system_os == "Linux":
        # Linux/Raspberry Pi audio diagnostics and configuration
        try:
            # Check audio devices
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                dprint(AUDIO_DEBUG, "[AUDIO-DIAG] Available audio devices:")
                dprint(AUDIO_DEBUG, result.stdout)
            else:
                dprint(AUDIO_DEBUG, "[AUDIO-DIAG] Could not list audio devices")
        except Exception as e:
            dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] aplay command failed: {e}")
        
        # Check USB devices specifically
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                found_audio = False
                for line in result.stdout.split('\n'):
                    if any(x in line.lower() for x in ['audio', 'mic', 'speaker', 'usb audio', 'uac']):
                        dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] USB audio: {line.strip()}")
                        found_audio = True
                if not found_audio:
                    dprint(AUDIO_DEBUG, "[AUDIO-DIAG] [INFO] No obvious audio devices in lsusb output")
        except Exception as e:
            dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] lsusb command failed: {e}")
        
        # Avoid printing current mixer state unless debugging
        if AUDIO_DEBUG:
            try:
                result = subprocess.run(['amixer', 'get', 'Master'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    dprint(AUDIO_DEBUG, "[AUDIO-DIAG] Master volume setting:")
                    dprint(AUDIO_DEBUG, result.stdout)
            except Exception as e:
                dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] amixer command failed: {e}")
        
        # Volume: Some USB speakers will clip / protect / shut down briefly at 100%.
        # Default to a safer level; override with SARAH_VOLUME_PERCENT or disable with SARAH_SET_VOLUME=0.
        try:
            set_volume = os.getenv("SARAH_SET_VOLUME", "1").strip().lower() not in ("0", "false", "no")
            vol = int(os.getenv("SARAH_VOLUME_PERCENT", "85"))
            vol = max(0, min(100, vol))
            if set_volume:
                print(f"[AUDIO] Setting Master volume to {vol}%...")
                subprocess.run(['amixer', 'set', 'Master', f'{vol}%'], capture_output=True, timeout=5)
            subprocess.run(['amixer', 'set', 'Master', 'unmute'], capture_output=True, timeout=5)
            dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] [OK] Master volume set to {vol}% and unmuted")
        except Exception as e:
            print(f"[AUDIO] [WARN] Could not set Master volume: {e}")
        
        # Also try to set PCM channel (important for USB devices)
        try:
            result = subprocess.run(['amixer', 'set', 'PCM', f'{vol}%'], capture_output=True, timeout=5, text=True)
            if result.returncode == 0:
                dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] Setting PCM volume to {vol}% (for USB speaker)...")
                subprocess.run(['amixer', 'set', 'PCM', 'unmute'], capture_output=True, timeout=5)
                dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] [OK] PCM volume set to {vol}% and unmuted")
        except Exception as e:
            # PCM might not exist, that's OK
            pass
        
        # Try to set Speaker volume
        try:
            result = subprocess.run(['amixer', 'set', 'Speaker', f'{vol}%'], capture_output=True, timeout=5, text=True)
            if result.returncode == 0:
                dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] Setting Speaker volume to {vol}% (for USB speaker)...")
                subprocess.run(['amixer', 'set', 'Speaker', 'unmute'], capture_output=True, timeout=5)
                dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] [OK] Speaker volume set to {vol}% and unmuted")
        except Exception as e:
            # Speaker might not exist, that's OK
            pass
    
    elif system_os == "Windows":
        # Windows audio diagnostics
        dprint(AUDIO_DEBUG, "[AUDIO-DIAG] Windows detected - using SAPI5 TTS engine")
        
        try:
            # List available audio devices using pyaudio if available
            if pyaudio:
                p = pyaudio.PyAudio()
                device_count = p.get_device_count()
                dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] PyAudio found {device_count} audio devices:")
                for i in range(device_count):
                    info = p.get_device_info_by_index(i)
                    if info['maxOutputChannels'] > 0:  # Output device
                        dprint(AUDIO_DEBUG, f"  [{i}] {info['name']} (channels: {info['maxOutputChannels']})")
                p.terminate()
            else:
                dprint(AUDIO_DEBUG, "[AUDIO-DIAG] PyAudio not available, cannot enumerate devices")
        except Exception as e:
            dprint(AUDIO_DEBUG, f"[AUDIO-DIAG] Could not enumerate audio devices: {e}")
        dprint(AUDIO_DEBUG, "[AUDIO-DIAG] Windows Volume: Adjust in Settings → Sound → Volume")
    
    elif system_os == "Darwin":
        # macOS audio diagnostics
        print("[AUDIO-DIAG] macOS detected - using NSS TTS engine")
        print("[AUDIO-DIAG] Audio will be routed to system default speaker")
    
    else:
        print(f"[AUDIO-DIAG] Unknown OS: {system_os}")

def test_ollama_connection():
    """
    Test if Ollama is running and llama3.1 model is available.
    """
    print("[TEST-OLLAMA] Testing Ollama connection...")
    try:
        with avatar_ai_activity():
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': 'Hi'}],
                stream=False,
                options={"temperature": 0.5}
            )
        print(f"[TEST-OLLAMA] SUCCESS: Ollama responded")
        print(f"[TEST-OLLAMA] Model: {model}")
        print(f"[TEST-OLLAMA] Sample response: {response['message']['content'][:100]}")
        return True
    except ConnectionRefusedError:
        print(f"[TEST-OLLAMA] FAILED: Cannot connect to Ollama at {OLLAMA_SERVER_URL}")
        print(f"[TEST-OLLAMA] Fix: Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"[TEST-OLLAMA] FAILED: {e}")
        print(f"[TEST-OLLAMA] Make sure Ollama is running: ollama serve")
        print(f"[TEST-OLLAMA] Make sure model is pulled: ollama pull {model}")
        return False


def _sanitize_tts_text(text: str) -> str:
    """Remove any explicit facial-emotion metadata from text before TTS.

    The model (especially in voice/chat prompts) sometimes includes emotion metadata like:
      - "emotion: happy"
      - "(emotion: excited)"
      - "Emotion = sad"
    We use emotion for the avatar, but never want it spoken aloud.

    Note: JSON responses are handled separately by `_extract_speak_text_and_emotion()`.
    """
    if text is None:
        return ""
    raw = str(text)
    if not raw.strip():
        return raw

    emotions = r"neutral|happy|excited|thinking|surprised|concerned|sad|listening|proud"

    # Remove inline emotion annotations.
    # Examples: "emotion: happy", "(emotion: happy)", "[Emotion = excited]", "emotion: \"sad\""
    raw = re.sub(
        rf"(?i)\s*[\(\[\{{]?\s*emotion\s*[:=]\s*\"?\s*(?:{emotions})\s*\"?\s*[\)\]\}}]?\s*",
        " ",
        raw,
    )
    # Remove standalone lines like: "Emotion: happy"
    raw = re.sub(rf"(?im)^\s*emotion\s*[:=]\s*(?:{emotions})\s*$\n?", "", raw)

    # Collapse whitespace introduced by removals.
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    return raw

def llama_speak(text: str):
    """
    Use Piper TTS for Llama responses.
    Same as speak() - both use Piper neural TTS for natural-sounding output.
    """
    speak(text)


def _extract_speak_text_and_emotion(text: str):
    """If `text` is a JSON object string with a `speak` field, return (speak_text, emotion).

    This prevents TTS from reading JSON scaffolding like {"command":..., "emotion":...}.
    Returns (None, None) if it doesn't look like parseable JSON.
    """
    try:
        raw = "" if text is None else str(text)
        s = raw.strip()
        if not s:
            return (None, None)

        # Strip Markdown code fences if present.
        if s.startswith("```"):
            # Drop first fence line
            parts = s.splitlines()
            if len(parts) >= 3:
                # Remove leading ```lang and trailing ```
                if parts[0].startswith("```") and parts[-1].strip() == "```":
                    s = "\n".join(parts[1:-1]).strip()

        # Heuristic: only attempt JSON if it resembles an object.
        if not (s.startswith("{") and s.endswith("}")):
            if "{" in s and "}" in s:
                s = s[s.find("{"): s.rfind("}") + 1].strip()
            else:
                return (None, None)

        obj = json.loads(s)
        if not isinstance(obj, dict):
            return (None, None)

        speak_text = obj.get("speak")
        emotion = obj.get("emotion")
        
        # Debug: log when we successfully extract JSON
        if speak_text is not None or emotion is not None:
            dprint(SARAH_DEBUG, f"[JSON-EXTRACT] speak='{str(speak_text)[:50] if speak_text else 'None'}' emotion='{emotion}'")
        
        if speak_text is None:
            return (None, emotion if isinstance(emotion, str) else None)
        speak_text = str(speak_text).strip()
        if not speak_text:
            return (None, emotion if isinstance(emotion, str) else None)
        return (speak_text, emotion if isinstance(emotion, str) else None)
    except Exception as e:
        # Debug: log parse failures
        dprint(SARAH_DEBUG, f"[JSON-EXTRACT] Parse failed: {e}")
        return (None, None)


def speak(text: str):
    """
    Use Piper TTS for high-quality neural text-to-speech.
    Offline, fast, and natural-sounding.
    Falls back to printing if Piper unavailable.
    
    Uses a lock to ensure only one response speaks at a time,
    queuing requests sequentially.
    """
    global _AVATAR
    
    if not TTS_ENABLED:
        print(f"[SPEAK] {text}")
        return

    # If the model returned JSON with a `speak` field, only speak that field.
    # Never speak `emotion`, `command`, etc.
    extracted_speak, extracted_emotion = _extract_speak_text_and_emotion(text)
    if extracted_speak is not None:
        dprint(SARAH_DEBUG, f"[TTS] Extracted speak text from JSON (suppressing emotion/command fields)")
        text = extracted_speak
        # If an explicit emotion is provided, apply it to the avatar (silent; don't speak it).
        if extracted_emotion and _AVATAR:
            try:
                _AVATAR.set_emotion(extracted_emotion)
                dprint(SARAH_DEBUG, f"[AVATAR] Applied emotion '{extracted_emotion}' from JSON (not spoken)")
            except Exception:
                pass

    # Strip any explicit emotion metadata that may appear in plain text.
    text = _sanitize_tts_text(text)
    
    # Clean text for TTS: fix common escaped sequences without destroying legitimate backslashes (e.g., file paths).
    cleaned_text = str(text)
    if "\\" in cleaned_text:
        import re
        # Common JSON/string escapes
        cleaned_text = cleaned_text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
        cleaned_text = cleaned_text.replace("\\\"", '"').replace("\\'", "'")
        cleaned_text = cleaned_text.replace("\\/", "/")
        # Decode \uXXXX and \xNN sequences if present
        def _u(m):
            try:
                return chr(int(m.group(1), 16))
            except Exception:
                return m.group(0)
        cleaned_text = re.sub(r"\\u([0-9a-fA-F]{4})", _u, cleaned_text)
        cleaned_text = re.sub(r"\\x([0-9a-fA-F]{2})", _u, cleaned_text)
        # Collapse doubled backslashes that often appear in JSON dumps
        cleaned_text = cleaned_text.replace("\\\\", "\\")

    # Normalize whitespace
    cleaned_text = " ".join(cleaned_text.split()).strip()
    
    # === AVATAR: Set emotion based on speech content ===
    if _AVATAR:
        try:
            emotion = detect_emotion_from_text(cleaned_text)
            _AVATAR.set_emotion(emotion)
            # Silent: don't print emotion to avoid user confusion
            dprint(SARAH_DEBUG, f"[AVATAR] Emotion set to '{emotion}' from speech content")
        except Exception as e:
            dprint(SARAH_DEBUG, f"[AVATAR] Error setting emotion from speech: {e}")
    
    # Use cleaned text for TTS
    text = cleaned_text
    
    # Acquire lock to ensure sequential TTS (one response at a time)
    with TTS_LOCK:
        # Avatar: eyes open during TTS work + playback.
        if _AVATAR is not None:
            try:
                _AVATAR.set_speaking(True)
            except Exception:
                pass
        try:
            # Generate unique temporary file (use system temp dir)
            import tempfile
            timestamp = int(time.time() * 1000)
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, f"tts_piper_{timestamp}.wav")
            
            # Get Piper voice model path
            voice_model = os.path.expanduser(f"~/.local/share/piper/models/{PIPER_VOICE}.onnx")
            voice_config = os.path.expanduser(f"~/.local/share/piper/models/{PIPER_VOICE}.onnx.json")
            
            if not os.path.exists(voice_model):
                print(f"[WARN] Voice model not found: {voice_model}")
                print(f"[WARN] Download with: python setup_piper_voices.py")
                print(f"[SPEAK] {text}")
                return
            
            # Run Piper TTS using subprocess.
            # Try direct `piper` command first, fallback to `python -m piper`.
            try:
                process = subprocess.Popen(
                    ["piper", "--model", voice_model, "--output_file", output_file, "--length-scale", "1.1", "--noise-scale", "0.5"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except (FileNotFoundError, OSError):
                # Fallback: use python -m piper if command not in PATH
                process = subprocess.Popen(
                    [sys.executable, "-m", "piper", "--model", voice_model, "--output_file", output_file, "--length-scale", "1.1", "--noise-scale", "0.5"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Send text to piper and wait for completion
            try:
                stdout, stderr = process.communicate(input=text.encode(), timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"[WARN] Piper TTS timeout (>30 seconds)")
                print(f"[SPEAK] {text}")
                # Clean up temp file
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except Exception:
                    pass
                return
            
            # Check if synthesis succeeded
            if process.returncode == 0 and os.path.exists(output_file):
                try:
                    print(f"[TTS] Speaking: {text[:50]}...")
                    # If Piper produced a suspiciously short WAV for a longer sentence, warn (helps debug cut-off audio).
                    try:
                        with wave.open(output_file, 'rb') as wf:
                            frames = wf.getnframes()
                            rate = wf.getframerate() or 0
                            dur = frames / float(rate) if rate > 0 else None
                        if dur is not None and dur < 1.2 and len(text.strip()) >= 25:
                            print(f"[TTS] [WARN] Generated very short audio (≈{dur:.2f}s) for a longer sentence")
                    except Exception:
                        pass

                    # Option B: real-time mouth sync driven by the actual audio chunks.
                    # IMPORTANT (RPi/Linux): PyAudio may try to use JACK and can segfault when JACK isn't running.
                    # Default to envelope on Linux for stability; allow realtime only if explicitly enabled.
                    try:
                        _os_name = platform.system()
                    except Exception:
                        _os_name = ""
                    default_mouth = "envelope" if _os_name == "Linux" else ("realtime" if AVATAR_ENABLED else "envelope")
                    mouth_mode = os.getenv("SARAH_TTS_MOUTH_SYNC", default_mouth).strip().lower()
                    allow_rt = os.getenv("SARAH_TTS_ALLOW_PYAUDIO_MOUTH_SYNC", "0").strip().lower() in ("1", "true", "yes")
                    if _os_name == "Linux" and mouth_mode in ("realtime", "rt", "chunk") and (not allow_rt):
                        dprint(AUDIO_DEBUG, "[TTS] Realtime mouth sync disabled on Linux (set SARAH_TTS_ALLOW_PYAUDIO_MOUTH_SYNC=1 to enable)")
                        mouth_mode = "envelope"

                    played_ok = False
                    if mouth_mode in ("realtime", "rt", "chunk") and _AVATAR is not None:
                        try:
                            if SPEAKER_DEVICE_INDEX is not None:
                                print(f"[TTS] Real-time mouth sync (PyAudio). Speaker={SPEAKER_DEVICE_INDEX}")
                                played_ok = play_audio_on_device_realtime_mouth_sync(output_file, SPEAKER_DEVICE_INDEX, _AVATAR)
                            else:
                                print(f"[TTS] Real-time mouth sync (PyAudio). Speaker=default")
                                played_ok = play_audio_on_device_realtime_mouth_sync(output_file, -1, _AVATAR)
                        except Exception:
                            played_ok = False

                    if not played_ok:
                        # Fallback: precomputed envelope + ALSA/aplay playback.
                        if _AVATAR is not None:
                            try:
                                _AVATAR.begin_mouth_sync_from_wav(output_file)
                            except Exception:
                                pass

                        # Play the generated WAV file on speaker device
                        # CRITICAL: Pass SPEAKER_DEVICE_INDEX to ensure USB speaker is used
                        if SPEAKER_DEVICE_INDEX is not None:
                            print(f"[TTS] Using configured speaker device: {SPEAKER_DEVICE_INDEX}")
                            play_audio_on_device(output_file, SPEAKER_DEVICE_INDEX)
                        else:
                            print(f"[TTS] [WARN] SPEAKER_DEVICE_INDEX is None - audio may not play!")
                            print(f"[TTS] Troubleshooting: Check USB speaker is connected and detected")
                            play_audio_on_device(output_file, -1)  # Try all devices
                finally:
                    if _AVATAR is not None:
                        try:
                            _AVATAR.end_speaking()
                        except Exception:
                            pass
                    # Clean up temporary file
                    try:
                        keep_wav = os.getenv("SARAH_KEEP_TTS_WAV", "0").strip().lower() in ("1", "true", "yes")
                        if keep_wav:
                            print(f"[TTS] Keeping WAV for debugging: {output_file}")
                        else:
                            if os.path.exists(output_file):
                                os.remove(output_file)
                    except Exception:
                        pass
            else:
                # Piper failed, fall back to printing
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"[WARN] Piper TTS synthesis failed: {error_msg}")

                # Linux fallback: speak via espeak/espeak-ng if available.
                if platform.system() == "Linux":
                    try:
                        import shutil
                        espeak_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
                        if espeak_cmd:
                            subprocess.run([espeak_cmd, text], timeout=30)
                            # Clean up temp file
                            try:
                                if os.path.exists(output_file):
                                    os.remove(output_file)
                            except Exception:
                                pass
                            return
                    except Exception:
                        pass

                print(f"[SPEAK] {text}")
                # Clean up temp file
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except Exception:
                    pass
                
        except FileNotFoundError:
            print(f"[WARN] Piper TTS not found. Install with: pip install piper-tts")
            print(f"[WARN] Then run: python setup_piper_voices.py")
            # Linux fallback
            if platform.system() == "Linux":
                try:
                    import shutil
                    espeak_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
                    if espeak_cmd:
                        subprocess.run([espeak_cmd, text], timeout=30)
                        return
                except Exception:
                    pass
            print(f"[SPEAK] {text}")
        except subprocess.TimeoutExpired:
            print(f"[WARN] Piper TTS timeout (>30 seconds)")
            print(f"[SPEAK] {text}")
        except Exception as e:
            print(f"[WARN] TTS error: {e}")
            print(f"[SPEAK] {text}")
        finally:
            if _AVATAR is not None:
                try:
                    _AVATAR.end_speaking()
                except Exception:
                    pass



###############################################
# Camera Input Analysis
###############################################
class CameraAnalyzer:
    """
    Handles camera frame analysis using Llava.
    Provides scene description and obstacle detection.
    """
    def __init__(self, camera_thread: 'CameraThread'):
        self.camera_thread = camera_thread
        self.last_analysis = None
        self.analysis_lock = threading.Lock()

    def analyze_scene(self, img_b64: str | None = None):
        """
        Analyze current camera frame for environment and obstacles.
        Uses low temperature (0.05) to prevent hallucinations.
        Returns a dict with scene analysis.
        """
        dprint(VISION_DEBUG, "[CAM-ANALYSIS] Starting scene analysis...")
        dprint(VISION_DEBUG, f"[CAM-ANALYSIS] AI_MODE: {AI_MODE} | Client: {type(client).__name__}")
        
        if not img_b64:
            img_b64 = self.camera_thread.get_frame_base64()
        if not img_b64:
            dprint(VISION_DEBUG, "[CAM-ANALYSIS] No camera frame available")
            # If the camera is unavailable, do NOT force STOP.
            # Ultrasonic sensors + execution-time safety clamps still handle collision avoidance,
            # and returning STOP here would freeze autonomous mode indefinitely.
            return {
                "obstacles": False,
                "confidence": 0.0,
                "description": "No camera view",
                "recommendation": "PROCEED",
                "distance_estimate": "clear",
                "distance_cm": 100,  # Default safe distance when no camera
                "clear_left": True,
                "clear_right": True,
                "clear_forward": True,
            }

        dprint(VISION_DEBUG, f"[CAM-ANALYSIS] Got frame (base64 bytes: {len(img_b64)})")

        def _ollama_chat_http(server_url: str, model_name: str, messages: list, options: dict, timeout_s: float) -> dict:
            import urllib.request

            url = (server_url or "").rstrip("/") + "/api/chat"
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
            }
            if options:
                payload["options"] = options
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_s))) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="ignore"))

        try:
            prompt = (
                "CRITICAL: You are analyzing a robot's forward camera view for navigation and obstacle detection.\n"
                "\n"
                "ANALYZE:\n"
                "1. DISTANCE to obstacles (estimate in cm): very_close <20cm, close 20-50cm, medium 50-100cm, far >100cm\n"
                "2. OBSTACLES: Real physical objects directly ahead (walls, furniture, objects) that would block the robot's path\n"
                "3. CLEAR DIRECTIONS: Which directions (left/right/forward) have OPEN SPACE for movement?\n"
                "4. IGNORE: Shadows, floor markings, reflections, lighting changes, distant objects, background items\n"
                "\n"
                "IMPORTANT RULES:\n"
                "- Only report obstacles=true if you see a physical object DIRECTLY in the path\n"
                "- Empty floor, carpet, or open space = NOT an obstacle\n"
                "- If you see mostly floor/carpet = clear path = obstacles=false\n"
                "- Walls/furniture only count if they're directly ahead blocking movement\n"
                "- When uncertain, default to obstacles=false (sensor fusion will handle safety)\n"
                "\n"
                "DISTANCE ESTIMATION GUIDE:\n"
                "- Object fills most of frame = very_close (<20cm)\n"
                "- Can see clear details/texture = close (20-50cm)\n"
                "- Object visible but smaller = medium (50-100cm)\n"
                "- Object small in frame = far (>100cm)\n"
                "- Open space, no objects in path = clear\n"
                "\n"
                "REQUIRED JSON RESPONSE:\n"
                "{\"obstacles\": true/false, \"distance_estimate\": \"very_close/close/medium/far/clear\", \"distance_cm\": <number>, "
                "\"description\": \"detailed scene description\", \"clear_left\": true/false, \"clear_right\": true/false, \"clear_forward\": true/false, "
                "\"confidence\": 0.0-1.0, \"recommendation\": \"STOP/TURN_LEFT/TURN_RIGHT/PROCEED\"}\n"
                "\n"
                "If uncertain or see open floor, report obstacles=false and clear paths. Ultrasonic sensors will provide additional safety."
            )
            
            # Hard budget so vision can't stall the robot for 20-30s.
            # In autonomous we want fluid motion; outside autonomous allow slightly longer.
            try:
                in_auto = (get_current_mode() == "autonomous")
            except Exception:
                in_auto = False
            try:
                vision_budget_s = float(os.getenv("SARAH_AUTO_VISION_BUDGET_S", "2.0" if in_auto else "8.0") or ("2.0" if in_auto else "8.0"))
            except Exception:
                vision_budget_s = 2.0 if in_auto else 8.0
            vision_budget_s = max(0.8, float(vision_budget_s))

            dprint(VISION_DEBUG, "[CAM-ANALYSIS] Querying Llava via HTTP...")
            with avatar_ai_activity():
                response = _ollama_chat_http(
                    OLLAMA_SERVER_URL,
                    "llava",
                    [{'role': 'user', 'content': prompt, 'images': [img_b64]}],
                    {"temperature": 0.05, "num_predict": 75, "num_ctx": 1024},
                    vision_budget_s,
                )
            analysis_text = ((response.get('message', {}) or {}).get('content', '') or '').strip()
            dprint(VISION_DEBUG, f"[CAM-ANALYSIS] Analysis complete, response length: {len(analysis_text)}")
            
            dprint(VISION_DEBUG, f"[CAM-ANALYSIS] Raw response: {analysis_text[:200] if analysis_text else 'EMPTY'}")
            
            if not analysis_text or analysis_text.strip() == '':
                print("[CAM-ANALYSIS] ERROR: Got empty response from AI model!")
                return {"obstacles": False, "description": "AI model returned empty response", "recommendation": "PROCEED"}
            
            # Try to extract JSON
            match = re.search(r"\{.*\}", analysis_text, flags=re.DOTALL)
            if match:
                try:
                    analysis = json.loads(match.group(0))
                    with self.analysis_lock:
                        self.last_analysis = analysis
                    dprint(VISION_DEBUG, f"[CAM-ANALYSIS] Parsed JSON: {analysis}")
                    return analysis
                except json.JSONDecodeError as e:
                    dprint(VISION_DEBUG, f"[CAM-ANALYSIS] JSON parse error: {e}")

            # Fallback parsing - be conservative but not overly aggressive
            obstacles = "obstacle" in analysis_text.lower() or "blocked" in analysis_text.lower()
            return {
                "obstacles": obstacles,
                "description": analysis_text,
                "clear_left": "left" in analysis_text.lower() or "clear" in analysis_text.lower(),
                "clear_right": "right" in analysis_text.lower() or "clear" in analysis_text.lower(),
                "clear_forward": "forward" in analysis_text.lower() or "ahead" in analysis_text.lower() or "clear" in analysis_text.lower(),
                "recommendation": "PROCEED",  # Default to proceed, let sensors handle safety
                "confidence": 0.3,
                "distance_estimate": "medium",
                "distance_cm": 60
            }
        except Exception as e:
            error_msg = f"Unable to analyze scene: {e}"
            print(f"[CAM-ANALYSIS] CRITICAL ERROR: {error_msg}")
            print(f"[CAM-ANALYSIS] Make sure Llama/Llava model is available")
            print(f"[CAM-ANALYSIS] For local mode: ollama pull llava")
            # FIXED: Don't force STOP on vision errors - let ultrasonic sensors handle safety
            return {
                "obstacles": False,
                "description": error_msg,
                "recommendation": "PROCEED",
                "confidence": 0.0,
                "clear_left": True,
                "clear_right": True,
                "clear_forward": True,
                "distance_estimate": "medium",
                "distance_cm": 60
            }
    
    def get_last_analysis(self):
        """Get the most recent scene analysis."""
        with self.analysis_lock:
            return self.last_analysis if self.last_analysis else {"obstacles": False, "description": "No analysis available"}


###############################################
# Model / Ollama Command Distribution
###############################################
def query_ollama_for_command(system_prompt: str, user_prompt: str, image_data: str = None, camera_analysis: dict = None, timeout: int = None):
    """
    Query AI model (local or remote) to generate movement commands.
    Integrates camera analysis for informed decision-making.
    Returns a command dict with movement instructions.
    OPTIMIZED: Uses caching, output limits, and low temperature for faster responses.
    timeout: Override default timeout (for autonomous use shorter timeouts)
    """
    if timeout is None:
        timeout = AUTONOMOUS_DECISION_TIMEOUT if image_data else LLAMA_RESPONSE_TIMEOUT
    
    model_to_use = "llava" if image_data else model

    def _ollama_chat_http(server_url: str, model_name: str, messages: list, options: dict, timeout_s: float) -> dict:
        """Call Ollama's /api/chat over HTTP with a hard socket timeout."""
        import urllib.request

        url = (server_url or "").rstrip("/") + "/api/chat"
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_s))) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8", errors="ignore"))
    
    # Enhance prompt with camera analysis if available
    enhanced_prompt = user_prompt
    if camera_analysis:
        obstacles_detected = camera_analysis.get('obstacles', False)
        confidence = camera_analysis.get('confidence', 0.0)
        recommendation = camera_analysis.get('recommendation', 'PROCEED')
        description = camera_analysis.get('description', 'No description')
        distance_est = camera_analysis.get('distance_estimate', None)
        distance_cm = camera_analysis.get('distance_cm', None)
        clear_left = camera_analysis.get('clear_left', None)
        clear_right = camera_analysis.get('clear_right', None)
        clear_forward = camera_analysis.get('clear_forward', None)
        obstacle_size = camera_analysis.get('obstacle_size', None)
        
        # Extract ultrasonic sensor data if present in camera_analysis (merged by autonomous loop)
        # NOTE: Sensor 1 ("center") is mounted on the BACK of the robot.
        sensor_left = camera_analysis.get('ultrasonic_left_cm', None)   # front-left
        sensor_back = camera_analysis.get('ultrasonic_center_cm', None) # back
        sensor_right = camera_analysis.get('ultrasonic_right_cm', None) # front-right
        
        # Build sensor fusion context
        sensor_context = ""
        if sensor_left is not None or sensor_back is not None or sensor_right is not None:
            # Format sensor values properly (None -> N/A, numbers -> show with cm)
            # FIXED: Also filter out negative values (invalid readings)
            try:
                left_str = f"{sensor_left:.1f}" if sensor_left is not None and float(sensor_left) > 0 else "N/A"
                back_str = f"{sensor_back:.1f}" if sensor_back is not None and float(sensor_back) > 0 else "N/A"
                right_str = f"{sensor_right:.1f}" if sensor_right is not None and float(sensor_right) > 0 else "N/A"
            except (TypeError, ValueError):
                left_str = back_str = right_str = "N/A"

            # Front distance for camera/safety fusion: use min(front-left, front-right)
            front_vals = []
            try:
                if sensor_left is not None and float(sensor_left) > 0:
                    front_vals.append(float(sensor_left))
                if sensor_right is not None and float(sensor_right) > 0:
                    front_vals.append(float(sensor_right))
            except Exception:
                front_vals = []
            front_min = min(front_vals) if front_vals else None
            front_min_str = f"{front_min:.1f}" if front_min is not None else "N/A"
            
            sensor_context = (
                f"ULTRASONIC SENSORS (ground truth, always trust):\n"
                f"  Front min: {front_min_str}cm\n"
                f"  Left (front): {left_str}cm\n"
                f"  Center (back): {back_str}cm\n"
                f"  Right (front): {right_str}cm\n"
            )
            # Add fusion warning if vision and sensors disagree significantly
            try:
                if front_min is not None and distance_cm is not None:
                    sensor_val = float(front_min)
                    vision_val = float(distance_cm)
                    if sensor_val > 0 and vision_val > 0:
                        diff = abs(sensor_val - vision_val)
                        if diff > 20:
                            sensor_context += f"  WARNING: Vision estimate ({vision_val:.0f}cm) differs from ultrasonic ({sensor_val:.1f}cm) by {diff:.0f}cm - TRUST ULTRASONIC!\n"
            except (TypeError, ValueError, AttributeError):
                pass  # Skip warning if conversion fails
        
        # Inject camera analysis as critical context
        enhanced_prompt = (
            f"[CRITICAL SENSOR FUSION ANALYSIS]\n"
            f"{sensor_context}"
            f"CAMERA VISION:\n"
            f"  Obstacles detected: {obstacles_detected}\n"
            f"  Confidence: {confidence:.0%}\n"
            f"  Recommendation: {recommendation}\n"
            f"  Distance estimate: {distance_est}\n"
            f"  Distance cm: {distance_cm}\n"
            f"  Clear L/R/F: {clear_left}/{clear_right}/{clear_forward}\n"
            f"  Obstacle size: {obstacle_size}\n"
            f"  Details: {description}\n"
            f"[END CRITICAL]\n\n"
            f"{user_prompt}"
        )
        print(f"[MODEL] Sensor fusion injected - Vision: {obstacles_detected}, Ultrasonic front_min: {camera_analysis.get('ultrasonic_front_min_cm', None)}cm, Rec: {recommendation}")

    # Cache policy:
    # - Autonomy decisions must be fresh. Default is to DISABLE caching in autonomous.
    # - When enabled, include sensor+vision context by hashing the enhanced prompt.
    cache_enabled = False
    is_autonomous = False
    try:
        is_autonomous = (get_current_mode() == "autonomous")
    except Exception:
        is_autonomous = False

    if not image_data:
        if is_autonomous:
            cache_enabled = os.getenv("SARAH_AUTO_MODEL_CACHE", "0").strip().lower() in ("1", "true", "yes")
        else:
            cache_enabled = os.getenv("SARAH_MODEL_CACHE", "1").strip().lower() not in ("0", "false", "no")

    cache_key = None
    if (not image_data) and cache_enabled:
        try:
            key_source = f"{model_to_use}\n{system_prompt}\n{enhanced_prompt}"
            h = hashlib.sha1(key_source.encode("utf-8", errors="ignore")).hexdigest()[:16]
            cache_key = f"cmd:{model_to_use}:{h}"
            cached = response_cache.get(cache_key)
            if cached:
                print("[MODEL] Using cached response")
                return cached
        except Exception:
            cache_key = None
    
    try:
        dprint(SARAH_DEBUG, f"[MODEL] Querying {model_to_use} (image={bool(image_data)})...")
        dprint(SARAH_DEBUG, f"[MODEL] AI_MODE={AI_MODE}, Client type={type(client).__name__}")
        
        # IMPORTANT: Enforce a hard timeout by using HTTP directly.
        # This prevents 20-30s hangs when Ollama/LLM generation slows down.
        try:
            dprint(SARAH_DEBUG, f"[MODEL] Sending to Ollama via HTTP (image={bool(image_data)})...")
            dprint(SARAH_DEBUG, f"[MODEL] Server URL: {OLLAMA_SERVER_URL}")
            dprint(SARAH_DEBUG, f"[MODEL] Prompt length: {len(enhanced_prompt)}")

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': enhanced_prompt}
            ]
            if image_data:
                messages[-1]['images'] = [image_data]
                dprint(SARAH_DEBUG, f"[MODEL] Image size: {len(image_data)} bytes")

            # Use token caps consistently.
            # - Vision: fixed small budget
            # - Movement/command generation: always use MOVE cap (separate from chat)
            if image_data:
                max_tokens = 75
            else:
                max_tokens = int(LLAMA_MAX_OUTPUT_TOKENS_MOVE)
            opts = {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 1024}

            with avatar_ai_activity():
                response = _ollama_chat_http(OLLAMA_SERVER_URL, model_to_use, messages, opts, float(timeout or 5))

            response_text = ""
            try:
                response_text = (response.get('message', {}) or {}).get('content', '')
            except Exception:
                response_text = ""
            response_text = (response_text or "").strip()

            dprint(SARAH_DEBUG, f"[MODEL] Response received: {len(response_text)} bytes")
            dprint(SARAH_DEBUG, f"[MODEL] Response: {response_text[:150]}...")

            cmd = parse_command_from_response(response_text)

            # ENFORCE VISION SAFETY: Override if vision says STOP *and* it detected obstacles.
            if (
                camera_analysis
                and str(camera_analysis.get('recommendation', '')).strip().upper() == 'STOP'
                and bool(camera_analysis.get('obstacles', False))
            ):
                if cmd.get('command') != 'STOP':
                    print(f"[SAFETY] Vision recommends STOP - overriding model command {cmd.get('command')} with STOP")
                    cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Vision detected obstacles - stopping!"}
                
            if (not image_data) and cache_enabled and cache_key:
                response_cache.set(cache_key, cmd)
            return cmd
        except Exception as remote_err:
            print(f"[MODEL] Ollama HTTP error: {type(remote_err).__name__}: {remote_err}")
            print(f"[MODEL] Quick test: curl {OLLAMA_SERVER_URL}/api/tags")
            return {"command": "STOP", "duration": 0, "speed": 0, "speak": f"AI error"}

            # Try to extract JSON from response
            match = re.search(r"\{.*?\}", str(text), flags=re.DOTALL)
            if match:
                json_text = match.group(0)
                try:
                    cmd = json.loads(json_text)
                    if 'speak' not in cmd:
                        cmd['speak'] = f"Executing {cmd.get('command', 'STOP').lower()}"
                    
                    # ENFORCE VISION SAFETY: Override if vision says STOP *and* it detected obstacles.
                    if (
                        camera_analysis
                        and str(camera_analysis.get('recommendation', '')).strip().upper() == 'STOP'
                        and bool(camera_analysis.get('obstacles', False))
                    ):
                        if cmd.get('command') != 'STOP':
                            print(f"[SAFETY] Vision recommends STOP - overriding model command {cmd.get('command')} with STOP")
                            cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Vision detected obstacles - stopping!"}
                    
                    if (not image_data) and cache_enabled and cache_key:
                        response_cache.set(cache_key, cmd)
                    return cmd
                except json.JSONDecodeError:
                    print(f"[MODEL] JSON decode failed")
            result = parse_freeform_model_response(str(text))
            
            # ENFORCE VISION SAFETY: Override if vision says STOP *and* it detected obstacles.
            if (
                camera_analysis
                and str(camera_analysis.get('recommendation', '')).strip().upper() == 'STOP'
                and bool(camera_analysis.get('obstacles', False))
            ):
                if result.get('command') != 'STOP':
                    print(f"[SAFETY] Vision recommends STOP - overriding parsed command {result.get('command')} with STOP")
                    result = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Vision detected obstacles - stopping!"}
            
            if (not image_data) and cache_enabled and cache_key:
                response_cache.set(cache_key, result)
            return result
    except Exception as e:
        print(f"[MODEL] Query failed: {e}")
        return {"command": "STOP", "duration": 0, "speed": 0, "speak": "Error processing command"}


def parse_freeform_model_response(text: str):
    text = text.lower()
    cmd = {"command": "STOP", "duration": 0, "speed": 0}
    if "forward" in text:
        cmd["command"] = "FORWARD"
    elif "back" in text or "backward" in text:
        cmd["command"] = "BACKWARD"
    elif "left" in text:
        cmd["command"] = "LEFT"
    elif "right" in text:
        cmd["command"] = "RIGHT"
    elif "stop" in text:
        cmd["command"] = "STOP"

    # extract a duration (seconds)
    dur_match = re.search(r"(\d+(?:\.\d+)?)\s*(seconds|second|s)", text)
    if dur_match:
        cmd["duration"] = float(dur_match.group(1))
    else:
        # try single number ("for 2")
        num_match = re.search(r"for\s+(\d+(?:\.\d+)?)", text)
        if num_match:
            cmd["duration"] = float(num_match.group(1))
        else:
            # Default to 1 second for autonomous decisions if no duration specified
            if cmd["command"] != "STOP":
                cmd["duration"] = 1.0

    # speed - parse requested speed from text
    sp_match = re.search(r"(\d{1,3})\s*%", text)
    if sp_match:
        cmd["speed"] = int(sp_match.group(1))
    else:
        # words slow/fast
        if "slow" in text:
            cmd["speed"] = MIN_SPEED  # Use MIN_SPEED (40%) instead of DEFAULT_SPEED // 2
        elif "fast" in text:
            cmd["speed"] = MAX_SPEED  # Use MAX_SPEED (100%)
        else:
            cmd["speed"] = DEFAULT_SPEED  # Use DEFAULT_SPEED (60%)

    # clamp - enforce MIN_SPEED and MAX_SPEED limits
    cmd["duration"] = min(MAX_MOVE_DURATION, float(cmd.get("duration", 0) or 0))
    cmd["speed"] = int(max(MIN_SPEED, min(MAX_SPEED, cmd.get("speed", DEFAULT_SPEED))))
    cmd["speak"] = f"Executing {cmd['command'].lower()}"
    return cmd


def parse_command_from_response(response_text: str):
    """
    Parse movement command from model response text.
    Handles both JSON format and natural language responses.
    Extracts emotion field if AI provides it.
    """
    try:
        # First try to extract JSON
        match = re.search(r"\{[^{}]*\"command\"[^{}]*\}", response_text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if 'command' in data:
                    cmd = {
                        'command': data.get('command', 'STOP').upper(),
                        'duration': float(data.get('duration', 0) or 0),
                        'speed': int(data.get('speed', DEFAULT_SPEED)),
                        'speak': data.get('speak', 'Moving')
                    }

                    # Preserve optional vision-derived fields if the model includes them.
                    # This enables deterministic post-processing (e.g., reduce forward bias)
                    # without needing a second Llava call.
                    for k in (
                        'obstacles',
                        'recommendation',
                        'distance_estimate',
                        'distance_cm',
                        'clear_left',
                        'clear_right',
                        'clear_forward',
                    ):
                        if k in data:
                            cmd[k] = data.get(k)
                    # === NEW: Extract AI-provided emotion ===
                    if 'emotion' in data:
                        emotion = str(data.get('emotion', '')).lower().strip()
                        valid_emotions = ["neutral", "happy", "excited", "thinking", "surprised", "concerned", "sad", "listening", "proud"]
                        if emotion in valid_emotions:
                            cmd['emotion'] = emotion
                            dprint(SARAH_DEBUG, f"[PARSE] AI provided emotion: {emotion}")
                    
                    # Autonomous safety clamp at parse boundary
                    try:
                        if get_current_mode() == "autonomous" and cmd['command'] in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
                            cmd['duration'] = min(float(cmd.get('duration', 0) or 0), AUTONOMOUS_MAX_MOVE_DURATION)
                    except Exception:
                        pass
                    return cmd
            except (json.JSONDecodeError, ValueError):
                pass
    except Exception as e:
        print(f"[PARSE] JSON extraction error: {e}")
    
    # Fallback to natural language parsing
    return parse_freeform_model_response(response_text)


def parse_simple_voice_command(text: str):
    """
    Parse simple direct voice commands without requiring Llama inference.
    Checks for exact command patterns and executes immediately.
    
    Returns: (command_dict or None, is_simple_command: bool)
    """
    text_lower = text.lower().strip()

    # High-priority multi-word commands
    if any(phrase in text_lower for phrase in ("return to start", "return to the start", "go home", "return home")):
        request_return_to_start(source="voice")
        return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Returning to start.'}, True

    # Persistent map/memory commands (Option J)
    # Map stats query
    try:
        if any(phrase in text_lower for phrase in (
            "how much have you explored",
            "how much have you learned",
            "tell me about your map",
            "tell me about the map",
            "what do you remember",
            "map statistics",
            "map stats",
        )):
            # Signal to report map stats; actual stats retrieval happens in chat/response handler.
            # For now, just give a simple acknowledgment and let the user ask in chat mode for details.
            return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Switch to chat mode and ask me for map details.'}, True
    except Exception:
        pass
    
    # Map reset (requires explicit map/nav keywords to avoid accidental resets)
    try:
        wants_reset = any(k in text_lower for k in ("reset", "clear", "forget", "wipe"))
        targets_map = any(k in text_lower for k in ("map", "navigation", "exploration", "autonomous"))
        # Require explicit map/navigation terms to avoid resetting unrelated memory
        if wants_reset and targets_map:
            if any(phrase in text_lower for phrase in (
                "reset map",
                "reset the map",
                "reset map memory",
                "clear map",
                "clear the map",
                "clear map memory",
                "forget map",
                "forget the map",
                "forget what you learned",
                "wipe map",
                "wipe the map",
                "wipe map memory",
                "reset navigation memory",
                "clear navigation memory",
                "reset exploration memory",
                "clear exploration memory",
            )):
                request_reset_map_memory(source="voice")
                return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Okay. I reset my map memory.'}, True
    except Exception:
        pass

    if any(phrase in text_lower for phrase in ("explore", "start exploring", "keep exploring")):
        request_explore(source="voice")
        return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Exploring.'}, True

    # Game mode commands (requires activation phrase like: "sarah, activate game mode")
    if any(phrase in text_lower for phrase in ("stop game mode", "exit game mode", "quit game mode", "end game mode", "stop game", "exit game")):
        exit_game_mode(source="voice")
        return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Stopping game mode.'}, True

    if any(phrase in text_lower for phrase in ("activate game mode", "start game mode", "enter game mode")):
        request_game(source="voice")
        return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Starting game mode.'}, True

    # Dance mode commands
    if any(phrase in text_lower for phrase in ("stop dancing", "stop dance", "end dance", "end dancing", "quit dancing", "stop dance mode", "end dance mode")):
        exit_dance_mode(source="voice")
        return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Stopping dance mode.'}, True

    # Activation keyword: "sarah dance" - can be reactivated anytime
    if any(phrase in text_lower for phrase in ("sarah dance", "sarah, dance", "sarah start dancing", "sarah dance mode")) and ("not" not in text_lower):
        request_dance(source="voice")
        return {'command': 'STOP', 'duration': 0, 'speed': 0, 'speak': 'Starting dance mode.'}, True
    
    # Simple one-word commands
    simple_commands = {
        'stop': {'command': 'STOP', 'duration': 2, 'speed': 0, 'speak': 'Stopping.'},
        'forward': {'command': 'FORWARD', 'duration': 2, 'speed': 100, 'speak': 'Moving forward.'},
        'backward': {'command': 'BACKWARD', 'duration': 2, 'speed': 100, 'speak': 'Moving backward.'},
        'back': {'command': 'BACKWARD', 'duration': 2, 'speed': 100, 'speak': 'Moving backward.'},
        'left': {'command': 'RIGHT', 'duration': 2, 'speed': 100, 'speak': 'Turning left.'},
        'right': {'command': 'LEFT', 'duration': 2, 'speed': 100, 'speak': 'Turning right.'},
        # Speed control commands - all speeds stay within safe operating range (40-100%)
        'reduce speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Reducing speed to 100 percent.'},
        'decrease speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Reducing speed to 100 percent.'},
        'slow down': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Slowing down to 100 percent.'},
        'increase speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Increasing speed to 100 percent.'},
        'speed up': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Speeding up to 100 percent.'},
        'full speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Full speed ahead!'},
        'max speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Maximum speed.'},
        'half speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Half speed at 100 percent.'},
        'quarter speed': {'command': 'SPEED', 'duration': 2, 'speed': 100, 'speak': 'Quarter speed at 100 percent.'},
    }
    
    # Check for exact matches or simple patterns
    for keyword, cmd in simple_commands.items():
        if text_lower == keyword or text_lower.startswith(keyword + ' ') or text_lower.endswith(' ' + keyword):
            # Parse for duration if provided (e.g., "move forward for 5 seconds")
            dur_match = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(seconds?|s)?", text_lower)
            if dur_match:
                cmd['duration'] = min(MAX_MOVE_DURATION, float(dur_match.group(1)))
            
            # Parse for speed if provided (e.g., "reduce speed to 30%")
            speed_match = re.search(r"(?:to\s+)?(\d{1,3})\s*%?", text_lower)
            if speed_match and cmd['command'] == 'SPEED':
                requested_speed = int(speed_match.group(1))
                # Enforce MIN_SPEED to prevent motor stall
                cmd['speed'] = max(MIN_SPEED, min(MAX_SPEED, requested_speed))
            
            return cmd, True
    
    return None, False


###############################################
# Llama3.1 Query with Automatic TTS
###############################################
def query_llama_and_speak(prompt: str, system_prompt: str = None, speak_response: bool = True, activation_detected: bool = False):
    """
    Query Llama3.1 directly and automatically speak the response via TTS.
    OPTIMIZED: Uses caching, timeout, and output limits for faster responses.
    TTS only responds if activation_detected=True (after 'Sarah' is said) or if TTS_NEEDS_ACTIVATION is False.
    
    Args:
        if drive is not None and not silent:
            try:
                print(f"[EXECUTE] GPIO enabled: {getattr(drive, 'use_gpio', False)}")
            except Exception:
                pass
        prompt: User's prompt/question for Llama
        system_prompt: Optional system instructions (default: general assistant)
        speak_response: If True, speak the model response via TTS (subject to activation settings)
        activation_detected: Whether the activation word 'Sarah' was detected in this prompt
    
    Returns:
        The text response from Llama3.1
    """
    
    # Check cache first
    cache_key = f"{prompt}:{system_prompt}"
    cached = response_cache.get(cache_key)
    if cached:
        print(f"[LLAMA] Using cached response")
        # Only speak if activation was detected (or if TTS doesn't require activation)
        should_speak = speak_response and TTS_ENABLED and (activation_detected or not TTS_NEEDS_ACTIVATION)
        if should_speak:
            llama_speak(_sanitize_tts_text(cached))
        return cached
    
    try:
        print(f"[LLAMA] Querying AI model...")

        sys_prompt = system_prompt or "You are a helpful robot assistant."
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': prompt}
        ]

        # Works for both local and remote because `client` is an `ollama.Client(host=...)`.
        with avatar_ai_activity():
            response = client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={"temperature": 0.1, "num_predict": LLAMA_MAX_OUTPUT_TOKENS_CHAT}
            )

        reply = response['message']['content'].strip()
        
        print(f"[LLAMA] Response ({len(reply)} chars)")
        
        # Cache the response
        response_cache.set(cache_key, reply)
        
        # Only speak if activation was detected (or if TTS doesn't require activation)
        should_speak = speak_response and TTS_ENABLED and (activation_detected or not TTS_NEEDS_ACTIVATION)
        if should_speak:
            print(f"[LLAMA] Queuing response for TTS...")
            llama_speak(_sanitize_tts_text(reply))
        else:
            # Activation required but not detected - log response only
            if speak_response and TTS_NEEDS_ACTIVATION and not activation_detected:
                print(f"[LLAMA] Activation word not detected - TTS suppressed for this response")
        
        return reply
        
    except Exception as e:
        print(f"[LLAMA] Query failed: {e}")
        error_msg = "Sorry, I encountered an error."
        should_speak = speak_response and TTS_ENABLED and (activation_detected or not TTS_NEEDS_ACTIVATION)
        if should_speak:
            llama_speak(error_msg)
        return error_msg


###############################################
# Command executor
###############################################
def execute_command(drive, cmd: dict, silent=False):
    global _AVATAR, _ACTIVE_ULTRASONIC_SENSORS
    global _LAST_AUTO_FORWARD_MIDSTOP_TS, _LAST_AUTO_FORWARD_MIDSTOP_CM
    command = cmd.get("command", "STOP").upper()
    duration = float(cmd.get("duration", 0) or 0)
    speed = int(cmd.get("speed", DEFAULT_SPEED))
    # Enforce speed limits for movement commands only.
    if command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))
    else:
        speed = max(0, min(MAX_SPEED, speed))
    speak_msg = cmd.get("speak", f"Executing {command.lower()}")

    # === AVATAR: Set emotion (AI-provided takes priority over automatic) ===
    if _AVATAR:
        try:
            # Priority 1: AI-provided emotion in JSON
            if 'emotion' in cmd:
                ai_emotion = cmd['emotion']
                _AVATAR.set_emotion(ai_emotion)
                # Silent: don't print emotion to avoid user confusion
                dprint(SARAH_DEBUG, f"[AVATAR] Using AI-controlled emotion: '{ai_emotion}'")
            # Priority 2: Automatic movement-based emotion
            elif command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "SPEED"):
                set_emotion_for_movement(command)
        except Exception as e:
            dprint(SARAH_DEBUG, f"[AVATAR] Error setting emotion: {e}")

    # Autonomous safety: per-command hard caps.
    # Forward can be longer for room-scale exploration; turns/backward should remain short.
    if command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT") and duration > 0:
        try:
            if get_current_mode() == "autonomous":
                try:
                    cap_fwd = float(os.getenv("SARAH_AUTO_MAX_FORWARD_SEC", str(AUTONOMOUS_MAX_MOVE_DURATION)).strip() or str(AUTONOMOUS_MAX_MOVE_DURATION))
                except Exception:
                    cap_fwd = float(AUTONOMOUS_MAX_MOVE_DURATION)
                try:
                    cap_back = float(os.getenv("SARAH_AUTO_MAX_BACKWARD_SEC", "1.5").strip() or "1.5")
                except Exception:
                    cap_back = 1.5
                try:
                    cap_turn = float(os.getenv("SARAH_AUTO_MAX_TURN_SEC", "1.2").strip() or "1.2")
                except Exception:
                    cap_turn = 1.2

                cap_fwd = max(0.0, min(float(AUTONOMOUS_MAX_MOVE_DURATION), float(cap_fwd)))
                cap_back = max(0.0, min(float(AUTONOMOUS_MAX_MOVE_DURATION), float(cap_back)))
                cap_turn = max(0.0, min(float(AUTONOMOUS_MAX_MOVE_DURATION), float(cap_turn)))

                cap = cap_fwd if command == "FORWARD" else (cap_back if command == "BACKWARD" else cap_turn)
                duration = max(0.0, min(float(duration), float(cap)))
                cmd["duration"] = duration
        except Exception:
            pass

    # Optional: enforce a minimum duration (helps ensure motors have time to start).
    # NOTE: Default is 0.0s so autonomous can remain short and safe.
    if command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT") and duration > 0:
        try:
            min_move = float(os.environ.get("SARAH_MIN_MOVE_DURATION", "0.0").strip() or "0.0")
        except Exception:
            min_move = 0.0
        # Do not force a minimum in autonomous mode.
        try:
            if get_current_mode() != "autonomous":
                duration = max(min_move, duration)
        except Exception:
            duration = max(min_move, duration)

    if not silent:
        print(f"[EXECUTE] Command: {command}, Speed: {speed}, Duration: {duration}")
    
    # === AVATAR: Override emotion if speech has emotional content ===
    if not silent and _AVATAR and speak_msg:
        try:
            emotion_from_speech = detect_emotion_from_text(speak_msg)
            # Override movement emotion if speech has strong emotional content (not neutral/listening)
            if emotion_from_speech not in ["neutral", "listening"]:
                _AVATAR.set_emotion(emotion_from_speech)
                # Silent: don't print emotion to avoid user confusion
                dprint(SARAH_DEBUG, f"[AVATAR] Emotion override from speech: '{emotion_from_speech}'")
        except Exception as e:
            dprint(SARAH_DEBUG, f"[AVATAR] Error setting speech emotion: {e}")
    
    # Only speak if not in silent mode AND we actually have text to speak.
    if not silent:
        try:
            if speak_msg and str(speak_msg).strip():
                speak(speak_msg)
        except Exception:
            pass

    # Use the provided `drive` object if available, otherwise just print.
    try:
        if drive:
            # If we're on a Pi but GPIO motor control is disabled, make it obvious.
            if IS_RASPBERRY_PI and command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
                try:
                    if not bool(getattr(drive, "use_gpio", False)):
                        print("[EXECUTE] [WARN] Movement command requested but GPIO motor control is DISABLED (simulated drive).")
                        print("[EXECUTE]        Check gpiozero/rpi-lgpio install + permissions, and verify the correct script copy is running.")
                except Exception:
                    pass

            if command == "FORWARD":
                # SMOOTH VARIATION: Apply subtle weaving for natural movement
                weave = cmd.get("weave", 0)  # -10 to +10 for subtle left/right drift
                if weave != 0 and abs(weave) <= 15:
                    # Adjust motor speeds slightly for curved forward motion
                    left_speed = max(MIN_SPEED, min(MAX_SPEED, speed - weave))
                    right_speed = max(MIN_SPEED, min(MAX_SPEED, speed + weave))
                    dprint(SARAH_DEBUG, f"[WEAVE] Forward with drift: L={left_speed}, R={right_speed}")
                    # Use the drive's motors directly for differential speed
                    try:
                        drive.motor_left.forward(left_speed / 100.0)
                        drive.motor_right.forward(right_speed / 100.0)
                    except Exception:
                        drive.forward(speed)  # Fallback to normal forward
                else:
                    drive.forward(speed)
            elif command == "BACKWARD":
                drive.backward(speed)
            elif command == "LEFT":
                drive.left(speed)
            elif command == "RIGHT":
                drive.right(speed)
            elif command == "STOP":
                drive.stop()
            elif command == "SPEED":
                # Speed change command - just update PWM on current motors
                # The current direction is maintained, only speed changes
                if not silent:
                    print(f"[DRIVE] Speed changed to {speed}%")
                # Update speed on both motors regardless of current direction
                try:
                    if hasattr(drive, 'pwm_objects'):
                        for pwm in drive.pwm_objects.values():
                            if pwm:
                                drive._set_motor_speed(pwm, speed)
                except Exception as e:
                    print(f"[DRIVE] Error updating speed: {e}")
        else:
            if command == "FORWARD":
                if not silent:
                    print(f"[DRIVE-SIM] FORWARD @ {speed}%")
            elif command == "BACKWARD":
                if not silent:
                    print(f"[DRIVE-SIM] BACKWARD @ {speed}%")
            elif command == "LEFT":
                if not silent:
                    print(f"[DRIVE-SIM] LEFT @ {speed}%")
            elif command == "RIGHT":
                if not silent:
                    print(f"[DRIVE-SIM] RIGHT @ {speed}%")
            elif command == "STOP":
                if not silent:
                    print("[DRIVE-SIM] STOP")
            elif command == "SPEED":
                if not silent:
                    print(f"[DRIVE-SIM] Speed changed to {speed}%")
    except Exception as e:
        error_msg = f"Error controlling robot: {e}"
        print(f"[EXECUTE] {error_msg}")
        if not silent:
            speak(error_msg)

    if duration > 0 and command != "STOP":
        # For autonomous FORWARD moves, monitor sensors during movement for safety.
        # Split long movements into short check intervals to catch obstacles mid-movement.
        # NOTE: Do NOT monitor during BACKWARD, LEFT, RIGHT - these are escape/avoidance maneuvers.
        try:
            is_autonomous_forward = (command == "FORWARD" and get_current_mode() == "autonomous")
        except Exception:
            is_autonomous_forward = False
        
        if is_autonomous_forward and duration > 0.15:
            # Active monitoring: check sensors every 0.1s during movement
            check_interval = 0.1
            elapsed = 0.0
            emergency_stop_triggered = False
            start_time = time.time()  # Safety: overall timeout to prevent infinite loops
            
            while elapsed < duration and (time.time() - start_time) < (duration + 1.0):
                # If the user switches modes mid-move, stop immediately.
                try:
                    if get_current_mode() != "autonomous":
                        if drive:
                            drive.stop()
                        break
                except Exception:
                    pass
                sleep_time = min(check_interval, duration - elapsed)
                time.sleep(sleep_time)
                elapsed += sleep_time
                
                # Quick sensor check during movement
                try:
                    active_sensors = _ACTIVE_ULTRASONIC_SENSORS
                    if active_sensors:
                        readings = read_all_sensors(active_sensors)
                        # Forward safety should use FRONT sensors only (left/right). Center sensor is mounted on the BACK.
                        front_vals = []
                        try:
                            if readings and len(readings) >= 3:
                                if readings[0] is not None and isinstance(readings[0], (int, float)) and float(readings[0]) > 0:
                                    front_vals.append(float(readings[0]))
                                if readings[2] is not None and isinstance(readings[2], (int, float)) and float(readings[2]) > 0:
                                    front_vals.append(float(readings[2]))
                        except Exception:
                            front_vals = []
                        if front_vals:
                            min_dist = min(front_vals)
                            # Emergency stop if obstacle detected mid-movement.
                            # Keep this as a hard safety floor, but allow env tuning.
                            try:
                                mid_stop_cm = float(os.getenv('SARAH_AUTO_MID_MOVE_STOP_CM', str(ULTRASONIC_STOP_DISTANCE_CM)).strip())
                            except Exception:
                                mid_stop_cm = float(ULTRASONIC_STOP_DISTANCE_CM)
                            if min_dist <= float(mid_stop_cm):
                                if drive:
                                    drive.stop()
                                print(f"[EXECUTE] MID-MOVEMENT STOP: obstacle at {min_dist:.1f}cm (<= {mid_stop_cm:.1f}cm)")
                                try:
                                    _LAST_AUTO_FORWARD_MIDSTOP_TS = time.time()
                                    _LAST_AUTO_FORWARD_MIDSTOP_CM = float(min_dist)
                                except Exception:
                                    pass
                                emergency_stop_triggered = True
                                break
                except Exception:
                    pass  # If sensor check fails, continue movement (safer than crashing)
            
            # If we didn't emergency stop, do the normal stop
            if not emergency_stop_triggered:
                try:
                    if drive:
                        drive.stop()
                    if not silent:
                        print("[EXECUTE] Duration finished, stopping.")
                    else:
                        dprint(SARAH_DEBUG, "[EXECUTE] Duration finished, stopping.")
                except Exception as e:
                    print(f"[EXECUTE] Error stopping after duration: {e}")
        else:
            # Short move or non-autonomous: use simple blocking sleep
            time.sleep(duration)
            # Stop the drive after duration expires
            try:
                if drive:
                    drive.stop()
                if not silent:
                    print("[EXECUTE] Duration finished, stopping.")
                else:
                    dprint(SARAH_DEBUG, "[EXECUTE] Duration finished, stopping.")
            except Exception as e:
                print(f"[EXECUTE] Error stopping after duration: {e}")


    # ARC TURNS (autonomous only): after a turn, nudge forward briefly to create smoother paths.
    # This is guarded by ultrasonic checks and uses execute_command() for the forward move,
    # which preserves the existing mid-movement monitoring and hard duration clamps.
    try:
        if (
            drive
            and command in ("LEFT", "RIGHT")
            and float(duration or 0) > 0
            and get_current_mode() == "autonomous"
            and (not RETURN_TO_START_EVENT.is_set())
        ):
            arc_enabled = os.getenv('SARAH_AUTO_ARC_TURNS', '1').strip().lower() not in ('0', 'false', 'no')
            if arc_enabled:
                try:
                    arc_dur = float(os.getenv('SARAH_AUTO_ARC_FORWARD_DURATION', '0.5').strip())
                except Exception:
                    arc_dur = 0.5
                try:
                    cap_fwd = float(os.getenv("SARAH_AUTO_MAX_FORWARD_SEC", str(AUTONOMOUS_MAX_MOVE_DURATION)).strip() or str(AUTONOMOUS_MAX_MOVE_DURATION))
                except Exception:
                    cap_fwd = float(AUTONOMOUS_MAX_MOVE_DURATION)
                cap_fwd = max(0.0, min(float(AUTONOMOUS_MAX_MOVE_DURATION), float(cap_fwd)))
                arc_dur = max(0.0, min(float(arc_dur), float(cap_fwd)))

                try:
                    arc_min_clear = float(os.getenv('SARAH_AUTO_ARC_FORWARD_MIN_CLEAR_CM', '22').strip())
                except Exception:
                    arc_min_clear = 22.0
                arc_min_clear = max(0.0, float(arc_min_clear))
                try:
                    min_proceed = float(os.getenv('SARAH_AUTO_MIN_PROCEED_CM', '20').strip() or '20')
                except Exception:
                    min_proceed = 20.0
                arc_min_clear = max(float(min_proceed), float(arc_min_clear))

                can_nudge = False
                try:
                    active_sensors = _ACTIVE_ULTRASONIC_SENSORS
                    if active_sensors and arc_dur > 0:
                        readings = read_all_sensors(active_sensors)
                        # Forward clearance must use FRONT sensors only (left/right). Center sensor is mounted on the BACK.
                        front_vals = []
                        try:
                            if readings and len(readings) > 0:
                                v = float(readings[0])
                                if v > 0:
                                    front_vals.append(v)
                        except Exception:
                            pass
                        try:
                            if readings and len(readings) > 2:
                                v = float(readings[2])
                                if v > 0:
                                    front_vals.append(v)
                        except Exception:
                            pass

                        front_min = min(front_vals) if front_vals else None

                        if front_min is None:
                            # No front sensor info => don't auto-nudge.
                            can_nudge = False
                        else:
                            required_clear = max(float(arc_min_clear), float(ULTRASONIC_STOP_DISTANCE_CM) + 2.0)
                            can_nudge = float(front_min) >= required_clear

                        # If we just mid-stopped on a forward move, avoid immediately nudging forward again.
                        try:
                            if (time.time() - float(_LAST_AUTO_FORWARD_MIDSTOP_TS)) < 1.2:
                                can_nudge = False
                        except Exception:
                            pass
                except Exception:
                    can_nudge = False

                if can_nudge and arc_dur > 0:
                    # Silent forward nudge; keep speed within movement limits.
                    arc_cmd = {"command": "FORWARD", "duration": float(arc_dur), "speed": int(speed), "speak": ""}
                    execute_command(drive, arc_cmd, silent=True)
    except Exception:
        pass


###############################################
# Simple Drive abstraction (GPIO optional)
###############################################
class Drive:
    def __init__(self, use_gpio: bool = False):
        self.use_gpio = use_gpio and IS_RASPBERRY_PI  # Only enable GPIO on actual RPi
        self.last_action = None
        self.GPIO = None
        self.pwm_objects = {}  # Store PWM objects for cleanup
        self._speed_mode = "pwm"  # "pwm" or "digital" (enable-only)

        # When running on the Pi, avoid silently falling back to simulated drive.
        # Set SARAH_ALLOW_SIMULATED=1 to permit simulated mode (development/testing only).
        allow_simulated = os.environ.get("SARAH_ALLOW_SIMULATED", "0").strip().lower() in ("1", "true", "yes")

        # Force enable mode:
        # - auto (default): PWM if possible, else digital
        # - pwm: always try PWM (errors if not available)
        # - digital: always use on/off enable pins (full speed only)
        enable_mode = os.environ.get("SARAH_ENABLE_MODE", "auto").strip().lower()
        if enable_mode not in ("auto", "pwm", "digital"):
            enable_mode = "auto"

        def _env_int(name: str) -> Optional[int]:
            try:
                v = os.environ.get(name, "").strip()
            except Exception:
                v = ""
            if not v:
                return None
            try:
                return int(v)
            except Exception:
                return None

        # Optional: allow pin overrides without editing code.
        # Values are BCM GPIO numbers.
        #   SARAH_LEFT_IN1=17 SARAH_LEFT_IN2=27 SARAH_LEFT_ENA=12 ...
        def _get_pin(default_bcm: int, env_name: str) -> int:
            override = _env_int(env_name)
            return int(override) if override is not None else int(default_bcm)

        # Drivetrain note:
        # Many 4WD robot chassis mount the right-side motors mirrored, so the same
        # electrical polarity makes left/right wheels spin opposite directions.
        # This script defaults to inverting the right side so a FORWARD command spins
        # all wheels the same physical direction.
        self._invert_right = True
        
        if self.use_gpio:
            try:
                if not GPIOZERO_AVAILABLE or DigitalOutputDevice is None or PWMOutputDevice is None:
                    raise ImportError("gpiozero not available")

                # Force an explicit pin factory here so gpiozero cannot silently fall back
                # to another backend (e.g., pigpio) during DigitalOutputDevice construction.
                # This is critical on systems where pigpiod is unavailable.
                pin_factory = None
                pin_factory_name = None
                requested = os.environ.get("GPIOZERO_PIN_FACTORY", "").strip().lower()

                def _select_pin_factory():
                    nonlocal pin_factory, pin_factory_name
                    from gpiozero import Device  # type: ignore

                    euid = os.geteuid() if hasattr(os, "geteuid") else None
                    has_gpiomem = os.path.exists("/dev/gpiomem")
                    has_gpiochips = bool(glob.glob("/dev/gpiochip*"))
                    is_pi5 = IS_RASPBERRY_PI and ("Raspberry Pi 5" in PLATFORM_NAME if PLATFORM_NAME else False)

                    def _try_lgpio():
                        from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore

                        return LGPIOFactory(), "lgpio"

                    def _try_native():
                        # Native backend (RPi.GPIO style) has poor Pi 5 support
                        if is_pi5:
                            raise OSError("native backend not recommended for Raspberry Pi 5; use lgpio instead")
                        from gpiozero.pins.native import NativeFactory  # type: ignore

                        return NativeFactory(), "native"

                    def _try_pigpio():
                        import socket
                        from gpiozero.pins.pigpio import PiGPIOFactory  # type: ignore

                        host = (os.environ.get("SARAH_PIGPIO_HOST") or os.environ.get("PIGPIO_ADDR") or "localhost").strip() or "localhost"
                        try:
                            port = int((os.environ.get("SARAH_PIGPIO_PORT") or os.environ.get("PIGPIO_PORT") or "8888").strip())
                        except Exception:
                            port = 8888

                        # Preflight connect so we error with a clear reason.
                        try:
                            with socket.create_connection((host, port), timeout=0.5):
                                pass
                        except Exception as ex:
                            raise OSError(f"pigpio not reachable at {host}:{port}: {ex}")

                        return PiGPIOFactory(host=host, port=port), "pigpio"

                    # If user requested a factory explicitly, honor it and give targeted guidance.
                    if requested in ("lgpio", "native", "pigpio"):
                        if requested == "lgpio":
                            pin_factory, pin_factory_name = _try_lgpio()
                            return
                        if requested == "native":
                            pin_factory, pin_factory_name = _try_native()
                            return
                        if requested == "pigpio":
                            pin_factory, pin_factory_name = _try_pigpio()
                            return

                    # Auto-select:
                    # - Prefer lgpio if gpiochips exist (modern Pi stacks)
                    # - Else try native (may require /dev/gpiomem or root /dev/mem)
                    # - Pigpio last (requires pigpiod; often missing in some repos)
                    errors: list[str] = []
                    if has_gpiochips:
                        try:
                            pin_factory, pin_factory_name = _try_lgpio()
                            return
                        except Exception as ex:
                            errors.append(f"lgpio: {type(ex).__name__}: {ex}")

                    try:
                        pin_factory, pin_factory_name = _try_native()
                        return
                    except Exception as ex:
                        errors.append(f"native: {type(ex).__name__}: {ex}")

                    try:
                        pin_factory, pin_factory_name = _try_pigpio()
                        return
                    except Exception as ex:
                        errors.append(f"pigpio: {type(ex).__name__}: {ex}")

                    diag = (
                        f"requested={requested or 'auto'} euid={euid} /dev/gpiomem={has_gpiomem} /dev/gpiochip*={has_gpiochips} "
                        f"errors={'; '.join(errors) if errors else 'n/a'}"
                    )
                    raise RuntimeError(f"No usable GPIO pin factory found. {diag}")

                _select_pin_factory()
                try:
                    from gpiozero import Device  # type: ignore

                    Device.pin_factory = pin_factory
                    if pin_factory_name:
                        os.environ["GPIOZERO_PIN_FACTORY"] = pin_factory_name
                except Exception:
                    pass
                
                # Store pin_factory for sensor initialization and cleanup
                self.pin_factory = pin_factory
                self.pin_factory_name = pin_factory_name
                
                # Motor pins for L298N drivers (conflict-free pinout)
                # Left Motor (IN1, IN2 for direction; ENA for speed/PWM)
                # Left L298N: Motor A (OUT1=front) + Motor B (OUT4=back)
                self.LEFT_IN1 = _get_pin(17, "SARAH_LEFT_IN1")  # Motor A front
                self.LEFT_IN2 = _get_pin(18, "SARAH_LEFT_IN2")  # Motor A front
                self.LEFT_IN3 = _get_pin(15, "SARAH_LEFT_IN3")  # Motor B back
                self.LEFT_IN4 = _get_pin(27, "SARAH_LEFT_IN4")  # Motor B back
                
                # Right L298N: Motor A (OUT1=front) + Motor B (OUT4=back)
                # Using GPIO12/19/26/6 (physical pins 32/35/37/31)
                self.RIGHT_IN1 = _get_pin(12, "SARAH_RIGHT_IN1")   # Motor A front (physical pin 32)
                self.RIGHT_IN2 = _get_pin(19, "SARAH_RIGHT_IN2")  # Motor A front (physical pin 35)
                self.RIGHT_IN3 = _get_pin(26, "SARAH_RIGHT_IN3")  # Motor B back (physical pin 37)
                self.RIGHT_IN4 = _get_pin(6, "SARAH_RIGHT_IN4")   # Motor B back (physical pin 31)
                # ENA/ENB not used - jumpers are on (always enabled)
                
                # Initialize GPIO Zero output devices - all 8 control pins
                # Use active_high=True and initial_value=False to ensure pins start LOW
                self.left_in1 = DigitalOutputDevice(self.LEFT_IN1, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.left_in2 = DigitalOutputDevice(self.LEFT_IN2, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.left_in3 = DigitalOutputDevice(self.LEFT_IN3, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.left_in4 = DigitalOutputDevice(self.LEFT_IN4, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.right_in1 = DigitalOutputDevice(self.RIGHT_IN1, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.right_in2 = DigitalOutputDevice(self.RIGHT_IN2, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.right_in3 = DigitalOutputDevice(self.RIGHT_IN3, active_high=True, initial_value=False, pin_factory=pin_factory)
                self.right_in4 = DigitalOutputDevice(self.RIGHT_IN4, active_high=True, initial_value=False, pin_factory=pin_factory)
                
                # No PWM objects needed - motors run at full battery voltage
                self._speed_mode = "jumper-always-on"
                self.pwm_objects = {}  # Empty - no enable pins used
                
                print(f"[DRIVE] [OK] GPIO Zero initialized with 2× L298N drivers (4 motors total)")
                print(f"[DRIVE] Mode: Direction-only (ENA/ENB jumpers ON - always enabled)")
                print(f"[DRIVE] Left L298N: Motor A (IN1={self.LEFT_IN1}, IN2={self.LEFT_IN2}) + Motor B (IN3={self.LEFT_IN3}, IN4={self.LEFT_IN4})")
                print(f"[DRIVE] Right L298N: Motor A (IN1={self.RIGHT_IN1}, IN2={self.RIGHT_IN2}) + Motor B (IN3={self.RIGHT_IN3}, IN4={self.RIGHT_IN4})")
                print(f"[DRIVE] Right side: GPIO12/19/26/6 (physical pins 32/35/37/31)")
                print(f"[DRIVE] All 4 motors run at full battery voltage (no PWM speed control)")
                dprint(SARAH_DEBUG, f"[DRIVE] Backend: {pin_factory_name}")
                
            except ImportError:
                msg = "[DRIVE] GPIO Zero not available"
                if allow_simulated or not IS_RASPBERRY_PI:
                    print(msg + "; running simulated drive.")
                    self.use_gpio = False
                else:
                    raise RuntimeError(msg + "; refusing to run simulated motors on Raspberry Pi. Fix gpiozero/rpi-lgpio install.")
            except Exception as e:
                msg = f"[DRIVE] GPIO Zero initialization failed: {type(e).__name__}: {e}"
                if allow_simulated or not IS_RASPBERRY_PI:
                    print(msg + "; running simulated drive.")
                    self.use_gpio = False
                else:
                    # Provide very specific remediation depending on the likely root cause.
                    # Check if this is Pi 5 specifically
                    is_pi5 = IS_RASPBERRY_PI and ("Raspberry Pi 5" in PLATFORM_NAME if PLATFORM_NAME else False)
                    
                    if is_pi5 and "No module named 'lgpio'" in str(e):
                        hint = (
                            "\n[DRIVE] ** RASPBERRY PI 5 DETECTED **\n"
                            "[DRIVE] Pi 5 REQUIRES lgpio backend. Native backend does not work reliably on Pi 5.\n"
                            "[DRIVE] \n"
                            "[DRIVE] REQUIRED: Install system lgpio package:\n"
                            "[DRIVE]    sudo apt-get update\n"
                            "[DRIVE]    sudo apt-get install -y python3-lgpio python3-gpiozero\n"
                            "[DRIVE] \n"
                            "[DRIVE] Then run your script again with sudo:\n"
                            "[DRIVE]    sudo -E /home/liamdusanic/venv/bin/python3 /home/liamdusanic/Documents/sarah_pi.py\n"
                            "[DRIVE] \n"
                            "[DRIVE] The lgpio module MUST be installed via apt (not pip) for Pi 5.\n"
                        )
                    else:
                        hint = (
                            "\n[DRIVE] Common fixes:\n"
                            "[DRIVE] 1) Prefer lgpio on Pi 5 (requires lgpio module):\n"
                            "[DRIVE]    - Raspberry Pi OS: sudo apt-get install -y python3-lgpio python3-gpiozero\n"
                            "[DRIVE]    - Then run using system python/venv with --system-site-packages (often Python 3.11)\n"
                            "[DRIVE]    - export GPIOZERO_PIN_FACTORY=lgpio\n"
                            "[DRIVE] 2) Native backend requires /dev/gpiomem OR root access to /dev/mem:\n"
                            "[DRIVE]    - sudo usermod -aG gpio $USER ; sudo reboot\n"
                            "[DRIVE]    - If /dev/gpiomem does not exist, try: sudo -E python sarah_pi.py\n"
                            "[DRIVE] 3) pigpio backend requires pigpiod (not available on some OS repos).\n"
                            "[DRIVE]    - If you can't install pigpio, do NOT use GPIOZERO_PIN_FACTORY=pigpio\n"
                        )
                    raise RuntimeError(
                        msg + "; refusing to run simulated motors on Raspberry Pi. Fix GPIO backend/permissions." + hint
                    )

    def _set_motor_speed(self, pwm_obj, speed: int):
        """Set motor speed as PWM duty cycle.

        Supports gpiozero's PWMOutputDevice (preferred) and also tolerates
        RPi.GPIO PWM objects if swapped in.
        """
        if not pwm_obj:
            return

        speed = int(max(0, min(100, speed)))

        # gpiozero.PWMOutputDevice API
        if hasattr(pwm_obj, "value"):
            pwm_obj.value = speed / 100.0
            return

        # gpiozero.DigitalOutputDevice API (enable-only fallback)
        if hasattr(pwm_obj, "on") and hasattr(pwm_obj, "off"):
            if speed > 0:
                pwm_obj.on()
            else:
                pwm_obj.off()
            return

        # RPi.GPIO PWM API (ChangeDutyCycle/start)
        if hasattr(pwm_obj, "ChangeDutyCycle"):
            pwm_obj.ChangeDutyCycle(speed)
            if speed > 0 and hasattr(pwm_obj, "start"):
                try:
                    pwm_obj.start(speed)
                except RuntimeError:
                    pass
            return

        raise TypeError(f"Unsupported PWM object type: {type(pwm_obj).__name__}")

    def _apply_motor_speeds(self, left_speed: int, right_speed: int):
        """Apply left/right enable speeds with an optional short startup 'kick'.

        A brief 100% duty cycle can overcome static friction / gearbox stiction.
        Controlled via:
        - SARAH_MOTOR_KICK_MS (default 250ms - increased for better startup)
        - SARAH_MOTOR_KICK_SPEED (default 100)
        """
        left_speed = int(max(0, min(100, left_speed)))
        right_speed = int(max(0, min(100, right_speed)))

        try:
            kick_ms = int(float(os.environ.get("SARAH_MOTOR_KICK_MS", "250").strip() or "250"))
        except Exception:
            kick_ms = 250
        try:
            kick_speed = int(float(os.environ.get("SARAH_MOTOR_KICK_SPEED", "100").strip() or "100"))
        except Exception:
            kick_speed = 100

        kick_ms = max(0, min(1000, kick_ms))
        kick_speed = max(0, min(100, kick_speed))

        # Only kick if we're using PWM-style enable and requesting movement.
        do_kick = (
            kick_ms > 0
            and kick_speed > 0
            and getattr(self, "_speed_mode", "pwm") == "pwm"
            and (left_speed > 0 or right_speed > 0)
        )

        if do_kick:
            self._set_motor_speed(self.pwm_objects.get('left'), kick_speed if left_speed > 0 else 0)
            self._set_motor_speed(self.pwm_objects.get('right'), kick_speed if right_speed > 0 else 0)
            time.sleep(kick_ms / 1000.0)
            print(f"[DRIVE-GPIO] ⚡ Motor kick: {kick_ms}ms at {kick_speed}%")

        self._set_motor_speed(self.pwm_objects.get('left'), left_speed)
        self._set_motor_speed(self.pwm_objects.get('right'), right_speed)

    def _debug_dump_pin_states(self, label: str = ""):
        """Best-effort GPIO state dump on Raspberry Pi.

        This helps confirm whether pins are actually being driven when motors don't move.
        Requires either `pinctrl` or `raspi-gpio` to be installed on the Pi.
        """
        if not (IS_RASPBERRY_PI and SARAH_DEBUG and self.use_gpio):
            return

        pins = [
            getattr(self, "LEFT_IN1", None),
            getattr(self, "LEFT_IN2", None),
            getattr(self, "LEFT_IN3", None),
            getattr(self, "LEFT_IN4", None),
            getattr(self, "RIGHT_IN1", None),
            getattr(self, "RIGHT_IN2", None),
            getattr(self, "RIGHT_IN3", None),
            getattr(self, "RIGHT_IN4", None),
        ]
        pins = [p for p in pins if isinstance(p, int)]
        if not pins:
            return

        tool = None
        if _which("pinctrl"):
            tool = "pinctrl"
        elif _which("raspi-gpio"):
            tool = "raspi-gpio"

        if not tool:
            dprint(True, "[DRIVE-GPIO] [DEBUG] No pin state tool found (install: sudo apt-get install -y raspi-utils)")
            return

        header = f"[DRIVE-GPIO] [DEBUG] Pin states {label}".strip()
        dprint(True, header)
        for pin in pins:
            try:
                if tool == "pinctrl":
                    # pinctrl takes BCM GPIO number
                    out = subprocess.run(
                        ["pinctrl", "get", str(pin)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=1.5,
                    ).stdout.strip()
                else:
                    out = subprocess.run(
                        ["raspi-gpio", "get", str(pin)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=1.5,
                    ).stdout.strip()
                if out:
                    dprint(True, f"[DRIVE-GPIO] [DEBUG] {out}")
            except Exception as e:
                dprint(True, f"[DRIVE-GPIO] [DEBUG] Failed reading pin {pin}: {type(e).__name__}: {e}")

    def forward(self, speed: int = DEFAULT_SPEED):
        self.last_action = ("FORWARD", speed)
        print(f"[DRIVE] ===== FORWARD @ FULL BATTERY POWER =====")
        
        if self.use_gpio and hasattr(self, 'left_in1'):
            try:
                # Defensive: ensure all motor objects exist before calling methods
                required_attrs = ['left_in1', 'left_in2', 'left_in3', 'left_in4', 
                                  'right_in1', 'right_in2', 'right_in3', 'right_in4']
                if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in required_attrs):
                    print("[DRIVE] ERROR: Motor control objects not fully initialized")
                    return
                
                # All 4 motors forward (inverted from previous - all working now)
                # Left L298N: Motor A (front) forward
                print(f"[DRIVE-DEBUG] Setting LEFT Motor A: IN1(GPIO{self.LEFT_IN1})=LOW, IN2(GPIO{self.LEFT_IN2})=HIGH")
                self.left_in1.off()
                self.left_in2.on()
                # Left L298N: Motor B (back) forward
                print(f"[DRIVE-DEBUG] Setting LEFT Motor B: IN3(GPIO{self.LEFT_IN3})=HIGH, IN4(GPIO{self.LEFT_IN4})=LOW")
                self.left_in3.on()
                self.left_in4.off()
                # Right side inverted (mirrored mount)
                print(f"[DRIVE-DEBUG] Setting RIGHT Motor A (INVERTED): IN1(GPIO{self.RIGHT_IN1})=HIGH, IN2(GPIO{self.RIGHT_IN2})=LOW")
                self.right_in1.on()
                self.right_in2.off()
                print(f"[DRIVE-DEBUG] Setting RIGHT Motor B (INVERTED): IN3(GPIO{self.RIGHT_IN3})=LOW, IN4(GPIO{self.RIGHT_IN4})=HIGH")
                self.right_in3.off()
                self.right_in4.on()
                print("[DRIVE-GPIO] OK: All 4 motors set to FORWARD")
                self._debug_dump_pin_states("after FORWARD")
            except Exception as e:
                print(f"[DRIVE] ❌ ERROR in forward(): {e}")
                import traceback
                traceback.print_exc()

    def backward(self, speed: int = DEFAULT_SPEED):
        self.last_action = ("BACKWARD", speed)
        print(f"[DRIVE] ===== BACKWARD @ FULL BATTERY POWER =====")
        
        if self.use_gpio and hasattr(self, 'left_in1'):
            try:
                # Defensive: ensure all motor objects exist
                required_attrs = ['left_in1', 'left_in2', 'left_in3', 'left_in4', 
                                  'right_in1', 'right_in2', 'right_in3', 'right_in4']
                if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in required_attrs):
                    print("[DRIVE] ERROR: Motor control objects not fully initialized")
                    return
                
                # All 4 motors backward (reverse polarity from forward)
                print(f"[DRIVE-DEBUG] Setting LEFT Motor A: IN1(GPIO{self.LEFT_IN1})=HIGH, IN2(GPIO{self.LEFT_IN2})=LOW")
                self.left_in1.on()
                self.left_in2.off()
                print(f"[DRIVE-DEBUG] Setting LEFT Motor B: IN3(GPIO{self.LEFT_IN3})=LOW, IN4(GPIO{self.LEFT_IN4})=HIGH")
                self.left_in3.off()
                self.left_in4.on()

                # Right side inverted (mirrored mount)
                print(f"[DRIVE-DEBUG] Setting RIGHT Motor A (INVERTED): IN1(GPIO{self.RIGHT_IN1})=LOW, IN2(GPIO{self.RIGHT_IN2})=HIGH")
                self.right_in1.off()
                self.right_in2.on()
                print(f"[DRIVE-DEBUG] Setting RIGHT Motor B (INVERTED): IN3(GPIO{self.RIGHT_IN3})=HIGH, IN4(GPIO{self.RIGHT_IN4})=LOW")
                self.right_in3.on()
                self.right_in4.off()

                print("[DRIVE-GPIO] OK: All 4 motors set to BACKWARD")
                self._debug_dump_pin_states("after BACKWARD")
            except Exception as e:
                print(f"[DRIVE] ERROR backward: {e}")
                import traceback
                traceback.print_exc()

    def left(self, speed: int = DEFAULT_SPEED):
        self.last_action = ("LEFT", speed)
        print(f"[DRIVE] ===== LEFT @ FULL BATTERY POWER =====")
        
        if self.use_gpio and hasattr(self, 'left_in1'):
            try:
                # Defensive: ensure all motor objects exist
                required_attrs = ['left_in1', 'left_in2', 'left_in3', 'left_in4', 
                                  'right_in1', 'right_in2', 'right_in3', 'right_in4']
                if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in required_attrs):
                    print("[DRIVE] ERROR: Motor control objects not fully initialized")
                    return
                
                # Left motors stop, right motors forward (turn left)
                print(f"[DRIVE-DEBUG] Setting LEFT Motors: ALL OFF (brake)")
                self.left_in1.off()
                self.left_in2.off()
                self.left_in3.off()
                self.left_in4.off()

                # Right side forward with inverted polarity
                print(f"[DRIVE-DEBUG] Setting RIGHT Motor A (INVERTED): IN1(GPIO{self.RIGHT_IN1})=HIGH, IN2(GPIO{self.RIGHT_IN2})=LOW")
                self.right_in1.on()
                self.right_in2.off()
                print(f"[DRIVE-DEBUG] Setting RIGHT Motor B (INVERTED): IN3(GPIO{self.RIGHT_IN3})=LOW, IN4(GPIO{self.RIGHT_IN4})=HIGH")
                self.right_in3.off()
                self.right_in4.on()

                print("[DRIVE-GPIO] OK: Left motors OFF, Right motors FORWARD (turning LEFT)")
                self._debug_dump_pin_states("after LEFT")
            except Exception as e:
                print(f"[DRIVE] ERROR left: {e}")
                import traceback
                traceback.print_exc()

    def right(self, speed: int = DEFAULT_SPEED):
        self.last_action = ("RIGHT", speed)
        print(f"[DRIVE] ===== RIGHT @ FULL BATTERY POWER =====")
        
        if self.use_gpio and hasattr(self, 'left_in1'):
            try:
                # Defensive: ensure all motor objects exist
                required_attrs = ['left_in1', 'left_in2', 'left_in3', 'left_in4', 
                                  'right_in1', 'right_in2', 'right_in3', 'right_in4']
                if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in required_attrs):
                    print("[DRIVE] ERROR: Motor control objects not fully initialized")
                    return
                
                # Left motors forward, right motors stop (turn right)
                print(f"[DRIVE-DEBUG] Setting LEFT Motor A: IN1(GPIO{self.LEFT_IN1})=LOW, IN2(GPIO{self.LEFT_IN2})=HIGH")
                self.left_in1.off()
                self.left_in2.on()
                print(f"[DRIVE-DEBUG] Setting LEFT Motor B: IN3(GPIO{self.LEFT_IN3})=HIGH, IN4(GPIO{self.LEFT_IN4})=LOW")
                self.left_in3.on()
                self.left_in4.off()

                print(f"[DRIVE-DEBUG] Setting RIGHT Motors: ALL OFF (brake)")
                self.right_in1.off()
                self.right_in2.off()
                self.right_in3.off()
                self.right_in4.off()

                print("[DRIVE-GPIO] OK: Left motors FORWARD, Right motors OFF (turning RIGHT)")
                self._debug_dump_pin_states("after RIGHT")
            except Exception as e:
                print(f"[DRIVE] ERROR right: {e}")
                import traceback
                traceback.print_exc()

    def stop(self):
        self.last_action = ("STOP", 0)
        print("[DRIVE] ===== STOP (ALL MOTORS OFF) =====")
        
        if self.use_gpio and hasattr(self, 'left_in1'):
            try:
                # Defensive: ensure all motor objects exist (but attempt to stop what we can)
                required_attrs = ['left_in1', 'left_in2', 'left_in3', 'left_in4', 
                                  'right_in1', 'right_in2', 'right_in3', 'right_in4']
                available_motors = [attr for attr in required_attrs 
                                   if hasattr(self, attr) and getattr(self, attr) is not None]
                if len(available_motors) < len(required_attrs):
                    print(f"[DRIVE] WARNING: Only {len(available_motors)}/{len(required_attrs)} motors available for stop")
                
                # All 8 motor pins low (stop all 4 motors) - attempt each individually
                if hasattr(self, 'left_in1') and self.left_in1: self.left_in1.off()
                if hasattr(self, 'left_in2') and self.left_in2: self.left_in2.off()
                if hasattr(self, 'left_in3') and self.left_in3: self.left_in3.off()
                if hasattr(self, 'left_in4') and self.left_in4: self.left_in4.off()
                if hasattr(self, 'right_in1') and self.right_in1: self.right_in1.off()
                if hasattr(self, 'right_in2') and self.right_in2: self.right_in2.off()
                if hasattr(self, 'right_in3') and self.right_in3: self.right_in3.off()
                if hasattr(self, 'right_in4') and self.right_in4: self.right_in4.off()
                dprint(SARAH_DEBUG, "[DRIVE-GPIO] All 4 motors stopped")
                self._debug_dump_pin_states("after STOP")
            except Exception as e:
                print(f"[DRIVE] ERROR stop: {e}")
                import traceback
                traceback.print_exc()
    
    def cleanup(self):
        """Clean up GPIO Zero resources (resilient to partial initialization)."""
        if not self.use_gpio:
            return
        
        cleanup_errors = []
        for attr_name in ('left_in1', 'left_in2', 'left_in3', 'left_in4', 'right_in1', 'right_in2', 'right_in3', 'right_in4'):
            if hasattr(self, attr_name):
                try:
                    obj = getattr(self, attr_name)
                    if obj and hasattr(obj, 'close'):
                        obj.close()
                except Exception as e:
                    cleanup_errors.append(f"{attr_name}: {type(e).__name__}")
        
        if cleanup_errors:
            print(f"[DRIVE] Cleanup warnings: {', '.join(cleanup_errors)}")
        else:
            dprint(SARAH_DEBUG, "[DRIVE] GPIO cleanup complete")


###############################################
# Ultrasonic sensor support (HC-SR04)
###############################################
class UltrasonicSensor:
    def __init__(self, trigger_pin: int, echo_pin: int, use_gpio: bool = True, pin_factory=None):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.use_gpio = use_gpio
        self.trigger = None
        self.echo = None
        self.enabled = False
        
        if use_gpio:
            try:
                if not GPIOZERO_AVAILABLE or DigitalOutputDevice is None or DigitalInputDevice is None:
                    raise ImportError("gpiozero not available")
                # Set up trigger as output and echo as input with pin_factory
                self.trigger = DigitalOutputDevice(trigger_pin, initial_value=False, pin_factory=pin_factory)
                # IMPORTANT: Do NOT debounce echo for HC-SR04.
                # Echo pulses can be ~100us-20ms; bounce_time can suppress valid pulses.
                self.echo = DigitalInputDevice(echo_pin, pull_up=False, pin_factory=pin_factory)
                self.enabled = True
                print(f"[ULTRA] ✓ GPIO configured: trigger=GPIO{trigger_pin}, echo=GPIO{echo_pin}")
            except Exception as e:
                print(f"[ULTRA] ✗ FAILED to initialize ultrasonic sensor: trigger=GPIO{trigger_pin}, echo=GPIO{echo_pin}")
                print(f"[ULTRA] Error: {type(e).__name__}: {e}")
                print(f"[ULTRA] This sensor will return None (no readings)")
                self.enabled = False
        else:
            print(f"[ULTRA] Ultrasonic sensor on pins {trigger_pin}/{echo_pin} disabled (GPIO disabled)")
            self.enabled = False

    def read_distance_cm(self, timeout: float = ULTRASONIC_READ_TIMEOUT_S):
        """Returns distance in cm or None on failure. Retries once for intermittent issues."""
        if not self.enabled:
            return None
        if not self.trigger or not self.echo:
            dprint(True, f"[ULTRA-ERROR] Sensor GPIO{self.trigger_pin}/{self.echo_pin} not initialized")
            return None
        
        # Try twice for intermittent connection issues
        for attempt in range(2):
            try:
                # Send 10us pulse
                self.trigger.on()
                time.sleep(0.00001)
                self.trigger.off()

                start = time.time()
                pulse_start = start
                # Wait for echo to go high
                timeout_count = 0
                while not self.echo.value:
                    pulse_start = time.time()
                    if pulse_start - start > timeout:
                        # Timeout - retry once if first attempt
                        if attempt == 0:
                            time.sleep(0.01)
                            break
                        return None
                    timeout_count += 1
                
                if pulse_start - start > timeout:
                    continue  # Retry

                pulse_end = time.time()
                # Wait for echo to go low
                while self.echo.value:
                    pulse_end = time.time()
                    if pulse_end - pulse_start > ULTRASONIC_ECHO_HIGH_TIMEOUT_S:
                        break

                pulse_duration = pulse_end - pulse_start
                # Speed of sound 34300 cm/s divided by 2 (round trip)
                distance_cm = (pulse_duration * 34300) / 2
                
                # Sanity check: HC-SR04 range is 2-400cm
                if distance_cm < 0 or distance_cm > 500:
                    if attempt == 0:
                        time.sleep(0.01)
                        continue  # Retry
                    return None
                    
                return round(distance_cm, 2)
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.01)
                    continue  # Retry once
                print(f"[ULTRA-ERROR] GPIO{self.trigger_pin}/{self.echo_pin} exception: {type(e).__name__}: {e}")
                return None
        
        return None  # Both attempts failed

    def cleanup(self):
        try:
            if self.trigger:
                self.trigger.close()
            if self.echo:
                self.echo.close()
        except (AttributeError, RuntimeError):
            pass


def init_ultrasonic_sensors(pins=ULTRASONIC_PINS, use_gpio=False, pin_factory=None):
    """Initialize ultrasonic sensors with the specified pin factory.
    
    Args:
        pins: List of (trigger, echo) pin tuples
        use_gpio: Whether to use real GPIO or simulate
        pin_factory: gpiozero pin factory to use (must match motor driver factory)
    """
    print(f"[ULTRA-INIT] Initializing {len(pins)} ultrasonic sensors with use_gpio={use_gpio}")
    sensors = []
    for i, (trig, echo) in enumerate(pins):
        label = ['left', 'center', 'right'][i] if i < 3 else f'sensor{i}'
        print(f"[ULTRA-INIT] Initializing {label} sensor: trigger=GPIO{trig}, echo=GPIO{echo}")
        sensor = UltrasonicSensor(trig, echo, use_gpio=use_gpio, pin_factory=pin_factory)
        sensors.append(sensor)
        if sensor.enabled:
            print(f"[ULTRA-INIT] ✓ {label} sensor enabled and ready")
        else:
            print(f"[ULTRA-INIT] ✗ {label} sensor FAILED - will return None readings")
    
    enabled_count = sum(1 for s in sensors if s.enabled)
    print(f"[ULTRA-INIT] Summary: {enabled_count}/{len(sensors)} sensors operational")
    return sensors


def run_motor_smoke_test(drive) -> None:
    """Run a short motor test sequence and then return.

    Enabled via env var SARAH_MOTOR_SMOKE_TEST=1.
    Designed to be safe: short bursts and always stops in finally.
    """
    print("\n" + "!" * 70)
    print("[MOTOR-TEST] Motor smoke test ENABLED")
    print("[MOTOR-TEST] Safety: raise wheels / disconnect drivetrain before testing.")
    print("[MOTOR-TEST] Sequence: FORWARD, BACKWARD, LEFT, RIGHT")
    print("!" * 70 + "\n")

    if not drive or not getattr(drive, "use_gpio", False):
        print("[MOTOR-TEST] GPIO drive not enabled/available; cannot run motor test.")
        return

    try:
        speed = int(float(os.environ.get("SARAH_MOTOR_TEST_SPEED", "80").strip() or "80"))
    except Exception:
        speed = 80
    speed = max(MIN_SPEED, min(MAX_SPEED, speed))

    try:
        duration = float(os.environ.get("SARAH_MOTOR_TEST_DURATION", "5.0").strip() or "5.0")
    except Exception:
        duration = 5.0
    duration = max(0.2, min(10.0, duration))

    print(f"[MOTOR-TEST] Speed={speed}% Duration={duration:.1f}s (override with SARAH_MOTOR_TEST_SPEED / SARAH_MOTOR_TEST_DURATION)")
    seq = [
        {"command": "FORWARD", "duration": duration, "speed": speed, "speak": ""},
        {"command": "BACKWARD", "duration": duration, "speed": speed, "speak": ""},
        {"command": "LEFT", "duration": duration, "speed": speed, "speak": ""},
        {"command": "RIGHT", "duration": duration, "speed": speed, "speak": ""},
        {"command": "STOP", "duration": 0, "speed": 0, "speak": ""},
    ]

    try:
        for cmd in seq:
            execute_command(drive, cmd, silent=False)
            time.sleep(0.2)
    finally:
        try:
            drive.stop()
        except Exception:
            pass
        try:
            drive.cleanup()
        except Exception:
            pass
        print("[MOTOR-TEST] Complete. Motors stopped and GPIO cleaned up.")


def read_all_sensors(sensors, verbose: Optional[bool] = None):
    if verbose is None:
        verbose = bool(globals().get("SARAH_DEBUG", False))

    results = []
    for i, s in enumerate(sensors):
        try:
            d = s.read_distance_cm(timeout=ULTRASONIC_READ_TIMEOUT_S)
            # FIXED: Return None for failed readings instead of -1
            # Also filter out readings below minimum valid threshold
            if d is not None and float(d) >= ULTRASONIC_MIN_VALID_CM:
                results.append(d)
            else:
                results.append(None)
            # Diagnostic output
            if verbose:
                if d is not None:
                    label = ('left' if i==0 else 'center' if i==1 else 'right')
                    if float(d) < ULTRASONIC_MIN_VALID_CM:
                        print(f"[ULTRA-READ] Sensor {i} ({label}): {d:.1f}cm (below {ULTRASONIC_MIN_VALID_CM:.1f}cm; ignored for decisions)")
                    else:
                        print(f"[ULTRA-READ] Sensor {i} ({label}): {d:.1f}cm")
                else:
                    label = ('left' if i==0 else 'center' if i==1 else 'right')
                    print(f"[ULTRA-READ] Sensor {i} ({label}): TIMEOUT/ERROR (check wiring on pins {s.trigger_pin}/{s.echo_pin})")
        except Exception as ex:
            results.append(None)
            if verbose:
                label = ('left' if i==0 else 'center' if i==1 else 'right')
                print(f"[ULTRA-READ] Sensor {i} ({label}) EXCEPTION: {ex}")
    return results

###############################################
# Autonomous thread
###############################################
class AutonomousThread(threading.Thread):
    def __init__(self, drive, stop_event: threading.Event, camera_thread: 'CameraThread', sensors=None):
        super().__init__(daemon=True)
        self.drive = drive
        self.stop_event = stop_event
        self.camera_thread = camera_thread
        self.camera_analyzer = CameraAnalyzer(camera_thread)
        self.sensors = sensors or []

        # Exploration tuning knobs (reduce forward bias without relying solely on the model).
        # These are intentionally env-configurable so you can tune behavior on the Pi without code edits.
        def _env_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)).strip())
            except Exception:
                return float(default)

        def _env_int(name: str, default: int) -> int:
            try:
                return int(float(os.getenv(name, str(default)).strip()))
            except Exception:
                return int(default)

        # Extremely aggressive explore defaults: bias toward forward progress and longer-range exploration.
        # Hard safety still comes from ULTRASONIC_STOP_DISTANCE_CM + emergency stop + mid-movement checks.
        self._explore_forward_w = max(0.0, _env_float('SARAH_AUTO_FORWARD_WEIGHT', 0.65))
        self._explore_left_w = max(0.0, _env_float('SARAH_AUTO_LEFT_WEIGHT', 0.175))
        self._explore_right_w = max(0.0, _env_float('SARAH_AUTO_RIGHT_WEIGHT', 0.175))
        self._max_forward_streak = max(1, _env_int('SARAH_AUTO_MAX_FORWARD_STREAK', 4))
        self._turn_when_clear_prob = min(1.0, max(0.0, _env_float('SARAH_AUTO_TURN_WHEN_CLEAR_PROB', 0.25)))
        # Loosen "clear" thresholds so the planner will attempt moves in more situations.
        # Hard safety still comes from ULTRASONIC_STOP_DISTANCE_CM + emergency stop + mid-movement checks.
        # User-requested minimum clearance to proceed (must account for robot width).
        self._min_proceed_cm = max(0.0, _env_float('SARAH_AUTO_MIN_PROCEED_CM', 20.0))
        # CRITICAL: Forward clearance must ensure robot can FIT through (both front sensors clear)
        # We check BOTH left and right front sensors - if either is too close, robot won't fit
        self._clear_forward_cm = max(float(self._min_proceed_cm), max(10.0, _env_float('SARAH_AUTO_CLEAR_FORWARD_CM', 20.0)))
        self._clear_side_cm = max(float(self._min_proceed_cm), max(10.0, _env_float('SARAH_AUTO_CLEAR_SIDE_CM', 20.0)))

        # Adaptive forward-duration tuning.
        # Loosened defaults so exploration is quicker (still gated by ultrasonic stops + mid-move monitoring).
        self._safety_margin_cm = max(6.0, _env_float('SARAH_AUTO_SAFETY_MARGIN_CM', 12.0))
        # Calibrate: user-reported ≈1m per 5s at 100% => 20cm/s.
        # Keep env override so you can fine-tune on the Pi.
        self._cm_per_sec = max(5.0, _env_float('SARAH_AUTO_CM_PER_SEC', 20.0))
        # Reduced min duration from 0.15→0.10 so short safe moves are allowed.
        self._forward_min_duration = max(0.05, _env_float('SARAH_AUTO_FORWARD_MIN_DURATION', 0.10))
        # Smooth intelligent movements: longer forward bursts allow better coverage while safety checks prevent collisions.
        # With ~20cm/s, 3.0s covers ~60cm. Mid-movement monitoring + pre-execution checks ensure safety.
        self._forward_max_duration = min(AUTONOMOUS_MAX_MOVE_DURATION, max(0.05, _env_float('SARAH_AUTO_FORWARD_MAX_DURATION', 3.0)))
        # Brake distance accounts for command latency + stopping distance.
        # Loosened slightly for faster exploration.
        self._brake_distance_cm = max(1.0, _env_float('SARAH_AUTO_BRAKE_DISTANCE_CM', 3.5))
        # Mechanical stopping distance/inertia (added buffer). Tune on-robot.
        self._motor_stopping_distance_cm = max(0.0, _env_float('SARAH_AUTO_MOTOR_STOPPING_DISTANCE_CM', 6.0))
        # Emergency stop threshold (tighter than regular safety margin).
        self._emergency_stop_cm = max(4.0, _env_float('SARAH_AUTO_EMERGENCY_STOP_CM', 10.0))
        self._turn_duration = min(AUTONOMOUS_MAX_MOVE_DURATION, max(0.05, _env_float('SARAH_AUTO_TURN_DURATION', 0.6)))

        # Per-command autonomous duration caps (in addition to AUTONOMOUS_MAX_MOVE_DURATION).
        # These caps apply to execution-level clamping and to the autonomous loop.
        self._max_forward_sec = min(AUTONOMOUS_MAX_MOVE_DURATION, max(0.05, _env_float('SARAH_AUTO_MAX_FORWARD_SEC', 4.5)))  # Increased from 5.0 for longer exploration
        self._max_backward_sec = min(AUTONOMOUS_MAX_MOVE_DURATION, max(0.05, _env_float('SARAH_AUTO_MAX_BACKWARD_SEC', 1.5)))
        self._max_turn_sec = min(AUTONOMOUS_MAX_MOVE_DURATION, max(0.05, _env_float('SARAH_AUTO_MAX_TURN_SEC', 1.0)))  # Reduced from 1.2 for quicker turns

        # Lightweight mapping (dead-reckoning). This is approximate but enables:
        # - avoiding immediate repeats ("I've been here")
        # - return-to-start by retracing executed commands.
        self._pose_lock = threading.Lock()
        self._x_cm = 0.0
        self._y_cm = 0.0
        self._heading_deg = 0.0  # 0 = +X, 90 = +Y
        # Robot physical dimensions: width ≈11in (~28cm), need margin on both sides
        self._robot_width_cm = max(15.0, _env_float('SARAH_ROBOT_WIDTH_CM', 28.0))
        self._robot_side_margin_cm = float(self._robot_width_cm) / 2.0 + 3.0  # Half-width + safety margin
        # Larger grid => bigger step goals and more aggressive coverage.
        self._grid_cm = max(10.0, _env_float('SARAH_AUTO_GRID_CM', 45.0))
        self._turn_deg_per_sec = max(10.0, _env_float('SARAH_AUTO_TURN_DEG_PER_SEC', 90.0))
        self._motion_history: list[dict] = []
        self._visited: set[tuple[int, int]] = {(0, 0)}
        # Cells we infer are blocked by obstacles (approx). Avoid planning into these repeatedly.
        self._blocked: set[tuple[int, int]] = set()

        # Option J: Persistent long-term memory + relative map (disk-backed).
        # This stores visited/blocked cells + lightweight sensor/vision landmarks across runs.
        # Disable with SARAH_AUTO_PERSIST=0. Reset by deleting the file or setting SARAH_AUTO_PERSIST_RESET=1.
        try:
            self._persist_enabled = os.getenv('SARAH_AUTO_PERSIST', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        except Exception:
            self._persist_enabled = True
        try:
            reset = os.getenv('SARAH_AUTO_PERSIST_RESET', '0').strip().lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            reset = False
        self._persist_reset_on_start = bool(reset)

        def _default_persist_path() -> str:
            try:
                base = os.path.dirname(os.path.abspath(__file__))
            except Exception:
                base = os.getcwd()
            return os.path.join(base, 'sarah_auto_persist.json')

        try:
            p = str(os.getenv('SARAH_AUTO_PERSIST_PATH', '') or '').strip()
        except Exception:
            p = ''
        self._persist_path = p or _default_persist_path()
        self._persist_save_interval_s = max(2.0, float(_env_float('SARAH_AUTO_PERSIST_SAVE_INTERVAL_S', 15.0)))
        self._persist_landmark_interval_s = max(1.0, float(_env_float('SARAH_AUTO_PERSIST_LANDMARK_INTERVAL_S', 6.0)))
        self._persist_last_save_ts = 0.0
        self._persist_last_landmark_ts = 0.0
        self._persist_max_landmarks = max(0, _env_int('SARAH_AUTO_PERSIST_MAX_LANDMARKS', 300))
        self._persist_max_cells = max(500, _env_int('SARAH_AUTO_PERSIST_MAX_CELLS', 12000))

        # Evidence-style occupancy scores: + means blocked evidence, - means free evidence.
        # This is conservative and primarily used to reinforce the existing _blocked set.
        self._occ_score: dict[tuple[int, int], float] = {}
        self._persist_landmarks: list[dict] = []
        self._explore_exhausted = False
        self._exhausted_cycles = 0
        # Frontier-based exploration (Option C)
        self._frontier_enabled = os.getenv('SARAH_AUTO_FRONTIER', '1').strip().lower() not in ('0', 'false', 'no')
        # Increase default radius so the robot will push much farther outward instead of hovering locally.
        self._frontier_max_radius_cells = max(2, _env_int('SARAH_AUTO_FRONTIER_MAX_RADIUS', 80))
        self._frontier_debug = os.getenv('SARAH_AUTO_FRONTIER_DEBUG', '0').strip().lower() in ('1', 'true', 'yes')
        # Frontier target persistence: commit to a chosen target for a few steps
        # (reduces dithering and makes exploration look more intentional).
        self._frontier_active_target: tuple[int, int] | None = None
        self._frontier_target_failures = 0
        self._frontier_target_set_time = 0.0
        self._frontier_target_max_failures = max(1, _env_int('SARAH_AUTO_FRONTIER_TARGET_MAX_FAIL', 4))

        # Open-space cruising: if BOTH ultrasonic + vision indicate very clear forward space,
        # bias desired forward distance upward (still bounded by forward max duration + safety clamp).
        self._open_space_cruise = os.getenv('SARAH_AUTO_OPEN_SPACE_CRUISE', '1').strip().lower() not in ('0', 'false', 'no')
        self._open_space_min_ultra_cm = max(0.0, _env_float('SARAH_AUTO_OPEN_SPACE_MIN_ULTRA_CM', 70.0))
        self._open_space_min_vision_cm = max(0.0, _env_float('SARAH_AUTO_OPEN_SPACE_MIN_VISION_CM', 120.0))
        self._open_space_desired_cm = max(float(self._grid_cm), _env_float('SARAH_AUTO_OPEN_SPACE_DESIRED_CM', 120.0))

        # Ultrasonic debounce + hysteresis for smoother planning (hard safety still uses raw readings).
        self._ultra_debounce_n = max(1, _env_int('SARAH_AUTO_ULTRA_DEBOUNCE_SAMPLES', 3))
        self._ultra_hysteresis_cm = max(0.0, _env_float('SARAH_AUTO_ULTRA_HYSTERESIS_CM', 5.0))
        self._ultra_clear_cycles = max(1, _env_int('SARAH_AUTO_ULTRA_CLEAR_CYCLES', 2))
        self._ultra_hist_left: deque[float] = deque(maxlen=int(self._ultra_debounce_n))
        self._ultra_hist_center: deque[float] = deque(maxlen=int(self._ultra_debounce_n))
        self._ultra_hist_right: deque[float] = deque(maxlen=int(self._ultra_debounce_n))
        self._ultra_forward_blocked = False
        self._ultra_forward_clear_count = 0

        # Pre-scan (one-time on entering autonomous): rotate and sample sensors/vision to choose aggressiveness.
        self._prescan_enabled = os.getenv('SARAH_AUTO_PRESCAN', '1').strip().lower() not in ('0', 'false', 'no')
        # Default shortened to reduce startup latency in autonomous.
        self._prescan_max_seconds = max(0.0, _env_float('SARAH_AUTO_PRESCAN_MAX_SECONDS', 6.0))
        self._prescan_turn_step_sec = max(0.05, _env_float('SARAH_AUTO_PRESCAN_TURN_STEP_SEC', 0.35))
        self._prescan_turn_speed = int(max(40.0, min(100.0, _env_float('SARAH_AUTO_PRESCAN_TURN_SPEED', 70.0))))
        self._prescan_vision_every_n = max(1, _env_int('SARAH_AUTO_PRESCAN_VISION_EVERY_N', 3))
        self._did_prescan = False
        self._last_prescan_time = 0.0

        # Baseline values (so prescan scales predictably without permanently drifting).
        self._base_explore_forward_w = float(self._explore_forward_w)
        self._base_explore_left_w = float(self._explore_left_w)
        self._base_explore_right_w = float(self._explore_right_w)
        self._base_turn_when_clear_prob = float(self._turn_when_clear_prob)
        self._base_forward_max_duration = float(self._forward_max_duration)
        self._base_clear_forward_cm = float(self._clear_forward_cm)
        self._base_clear_side_cm = float(self._clear_side_cm)
        self._base_open_space_min_ultra_cm = float(self._open_space_min_ultra_cm)
        self._base_open_space_min_vision_cm = float(self._open_space_min_vision_cm)
        self._base_open_space_desired_cm = float(self._open_space_desired_cm)
        self.system_prompt = (
            "You are SARAH, an autonomous exploration robot. Your goal is to explore safely and intelligently.\n"
            "\n"
            "EXPLORATION PRIORITIES:\n"
            "1. SAFETY FIRST: Avoid collisions. Maintain a safe distance (>20cm from obstacles when possible).\n"
            "2. EXPLORE: Don't just go forward repeatedly. Mix it up - turn left, turn right, explore new areas.\n"
            "3. USE VISION: Estimate distances from camera images. Close objects = turn away. Far objects = safe to proceed.\n"
            "4. VARIETY: Randomly choose different directions to discover your environment fully.\n"
            "\n"
            "DISTANCE ESTIMATION FROM CAMERA:\n"
            "- Large objects filling frame = VERY CLOSE (<20cm) → Turn immediately or stop\n"
            "- Clear details visible = CLOSE (20-50cm) → Consider turning\n"
            "- Objects visible but smaller = MEDIUM (50-100cm) → Safe to move forward but prepare to turn\n"
            "- Small/distant objects = FAR (>100cm) → Safe to explore forward\n"
            "- Open space, no obstacles = CLEAR → Good for forward movement\n"
            "\n"
            "DECISION RULES:\n"
            "- If obstacle VERY CLOSE ahead (<30cm): Turn LEFT or RIGHT (pick randomly)\n"
            "- If obstacle CLOSE on one side: Turn away from it\n"
            "- If path is clear: prefer forward to cover ground quickly (default target: ~65% FORWARD, ~17.5% LEFT, ~17.5% RIGHT).\n"
            "- If stuck or facing wall: Turn LEFT or RIGHT, or BACKWARD briefly\n"
            "- If you went FORWARD last cycle and left/right are clear, choose LEFT or RIGHT next.\n"
            "\n"
            "WHEN AN IMAGE IS PROVIDED, ALSO INCLUDE THESE OPTIONAL FIELDS IN YOUR JSON (best effort):\n"
            "- distance_estimate: very_close/close/medium/far/clear\n"
            "- distance_cm: number\n"
            "- clear_left/clear_right/clear_forward: true/false\n"
            "- obstacles: true/false\n"
            "- recommendation: STOP/TURN_LEFT/TURN_RIGHT/PROCEED\n"
            "- obstacle_size: 0.0-1.0 (approx fraction of frame filled by nearest obstacle; larger means closer)\n"
            "\n"
            "Valid commands: FORWARD, BACKWARD, LEFT, RIGHT, STOP\n"
            "Use 100% speed and 2-4 seconds duration for movements (max 5.0s).\n"
            "\n"
            "REQUIRED JSON RESPONSE FORMAT:\n"
            "{\"command\": \"FORWARD\", \"duration\": 4.0, \"speed\": 100, \"speak\": \"Exploring ahead.\", \"emotion\": \"excited\", \"reasoning\": \"Path clear, moving forward\"}\n"
            "\n"
            "The 'emotion' field controls your facial expression. Valid: neutral, happy, excited, thinking, surprised, concerned, sad.\n"
            "The 'reasoning' field explains why you chose this action (for learning).\n"
            "Always include emotion and reasoning fields."
        )

    def _persist__serialize_cells(self, cells: set[tuple[int, int]]) -> list[list[int]]:
        try:
            return [[int(x), int(y)] for (x, y) in cells]
        except Exception:
            return []

    def _persist__deserialize_cells(self, raw) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        try:
            if not raw:
                return out
            for item in raw:
                try:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        continue
                    out.add((int(item[0]), int(item[1])))
                except Exception:
                    continue
        except Exception:
            return set()
        return out

    def _persist__limit_cells(self, cells: set[tuple[int, int]], max_cells: int) -> set[tuple[int, int]]:
        try:
            max_cells = int(max_cells)
        except Exception:
            max_cells = 0
        if max_cells <= 0:
            return set()
        if len(cells) <= max_cells:
            return cells
        try:
            # Keep cells closest to origin (stable across sessions; avoids unbounded growth).
            kept = sorted(cells, key=lambda c: abs(int(c[0])) + abs(int(c[1])))[:max_cells]
            return set(kept)
        except Exception:
            # Fallback: arbitrary truncation.
            out: set[tuple[int, int]] = set()
            for c in cells:
                out.add(c)
                if len(out) >= max_cells:
                    break
            return out

    def _persist__limit_occ_score(self, scores: dict[tuple[int, int], float], max_cells: int) -> dict[tuple[int, int], float]:
        try:
            max_cells = int(max_cells)
        except Exception:
            max_cells = 0
        if max_cells <= 0:
            return {}
        if len(scores) <= max_cells:
            return scores
        try:
            items = list(scores.items())
            items.sort(key=lambda kv: abs(float(kv[1])), reverse=True)
            items = items[:max_cells]
            return {k: float(v) for (k, v) in items}
        except Exception:
            return dict(list(scores.items())[:max_cells])

    def _load_persistent_state(self) -> None:
        if not bool(getattr(self, '_persist_enabled', False)):
            return
        path = str(getattr(self, '_persist_path', '') or '').strip()
        if not path:
            return
        try:
            if bool(getattr(self, '_persist_reset_on_start', False)):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                return

            if not os.path.exists(path):
                return
            
            # Safety: reject unreasonably large files (>50MB) to prevent loading corrupted/malicious data.
            try:
                file_size = os.path.getsize(path)
                if file_size > 50 * 1024 * 1024:
                    print(f"[AUTO-PERSIST] WARNING: persist file too large ({file_size} bytes), ignoring.")
                    return
            except Exception:
                pass
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                print("[AUTO-PERSIST] WARNING: corrupt persist file (not a dict), ignoring.")
                return

            version = int(data.get('version', 1) or 1)
            if version < 1:
                return

            loaded_grid = data.get('grid_cm', None)
            try:
                loaded_grid = float(loaded_grid) if loaded_grid is not None else None
            except Exception:
                loaded_grid = None
            # If grid resolution changes a lot, ignore old map (avoids garbage scaling).
            if loaded_grid is not None:
                try:
                    if float(self._grid_cm) > 0 and abs(float(loaded_grid) - float(self._grid_cm)) / float(self._grid_cm) > 0.25:
                        print(f"[AUTO-PERSIST] Ignoring saved map due to grid mismatch (saved={loaded_grid}, current={self._grid_cm}).")
                        return
                except Exception:
                    pass

            visited = self._persist__deserialize_cells(data.get('visited', []))
            blocked = self._persist__deserialize_cells(data.get('blocked', []))
            occ_raw = data.get('occ_score', {})
            occ: dict[tuple[int, int], float] = {}
            try:
                if isinstance(occ_raw, dict):
                    for k, v in occ_raw.items():
                        try:
                            if not isinstance(k, str) or ',' not in k:
                                continue
                            xs, ys = k.split(',', 1)
                            occ[(int(xs), int(ys))] = float(v)
                        except Exception:
                            continue
            except Exception:
                occ = {}

            landmarks = data.get('landmarks', [])
            if not isinstance(landmarks, list):
                landmarks = []

            pose = data.get('pose', {})
            if not isinstance(pose, dict):
                pose = {}

            with self._pose_lock:
                self._visited |= self._persist__limit_cells(visited, int(self._persist_max_cells))
                self._blocked |= self._persist__limit_cells(blocked, int(self._persist_max_cells))
                self._occ_score.update(self._persist__limit_occ_score(occ, int(self._persist_max_cells)))
                self._persist_landmarks = list(landmarks)[-int(self._persist_max_landmarks):] if int(self._persist_max_landmarks) > 0 else []

                # Restore pose only if it looks sane.
                try:
                    x_cm = float(pose.get('x_cm', self._x_cm))
                    y_cm = float(pose.get('y_cm', self._y_cm))
                    hd = float(pose.get('heading_deg', self._heading_deg))
                    if abs(x_cm) < 1e6 and abs(y_cm) < 1e6:
                        self._x_cm = x_cm
                        self._y_cm = y_cm
                        self._heading_deg = hd % 360.0
                except Exception:
                    pass
                self._mark_visited_locked()

            print(f"[AUTO-PERSIST] Loaded map: visited={len(self._visited)}, blocked={len(self._blocked)}, landmarks={len(self._persist_landmarks)}")
        except Exception as e:
            dprint(True, f"[AUTO-PERSIST] Load failed: {e}")

    def _save_persistent_state(self) -> None:
        if not bool(getattr(self, '_persist_enabled', False)):
            return
        path = str(getattr(self, '_persist_path', '') or '').strip()
        if not path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

        try:
            with self._pose_lock:
                visited = self._persist__limit_cells(set(self._visited), int(self._persist_max_cells))
                blocked = self._persist__limit_cells(set(self._blocked), int(self._persist_max_cells))
                occ_score = self._persist__limit_occ_score(dict(self._occ_score), int(self._persist_max_cells))
                landmarks = list(self._persist_landmarks)
                if int(self._persist_max_landmarks) > 0:
                    landmarks = landmarks[-int(self._persist_max_landmarks):]
                else:
                    landmarks = []

                payload = {
                    'version': 1,
                    'saved_ts': float(time.time()),
                    'grid_cm': float(self._grid_cm),
                    'robot_width_cm': float(self._robot_width_cm),
                    'pose': {
                        'x_cm': float(self._x_cm),
                        'y_cm': float(self._y_cm),
                        'heading_deg': float(self._heading_deg),
                        'grid_cell': list(self._grid_cell_from_xy(self._x_cm, self._y_cm)),
                    },
                    'visited': self._persist__serialize_cells(visited),
                    'blocked': self._persist__serialize_cells(blocked),
                    'occ_score': {f"{int(k[0])},{int(k[1])}": float(v) for (k, v) in occ_score.items()},
                    'landmarks': landmarks,
                }

            tmp = f"{path}.tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
            self._persist_last_save_ts = float(time.time())
        except Exception as e:
            dprint(True, f"[AUTO-PERSIST] Save failed: {e}")

    def _maybe_persist(self, force: bool = False) -> None:
        if not bool(getattr(self, '_persist_enabled', False)):
            return
        try:
            now = float(time.time())
        except Exception:
            now = 0.0
        try:
            last = float(getattr(self, '_persist_last_save_ts', 0.0) or 0.0)
        except Exception:
            last = 0.0
        interval = float(getattr(self, '_persist_save_interval_s', 15.0) or 15.0)
        if force or (now - last) >= interval:
            self._save_persistent_state()

    def _update_relative_map(self, sensor_readings_raw, min_front_cm: float | None, camera_analysis: dict | None) -> None:
        """Fuse ultrasonic + vision into a conservative relative occupancy memory.

        Notes:
        - Uses dead-reckoning pose; only intended to avoid repeated bad directions.
        - Primarily reinforces _blocked; does not aggressively "clear" blocked cells.
        """

        def _safe_float(v):
            try:
                x = float(v)
                return x if x > 0 else None
            except Exception:
                return None

        left_cm = None
        right_cm = None
        try:
            if sensor_readings_raw and len(sensor_readings_raw) >= 3:
                left_cm = _safe_float(sensor_readings_raw[0])
                right_cm = _safe_float(sensor_readings_raw[2])
        except Exception:
            left_cm = right_cm = None

        front_min = _safe_float(min_front_cm)

        # Vision signals.
        v_est = None
        v_cm = None
        v_obstacles = None
        v_clear_fwd = None
        v_desc = None
        if camera_analysis and isinstance(camera_analysis, dict):
            try:
                v_est = str(camera_analysis.get('distance_estimate', '') or '').strip().lower() or None
            except Exception:
                v_est = None
            v_cm = _safe_float(camera_analysis.get('distance_cm', None))
            try:
                v_obstacles = bool(camera_analysis.get('obstacles', False))
            except Exception:
                v_obstacles = None
            try:
                v_clear_fwd = bool(camera_analysis.get('clear_forward', True))
            except Exception:
                v_clear_fwd = None
            try:
                v_desc = str(camera_analysis.get('description', '') or '').strip()
                if v_desc:
                    v_desc = v_desc[:140]
            except Exception:
                v_desc = None

        try:
            stop_cm = float(max(float(ULTRASONIC_STOP_DISTANCE_CM), float(self._min_proceed_cm)))
        except Exception:
            stop_cm = float(ULTRASONIC_STOP_DISTANCE_CM)

        with self._pose_lock:
            neigh = self._neighbor_cells_from_pose_locked()

            # Always reinforce blocked from ultrasonic.
            try:
                self._mark_blocked_from_sensors_locked(front_min, left_cm, right_cm)
            except Exception:
                pass

            def _bump(cell: tuple[int, int], delta: float) -> None:
                try:
                    self._occ_score[cell] = float(self._occ_score.get(cell, 0.0)) + float(delta)
                except Exception:
                    return

            # Ultrasonic evidence.
            if front_min is not None:
                if float(front_min) <= float(stop_cm):
                    _bump(neigh['F'], +2.0)
                elif float(front_min) >= float(self._clear_forward_cm):
                    _bump(neigh['F'], -0.35)

            if left_cm is not None:
                if float(left_cm) <= float(stop_cm):
                    _bump(neigh['L'], +1.5)
                elif float(left_cm) >= float(self._clear_side_cm):
                    _bump(neigh['L'], -0.20)

            if right_cm is not None:
                if float(right_cm) <= float(stop_cm):
                    _bump(neigh['R'], +1.5)
                elif float(right_cm) >= float(self._clear_side_cm):
                    _bump(neigh['R'], -0.20)

            # Vision evidence (conservative; only affects forward direction).
            if v_obstacles is True or v_est in ('very_close', 'close'):
                _bump(neigh['F'], +0.75)
            if v_clear_fwd is True and (v_est in ('far', 'clear') or (v_cm is not None and float(v_cm) >= float(self._open_space_min_vision_cm))):
                _bump(neigh['F'], -0.35)

            # Promote evidence score to blocked status (never auto-unblock).
            try:
                if float(self._occ_score.get(neigh['F'], 0.0)) >= 2.5:
                    self._blocked.add(neigh['F'])
                if float(self._occ_score.get(neigh['L'], 0.0)) >= 2.5:
                    self._blocked.add(neigh['L'])
                if float(self._occ_score.get(neigh['R'], 0.0)) >= 2.5:
                    self._blocked.add(neigh['R'])
            except Exception:
                pass

            # Record sparse landmarks for long-term memory.
            try:
                if self._persist_max_landmarks > 0 and (v_desc or v_est or v_cm is not None):
                    now = float(time.time())
                    if (now - float(self._persist_last_landmark_ts)) >= float(self._persist_landmark_interval_s):
                        self._persist_last_landmark_ts = now
                        cell = self._grid_cell_from_xy(self._x_cm, self._y_cm)
                        self._persist_landmarks.append({
                            'ts': now,
                            'cell': [int(cell[0]), int(cell[1])],
                            'heading_deg': float(self._heading_deg),
                            'vision': {
                                'distance_estimate': v_est,
                                'distance_cm': v_cm,
                                'obstacles': v_obstacles,
                                'clear_forward': v_clear_fwd,
                                'description': v_desc,
                            },
                            'ultra': {
                                'left_cm': left_cm,
                                'right_cm': right_cm,
                                'front_min_cm': front_min,
                            },
                        })
                        if len(self._persist_landmarks) > int(self._persist_max_landmarks):
                            self._persist_landmarks = self._persist_landmarks[-int(self._persist_max_landmarks):]
            except Exception:
                pass

            # Prevent unbounded growth.
            try:
                if len(self._visited) > int(self._persist_max_cells):
                    self._visited = self._persist__limit_cells(self._visited, int(self._persist_max_cells))
                if len(self._blocked) > int(self._persist_max_cells):
                    self._blocked = self._persist__limit_cells(self._blocked, int(self._persist_max_cells))
                if len(self._occ_score) > int(self._persist_max_cells):
                    self._occ_score = self._persist__limit_occ_score(self._occ_score, int(self._persist_max_cells))
            except Exception:
                pass

    def _reset_map_memory_runtime(self) -> None:
        """Clear in-memory mapping + delete persistence file (Option J)."""

        # Stop motors for safety (best effort).
        try:
            if self.drive:
                self.drive.stop()
        except Exception:
            pass

        # Delete persist file.
        try:
            path = str(getattr(self, '_persist_path', '') or '').strip()
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

        # Reset internal state.
        with self._pose_lock:
            self._x_cm = 0.0
            self._y_cm = 0.0
            self._heading_deg = 0.0
            self._motion_history = []
            self._visited = {(0, 0)}
            self._blocked = set()
            self._occ_score = {}
            self._persist_landmarks = []
            self._frontier_active_target = None
            self._frontier_target_failures = 0
            self._explore_exhausted = False
            self._exhausted_cycles = 0
            try:
                self._mark_visited_locked()
            except Exception:
                pass

        # Avoid immediately re-saving an empty file.
        try:
            self._persist_last_save_ts = float(time.time())
        except Exception:
            self._persist_last_save_ts = 0.0

        print("[AUTO-PERSIST] Map memory reset (in-memory + disk).")
        
        # Speak confirmation to user.
        try:
            speak("Map memory cleared. Starting fresh.")
        except Exception:
            pass

    def _grid_cell_from_xy(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        g = float(self._grid_cm) if float(self._grid_cm) > 0 else 50.0
        return (int(round(x_cm / g)), int(round(y_cm / g)))

    def _heading_snap_90(self, heading_deg: float) -> float:
        try:
            h = float(heading_deg)
        except Exception:
            h = 0.0
        return (round(h / 90.0) * 90.0) % 360.0

    def _dir_vec_from_heading(self, heading_deg: float) -> tuple[int, int]:
        h = self._heading_snap_90(heading_deg)
        if h == 0.0:
            return (1, 0)
        if h == 90.0:
            return (0, 1)
        if h == 180.0:
            return (-1, 0)
        return (0, -1)

    def _neighbor_cells_from_pose_locked(self) -> dict[str, tuple[int, int]]:
        """Return the next grid cell if we step F/L/R/B (based on snapped heading)."""
        cur = self._grid_cell_from_xy(self._x_cm, self._y_cm)
        dx, dy = self._dir_vec_from_heading(self._heading_deg)
        f = (cur[0] + dx, cur[1] + dy)
        l = (cur[0] - dy, cur[1] + dx)
        r = (cur[0] + dy, cur[1] - dx)
        b = (cur[0] - dx, cur[1] - dy)
        return {'F': f, 'L': l, 'R': r, 'B': b}

    def _compute_frontiers_locked(self) -> set[tuple[int, int]]:
        """Frontier = unvisited cell adjacent to a visited cell (4-neighborhood)."""
        frontiers: set[tuple[int, int]] = set()
        for (x, y) in self._visited:
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n not in self._visited and n not in self._blocked:
                    frontiers.add(n)
        return frontiers

    def _mark_blocked_from_sensors_locked(self, front_min_cm: float | None, left_cm: float | None, right_cm: float | None) -> None:
        """Mark likely-blocked neighbor cells based on front ultrasonic readings."""
        try:
            stop_cm = float(max(float(ULTRASONIC_STOP_DISTANCE_CM), float(self._min_proceed_cm)))
        except Exception:
            stop_cm = float(ULTRASONIC_STOP_DISTANCE_CM)

        neigh = self._neighbor_cells_from_pose_locked()
        try:
            if front_min_cm is not None and float(front_min_cm) <= stop_cm:
                self._blocked.add(neigh['F'])
        except Exception:
            pass
        try:
            if left_cm is not None and float(left_cm) <= stop_cm:
                self._blocked.add(neigh['L'])
        except Exception:
            pass
        try:
            if right_cm is not None and float(right_cm) <= stop_cm:
                self._blocked.add(neigh['R'])
        except Exception:
            pass

    def _manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _choose_frontier_target_locked(self) -> tuple[int, int] | None:
        cur = self._grid_cell_from_xy(self._x_cm, self._y_cm)
        frontiers = self._compute_frontiers_locked()
        if not frontiers:
            return None
        # Choose nearest frontier within a max radius (prevents chasing far-away noise).
        best = None
        best_d = None
        for f in frontiers:
            d = self._manhattan(cur, f)
            if d > int(self._frontier_max_radius_cells):
                continue
            if best is None or (best_d is not None and d < best_d):
                best = f
                best_d = d
        # If nothing within radius, fall back to globally nearest.
        if best is None:
            for f in frontiers:
                d = self._manhattan(cur, f)
                if best is None or (best_d is not None and d < best_d):
                    best = f
                    best_d = d
        return best

    def _get_active_frontier_target_locked(self) -> tuple[int, int] | None:
        """Return the current persisted frontier target, choosing a new one if needed."""
        try:
            # Clear if reached/visited.
            if self._frontier_active_target is not None:
                if self._frontier_active_target in self._visited:
                    self._frontier_active_target = None
                    self._frontier_target_failures = 0

            if self._frontier_active_target is None:
                t = self._choose_frontier_target_locked()
                self._frontier_active_target = t
                self._frontier_target_failures = 0
                try:
                    self._frontier_target_set_time = float(time.time())
                except Exception:
                    self._frontier_target_set_time = 0.0
            return self._frontier_active_target
        except Exception:
            return None

    def _safe_forward_duration_for_cm(self, desired_cm: float, sensor_readings, camera_analysis, speed_pct: float = 100.0) -> float:
        """Compute a safe forward duration (seconds) to cover up to desired_cm, capped by sensors/vision.

        This intentionally biases conservative (to avoid wall hits) when vision is the only distance source.
        """
        try:
            desired_cm = max(0.0, float(desired_cm))
        except Exception:
            desired_cm = 0.0
        if desired_cm <= 0.0:
            return 0.0

        def _safe_float(v):
            try:
                x = float(v)
                return x if x > 0 else None
            except Exception:
                return None

        # Forward clearance must use FRONT sensors only (left/right). Center sensor is mounted on the BACK.
        left_cm = _safe_float(sensor_readings[0]) if sensor_readings and len(sensor_readings) > 0 else None
        right_cm = _safe_float(sensor_readings[2]) if sensor_readings and len(sensor_readings) > 2 else None
        front_vals = [v for v in (left_cm, right_cm) if v is not None]
        front_min_cm = min(front_vals) if front_vals else None
        vision_cm = None
        vision_est = None
        obstacle_size = None
        if camera_analysis and isinstance(camera_analysis, dict):
            vision_cm = _safe_float(camera_analysis.get('distance_cm'))
            vision_est = str(camera_analysis.get('distance_estimate', '')).strip().lower() or None
            obstacle_size = _safe_float(camera_analysis.get('obstacle_size'))

        # Effective commanded speed in cm/s.
        try:
            sp = float(speed_pct)
        except Exception:
            sp = 100.0
        sp = max(0.0, min(100.0, sp))
        speed_scale = max(0.05, min(1.0, sp / 100.0))
        cm_per_sec = float(self._cm_per_sec) * speed_scale

        # If we have a vision estimate label but no numeric cm, fall back to a mapping.
        # Camera is the PRIMARY forward distance source since left/right sensors are angled.
        if vision_cm is None and vision_est:
            vision_cm = {
                'very_close': 15.0,
                'close': 35.0,
                'medium': 70.0,
                'far': 120.0,
                'clear': 180.0,
            }.get(vision_est, None)

        # Vision is the main forward estimator. Apply modest pessimization only.
        # Ultrasonic left/right sensors catch immediate obstacles but may not see straight ahead.
        vision_only = (front_min_cm is None) and (vision_cm is not None)
        if vision_only:
            # Light correction: camera tends to be slightly optimistic
            vision_cm = max(0.0, float(vision_cm) * 0.85 - 5.0)

        # Fuse by taking the most conservative (minimum) available front clearance.
        eff_cm = None
        for cand in (front_min_cm, vision_cm):
            if cand is None:
                continue
            eff_cm = cand if eff_cm is None else min(eff_cm, cand)

        # Enforce minimum clearance to proceed forward.
        try:
            if eff_cm is None:
                return 0.0
            if float(eff_cm) < float(self._min_proceed_cm):
                return 0.0
        except Exception:
            return 0.0

        # Base target duration from desired distance.
        target = desired_cm / cm_per_sec if cm_per_sec > 0 else 0.0

        if eff_cm is not None:
            # Scale brake distance with speed (reaction time ~0.2s), but never below configured minimum.
            dynamic_brake_cm = max(float(self._brake_distance_cm), cm_per_sec * 0.20)
            dynamic_stop_cm = float(self._motor_stopping_distance_cm) * (0.6 + 0.4 * speed_scale)

            effective_avail = float(eff_cm) - dynamic_brake_cm - dynamic_stop_cm
            if effective_avail <= float(self._safety_margin_cm):
                return 0.0

            avail_cm = max(0.0, effective_avail - float(self._safety_margin_cm))

            # Do not plan to consume the entire clearance.
            # If relying on vision-only, use a smaller fraction.
            clearance_fraction = 0.45 if vision_only else 0.55
            max_safe_dur = (avail_cm * clearance_fraction) / cm_per_sec if cm_per_sec > 0 else 0.0
            target = min(target, max_safe_dur)

            # Additional vision-based shortening.
            if vision_est in ('very_close', 'close'):
                target *= 0.4
            elif vision_est == 'medium':
                target *= 0.7

            if obstacle_size is not None:
                target *= max(0.20, 1.0 - 0.80 * max(0.0, min(1.0, obstacle_size)))

            # If clearance is extremely tight, don't creep.
            if float(eff_cm) < (float(self._emergency_stop_cm) + dynamic_brake_cm + 5.0):
                return 0.0

        target = max(0.0, min(float(self._forward_max_duration), float(target)))
        if target < float(self._forward_min_duration):
            return 0.0
        return float(target)

    def _median(self, values: deque[float]) -> float | None:
        try:
            if not values:
                return None
            s = sorted(float(v) for v in values if v is not None)
            if not s:
                return None
            return float(s[len(s) // 2])
        except Exception:
            return None

    def _filter_ultrasonic_readings(self, raw_readings):
        """Return debounced ultrasonic readings [L, C, R] using a median of recent samples.

        This is for planning/decision stability only; hard safety still uses raw reads.
        """

        def _push(hist: deque[float], v):
            try:
                if v is None:
                    return
                x = float(v)
                if x <= float(ULTRASONIC_MIN_VALID_CM):
                    return
                hist.append(x)
            except Exception:
                return

        try:
            l = raw_readings[0] if raw_readings and len(raw_readings) > 0 else None
            c = raw_readings[1] if raw_readings and len(raw_readings) > 1 else None
            r = raw_readings[2] if raw_readings and len(raw_readings) > 2 else None
        except Exception:
            l = c = r = None

        _push(self._ultra_hist_left, l)
        _push(self._ultra_hist_center, c)
        _push(self._ultra_hist_right, r)

        fl = self._median(self._ultra_hist_left)
        fc = self._median(self._ultra_hist_center)
        fr = self._median(self._ultra_hist_right)

        # Forward clearance uses front sensors only.
        fwd_vals = [v for v in (fl, fr) if v is not None]
        fwd_min = min(fwd_vals) if fwd_vals else None

        # Forward hysteresis: once blocked, require clear readings for N cycles before allowing FORWARD.
        try:
            if fwd_min is not None and float(fwd_min) <= float(ULTRASONIC_STOP_DISTANCE_CM):
                self._ultra_forward_blocked = True
                self._ultra_forward_clear_count = 0
            elif self._ultra_forward_blocked:
                release_cm = float(ULTRASONIC_STOP_DISTANCE_CM) + float(self._ultra_hysteresis_cm)
                if fwd_min is not None and float(fwd_min) >= float(release_cm):
                    self._ultra_forward_clear_count += 1
                else:
                    self._ultra_forward_clear_count = 0
                if int(self._ultra_forward_clear_count) >= int(self._ultra_clear_cycles):
                    self._ultra_forward_blocked = False
                    self._ultra_forward_clear_count = 0
        except Exception:
            pass

        return [fl, fc, fr]

    def _vision_clearance_cm(self, camera_analysis) -> float | None:
        def _safe_float(v):
            try:
                x = float(v)
                return x if x > 0 else None
            except Exception:
                return None

        if not camera_analysis or not isinstance(camera_analysis, dict):
            return None
        v = _safe_float(camera_analysis.get('distance_cm'))
        if v is not None:
            return v
        est = str(camera_analysis.get('distance_estimate', '')).strip().lower() or None
        if not est:
            return None
        return {
            'very_close': 20.0,
            'close': 40.0,
            'medium': 80.0,
            'far': 140.0,
            'clear': 200.0,
        }.get(est, None)

    def _desired_forward_cm(self, sensor_readings, camera_analysis) -> float:
        """Pick desired forward distance for this step (grid by default; larger in open space)."""
        desired = float(self._grid_cm)
        if not self._open_space_cruise:
            return desired

        def _safe_float(v):
            try:
                x = float(v)
                return x if x > 0 else None
            except Exception:
                return None

        left = _safe_float(sensor_readings[0]) if sensor_readings and len(sensor_readings) > 0 else None
        right = _safe_float(sensor_readings[2]) if sensor_readings and len(sensor_readings) > 2 else None
        front_vals = [v for v in (left, right) if v is not None]
        front_min = min(front_vals) if front_vals else None
        vision = self._vision_clearance_cm(camera_analysis)
        try:
            clear_forward = bool((camera_analysis or {}).get('clear_forward', True))
        except Exception:
            clear_forward = True

        if front_min is None or vision is None:
            return desired
        if not clear_forward:
            return desired
        if float(front_min) >= float(self._open_space_min_ultra_cm) and float(vision) >= float(self._open_space_min_vision_cm):
            desired = max(desired, float(self._open_space_desired_cm))
        return float(desired)

    def _percentile(self, values: list[float], p: float) -> float | None:
        try:
            if not values:
                return None
            p = max(0.0, min(1.0, float(p)))
            s = sorted(float(v) for v in values if v is not None)
            if not s:
                return None
            if len(s) == 1:
                return float(s[0])
            idx = int(round(p * (len(s) - 1)))
            idx = max(0, min(len(s) - 1, idx))
            return float(s[idx])
        except Exception:
            return None

    def _tune_aggression_from_prescan(self, ultra_center_vals: list[float], vision_vals: list[float]) -> None:
        """Adjust exploration aggressiveness based on observed openness.

        This only adjusts *soft* planning knobs and forward-duration targets; hard safety gates remain unchanged.
        """

        def _clamp(x: float, lo: float, hi: float) -> float:
            try:
                x = float(x)
            except Exception:
                x = lo
            return max(float(lo), min(float(hi), float(x)))

        c50 = self._percentile(ultra_center_vals, 0.50)
        c75 = self._percentile(ultra_center_vals, 0.75)
        v50 = self._percentile(vision_vals, 0.50)
        v75 = self._percentile(vision_vals, 0.75)

        # Compute an "openness" score in [0,1] using conservative normalizations.
        # Values below these are considered cluttered; above these are open.
        # We use both sensors if available; if one is missing, fall back to the other.
        def _norm(dist: float | None, low: float, high: float) -> float | None:
            if dist is None:
                return None
            return _clamp((float(dist) - float(low)) / max(1e-6, float(high) - float(low)), 0.0, 1.0)

        u_score = _norm(c50, 25.0, 140.0)
        v_score = _norm(v50, 50.0, 200.0)
        if u_score is None and v_score is None:
            return
        if u_score is None:
            open_score = float(v_score)
        elif v_score is None:
            open_score = float(u_score)
        else:
            open_score = 0.55 * float(u_score) + 0.45 * float(v_score)
        open_score = _clamp(open_score, 0.0, 1.0)

        # Forward-duration target scaling: open rooms can safely use longer targets.
        dur_scale = 0.80 + 0.45 * open_score  # 0.80 .. 1.25
        new_forward_max = _clamp(float(self._base_forward_max_duration) * dur_scale, 0.2, float(self._max_forward_sec))
        self._forward_max_duration = min(float(self._max_forward_sec), float(new_forward_max))

        # Turning frequency: cluttered rooms need more turning; open rooms can cruise.
        turn_scale = 1.35 - 0.70 * open_score  # 1.35 .. 0.65
        self._turn_when_clear_prob = _clamp(float(self._base_turn_when_clear_prob) * turn_scale, 0.05, 0.90)

        # Exploration weights: shift more weight to forward in open rooms.
        f = _clamp(float(self._base_explore_forward_w) * (0.85 + 0.35 * open_score), 0.25, 0.92)
        # Keep side weights proportional to baseline split.
        side_total = max(0.0, 1.0 - f)
        base_side = max(1e-6, float(self._base_explore_left_w) + float(self._base_explore_right_w))
        l_ratio = float(self._base_explore_left_w) / base_side
        r_ratio = float(self._base_explore_right_w) / base_side
        self._explore_forward_w = float(f)
        self._explore_left_w = float(side_total * l_ratio)
        self._explore_right_w = float(side_total * r_ratio)

        # Open-space cruising thresholds tuned from observed distribution (use conservative mins).
        # If we saw a decent amount of clearance, set cruise desired distance based on 75th percentile.
        if c75 is not None and v75 is not None:
            conservative = float(min(float(c75), float(v75)))
            desired = _clamp(0.70 * conservative, float(self._grid_cm), 220.0)
            self._open_space_desired_cm = float(max(float(self._grid_cm), desired))
            # Trigger cruise when we are meaningfully above the stop distance.
            self._open_space_min_ultra_cm = _clamp(0.55 * self._open_space_desired_cm, 35.0, 140.0)
            self._open_space_min_vision_cm = _clamp(0.65 * self._open_space_desired_cm, 60.0, 220.0)
        else:
            # Partial info: keep defaults, but nudge a bit based on open_score.
            self._open_space_desired_cm = _clamp(float(self._base_open_space_desired_cm) * (0.90 + 0.25 * open_score), float(self._grid_cm), 220.0)
            self._open_space_min_ultra_cm = _clamp(float(self._base_open_space_min_ultra_cm) * (0.95 + 0.20 * open_score), 35.0, 140.0)
            self._open_space_min_vision_cm = _clamp(float(self._base_open_space_min_vision_cm) * (0.95 + 0.20 * open_score), 60.0, 220.0)

        dprint(
            SARAH_DEBUG,
            f"[PRESCAN] open_score={open_score:.2f} c50={c50} v50={v50} -> fwd_max={self._forward_max_duration:.2f}s turn_prob={self._turn_when_clear_prob:.2f} cruise_desired_cm={self._open_space_desired_cm:.0f}",
        )

    def _run_prescan(self) -> None:
        """Rotate in place and sample sensors/vision to choose initial aggressiveness."""

        if not self._prescan_enabled:
            return
        try:
            if get_current_mode() != "autonomous":
                return
        except Exception:
            return

        start = time.time()

        # Estimate degrees per step and guarantee at least one full 360 rotation.
        try:
            deg_per_step = float(self._turn_deg_per_sec) * float(self._prescan_turn_step_sec)
        except Exception:
            deg_per_step = 20.0
        if deg_per_step <= 1.0:
            deg_per_step = 20.0

        ultra_center_vals: list[float] = []
        vision_vals: list[float] = []

        angle_turned = 0.0
        step_idx = 0

        # Use the full budget (up to max seconds), but ensure >= 360 degrees turned at least once.
        # Any extra time is used to collect more samples for stability.
        while True:
            if self.stop_event.is_set():
                break
            try:
                if get_current_mode() != "autonomous":
                    break
            except Exception:
                break
            elapsed = (time.time() - start)
            if elapsed > float(self._prescan_max_seconds):
                break

            # Turn step.
            try:
                execute_command(
                    self.drive,
                    {"command": "LEFT", "duration": float(self._prescan_turn_step_sec), "speed": int(self._prescan_turn_speed), "speak": ""},
                    silent=True,
                )
            except Exception:
                pass

            try:
                angle_turned += float(deg_per_step)
            except Exception:
                angle_turned += 20.0

            # Sample ultrasonic.
            try:
                raw = read_all_sensors(self.sensors, verbose=False) if self.sensors else []
                filt = self._filter_ultrasonic_readings(raw) if raw else raw
                if filt and len(filt) >= 3:
                    fvals = []
                    try:
                        if filt[0] is not None:
                            fvals.append(float(filt[0]))
                    except Exception:
                        pass
                    try:
                        if filt[2] is not None:
                            fvals.append(float(filt[2]))
                    except Exception:
                        pass
                    if fvals:
                        cc = float(min(fvals))
                        if cc > float(ULTRASONIC_MIN_VALID_CM):
                            ultra_center_vals.append(cc)
            except Exception:
                pass

            # Sample vision every N steps (expensive).
            if (step_idx % int(self._prescan_vision_every_n)) == 0:
                try:
                    if self.camera_thread and getattr(self.camera_thread, 'camera_enabled', False):
                        # Only analyze if we can get a frame quickly.
                        got = self.camera_thread.get_frame_base64()
                        if got:
                            analysis = self.camera_analyzer.analyze_scene(img_b64=got) if self.camera_analyzer else None
                            vcm = self._vision_clearance_cm(analysis)
                            if vcm is not None:
                                vision_vals.append(float(vcm))
                except Exception:
                    pass

            step_idx += 1

            # Stop early only if we've completed a full rotation AND have enough samples.
            # Otherwise keep using the time budget to stabilize the estimate.
            try:
                completed_rotation = float(angle_turned) >= 360.0
            except Exception:
                completed_rotation = False

            if completed_rotation:
                have_ultra = len(ultra_center_vals) >= 8
                cam_enabled = bool(self.camera_thread and getattr(self.camera_thread, 'camera_enabled', False))
                have_vision = (len(vision_vals) >= 3) if cam_enabled else True
                if have_ultra and have_vision:
                    # If there's still lots of time remaining, we can keep sampling,
                    # but stop once we are within ~1s of the budget to avoid long startup.
                    if (float(self._prescan_max_seconds) - elapsed) <= 1.0:
                        break

        # Apply tuning.
        try:
            self._tune_aggression_from_prescan(ultra_center_vals, vision_vals)
        except Exception:
            pass

        try:
            self._last_prescan_time = float(time.time())
        except Exception:
            self._last_prescan_time = 0.0

    def _frontier_plan_command(self, sensor_readings, camera_analysis) -> dict | None:
        """Return a deterministic command that moves toward the nearest frontier cell."""
        def _safe_float(v):
            try:
                x = float(v)
                return x if x > 0 else None
            except Exception:
                return None

        left_cm = _safe_float(sensor_readings[0]) if sensor_readings and len(sensor_readings) > 0 else None
        center_cm = _safe_float(sensor_readings[1]) if sensor_readings and len(sensor_readings) > 1 else None
        right_cm = _safe_float(sensor_readings[2]) if sensor_readings and len(sensor_readings) > 2 else None

        # Forward clearance uses FRONT sensors only (left/right). Center is mounted on the BACK.
        front_vals = [v for v in (left_cm, right_cm) if v is not None]
        front_min = min(front_vals) if front_vals else None

        clear_left = bool((camera_analysis or {}).get('clear_left', (left_cm is None or left_cm >= self._clear_side_cm)))
        clear_right = bool((camera_analysis or {}).get('clear_right', (right_cm is None or right_cm >= self._clear_side_cm)))
        clear_forward = bool((camera_analysis or {}).get('clear_forward', (front_min is not None and front_min >= self._clear_forward_cm)))

        with self._pose_lock:
            cur = self._grid_cell_from_xy(self._x_cm, self._y_cm)
            target = self._get_active_frontier_target_locked()
            neigh = self._neighbor_cells_from_pose_locked()

        if target is None:
            return None

        # If we are already at the target (or it became visited), pick a fresh one.
        if cur == target:
            with self._pose_lock:
                self._frontier_active_target = None
                self._frontier_target_failures = 0
                target = self._get_active_frontier_target_locked()
                neigh = self._neighbor_cells_from_pose_locked()
                cur = self._grid_cell_from_xy(self._x_cm, self._y_cm)
            if target is None:
                return None

        # Choose the neighbor that reduces Manhattan distance to the target.
        def score(cell: tuple[int, int]) -> int:
            return self._manhattan(cell, target)

        candidates: list[tuple[str, tuple[int, int]]] = [('F', neigh['F']), ('L', neigh['L']), ('R', neigh['R']), ('B', neigh['B'])]
        candidates.sort(key=lambda it: score(it[1]))

        # Prefer a move that also goes to an unvisited cell when possible.
        def is_unvisited(c: tuple[int, int]) -> bool:
            try:
                with self._pose_lock:
                    return (c not in self._visited) and (c not in self._blocked)
            except Exception:
                return False

        best_dir = None
        for d, c in candidates:
            if score(c) >= score(cur):
                continue
            try:
                with self._pose_lock:
                    if c in self._blocked:
                        continue
            except Exception:
                pass
            if d == 'F' and not clear_forward:
                continue
            if d == 'L' and not clear_left:
                continue
            if d == 'R' and not clear_right:
                continue
            # Backward is a last resort; only allow if we have no forward/side option.
            if d == 'B':
                continue
            best_dir = d
            if is_unvisited(c):
                break

        if best_dir is None:
            # If we can't reduce distance, try to step into any unvisited adjacent cell that is clear.
            if clear_forward and is_unvisited(neigh['F']):
                best_dir = 'F'
            elif clear_left and is_unvisited(neigh['L']):
                best_dir = 'L'
            elif clear_right and is_unvisited(neigh['R']):
                best_dir = 'R'

        if best_dir is None:
            # Still nothing: choose a turn toward any clear side to escape loops.
            if clear_left or clear_right:
                best_dir = 'L' if (clear_left and (not clear_right or random.random() < 0.5)) else 'R'
            else:
                with self._pose_lock:
                    self._frontier_target_failures += 1
                    if int(self._frontier_target_failures) >= int(self._frontier_target_max_failures):
                        self._frontier_active_target = None
                        self._frontier_target_failures = 0
                return {"command": "STOP", "duration": 0, "speed": 0, "speak": "Blocked - stopping.", "emotion": "concerned"}

        # Convert plan direction into an executable command.
        if best_dir == 'F':
            desired_cm = self._desired_forward_cm(sensor_readings, camera_analysis)
            dur = self._safe_forward_duration_for_cm(desired_cm, sensor_readings, camera_analysis, speed_pct=100.0)
            if dur <= 0:
                # Can't safely advance a grid cell; turn to search for a new corridor.
                if clear_left or clear_right:
                    turn = 'LEFT' if (clear_left and (not clear_right or random.random() < 0.5)) else 'RIGHT'
                    with self._pose_lock:
                        self._frontier_target_failures += 1
                        if int(self._frontier_target_failures) >= int(self._frontier_target_max_failures):
                            self._frontier_active_target = None
                            self._frontier_target_failures = 0
                    return {"command": turn, "duration": float(self._turn_duration), "speed": 100, "speak": "Finding a new path.", "emotion": "thinking"}
                with self._pose_lock:
                    self._frontier_target_failures += 1
                    if int(self._frontier_target_failures) >= int(self._frontier_target_max_failures):
                        self._frontier_active_target = None
                        self._frontier_target_failures = 0
                return {"command": "STOP", "duration": 0, "speed": 0, "speak": "Too close ahead - stopping.", "emotion": "concerned"}
            if self._frontier_debug:
                print(f"[FRONTIER] cur={cur} target={target} step=F dur={dur:.2f}s failures={self._frontier_target_failures}")
            with self._pose_lock:
                self._frontier_target_failures = 0
            return {"command": "FORWARD", "duration": float(dur), "speed": 100, "speak": "Exploring forward.", "emotion": "excited"}

        # Turning: use about 90 degrees to align to the neighbor direction.
        turn_cmd = 'LEFT' if best_dir == 'L' else 'RIGHT'
        # 90deg / deg_per_sec = duration
        turn_dur = 90.0 / float(self._turn_deg_per_sec) if float(self._turn_deg_per_sec) > 0 else float(self._turn_duration)
        turn_dur = min(float(self._turn_duration), float(turn_dur), float(AUTONOMOUS_MAX_MOVE_DURATION))
        if self._frontier_debug:
            print(f"[FRONTIER] cur={cur} target={target} step={best_dir} turn={turn_cmd} dur={turn_dur:.2f}s failures={self._frontier_target_failures}")
        with self._pose_lock:
            self._frontier_target_failures = 0
        return {"command": turn_cmd, "duration": float(turn_dur), "speed": 100, "speak": "Repositioning.", "emotion": "thinking"}

    def _mark_visited_locked(self) -> None:
        try:
            self._visited.add(self._grid_cell_from_xy(self._x_cm, self._y_cm))
        except Exception:
            pass

    def _apply_motion_locked(self, command: str, duration: float, speed: int) -> None:
        cmd = (command or '').strip().upper()
        try:
            dur = max(0.0, float(duration or 0))
        except Exception:
            dur = 0.0
        try:
            spd = int(speed)
        except Exception:
            spd = 100

        # Speed scale: forward/back distance scales with PWM; turning is approximate.
        speed_scale = max(0.0, min(1.0, spd / 100.0))

        if cmd in ("FORWARD", "BACKWARD") and dur > 0:
            dist_cm = float(self._cm_per_sec) * dur * speed_scale
            if cmd == "BACKWARD":
                dist_cm = -dist_cm
            rad = math.radians(float(self._heading_deg))
            self._x_cm += math.cos(rad) * dist_cm
            self._y_cm += math.sin(rad) * dist_cm
            self._mark_visited_locked()
            return

        if cmd in ("LEFT", "RIGHT") and dur > 0:
            delta = float(self._turn_deg_per_sec) * dur
            if cmd == "RIGHT":
                delta = -delta
            self._heading_deg = (float(self._heading_deg) + delta) % 360.0
            # Turning in place doesn't change cell, but still mark as visited.
            self._mark_visited_locked()
            return

    def _record_motion(self, command: str, duration: float, speed: int) -> None:
        cmd = (command or '').strip().upper()
        if cmd not in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
            return
        try:
            dur = float(duration or 0)
        except Exception:
            dur = 0.0
        if dur <= 0:
            return
        with self._pose_lock:
            self._motion_history.append({"command": cmd, "duration": float(dur), "speed": int(speed)})
            # Keep history bounded.
            if len(self._motion_history) > 200:
                self._motion_history = self._motion_history[-200:]
            self._apply_motion_locked(cmd, float(dur), int(speed))

    def _next_return_step(self) -> dict | None:
        with self._pose_lock:
            if not self._motion_history:
                return None
            last = self._motion_history.pop()
        cmd = str(last.get('command', 'STOP')).upper()
        dur = float(last.get('duration', 0) or 0)
        spd = int(last.get('speed', 100) or 100)
        inv = {
            'FORWARD': 'BACKWARD',
            'BACKWARD': 'FORWARD',
            'LEFT': 'RIGHT',
            'RIGHT': 'LEFT',
        }.get(cmd, 'STOP')

        # Conservative: keep return moves short.
        if inv in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
            cap = self._forward_max_duration if inv in ("FORWARD", "BACKWARD") else self._turn_duration
            dur = min(float(dur), float(cap), AUTONOMOUS_MAX_MOVE_DURATION)
        else:
            dur = 0.0
            spd = 0
        return {"command": inv, "duration": float(dur), "speed": int(spd), "speak": "Returning to start.", "emotion": "thinking"}

    def run(self):
        print("[AUTO] Vision Autonomous thread starting.")
        
        # Announce mapping/persistence status.
        try:
            if bool(getattr(self, '_persist_enabled', True)):
                persist_path = str(getattr(self, '_persist_path', 'sarah_auto_persist.json'))
                print(f"[AUTO] Persistent mapping ENABLED (file: {os.path.basename(persist_path)})")
            else:
                print("[AUTO] Persistent mapping DISABLED (SARAH_AUTO_PERSIST=0)")
        except Exception:
            pass
        
        # Warmup camera
        # Use stop_event-aware wait so shutdown/emergency stop isn't delayed.
        self.stop_event.wait(2)
        decision_count = 0

        # Option J: load persistent map/memory once per thread lifetime.
        try:
            self._load_persistent_state()
            # Initialize save timer so we don't immediately rewrite the file on startup.
            self._persist_last_save_ts = float(time.time())
        except Exception:
            pass
        
        # Exploration memory: stores rich data (vision, sensors, pose, commands)
        # Configurable size via SARAH_AUTO_MEMORY_SIZE (default: 20 for longer-term learning)
        try:
            memory_max_size = max(5, int(os.getenv('SARAH_AUTO_MEMORY_SIZE', '20').strip() or '20'))
        except Exception:
            memory_max_size = 20
        exploration_memory = []  # Track what we've seen for better decision-making

        last_mode = None
        mode_seq = _get_mode_seq()

        last_frame_ok = None  # track camera availability to avoid repeating the same warning every cycle
        
        # LAWNMOWER PATTERN: Track exploration depth for systematic coverage
        lawnmower_depth = 0  # Positive = going deeper, negative = backing out
        lawnmower_phase = "explore"  # "explore" or "retreat"
        consecutive_forwards = 0
        last_turn_direction = None

        # Corner recovery: if front remains blocked for multiple cycles, force a back+turn escape.
        blocked_front_cycles = 0
        last_corner_escape_ts = 0.0

        # Fluidity budgets: keep perception/planning from stalling motion.
        last_vision_ts = 0.0
        last_vision_analysis = None
        
        while not self.stop_event.is_set():
            # Allow a runtime reset of the persistent map/memory from ANY mode.
            try:
                if RESET_MAP_MEMORY_EVENT.is_set():
                    try:
                        RESET_MAP_MEMORY_EVENT.clear()
                    except Exception:
                        pass
                    self._reset_map_memory_runtime()
            except Exception:
                pass

            # Only run autonomous decisions while in AUTONOMOUS mode.
            current_mode = get_current_mode()
            if last_mode != current_mode:
                # If we are leaving autonomous mode, ensure motors are stopped once.
                if last_mode == "autonomous" and current_mode != "autonomous":
                    try:
                        if self.drive:
                            self.drive.stop()
                    except Exception:
                        pass
                    # Persist what we learned before leaving autonomous.
                    try:
                        self._maybe_persist(force=True)
                    except Exception:
                        pass
                # If we are entering autonomous mode, show map status.
                elif last_mode != "autonomous" and current_mode == "autonomous":
                    # Remind user that mapping is active when entering autonomous.
                    try:
                        if bool(getattr(self, '_persist_enabled', True)):
                            with self._pose_lock:
                                v_count = len(self._visited)
                                b_count = len(self._blocked)
                            if v_count > 1 or b_count > 0:
                                dprint(True, f"[AUTO] Resuming with map: {v_count} visited, {b_count} blocked cells")
                    except Exception:
                        pass
                
                last_mode = current_mode

                # Track mode change sequence for efficient waiting when inactive.
                mode_seq = _get_mode_seq()

            if current_mode != "autonomous":
                # Event-driven wait reduces CPU while we are not the active mode.
                # Note: timeout ensures we still check stop_event periodically
                mode_seq = _wait_for_mode_change(mode_seq, timeout=0.5)
                continue

            # One-time prescan at autonomous entry.
            if not bool(self._did_prescan):
                try:
                    self._did_prescan = True
                    self._run_prescan()
                except Exception:
                    pass

            # TIME-BASED CYCLE: Track cycle start time to ensure decisions every 2 seconds
            cycle_start_time = time.time()

            # Hard time budget for analysis + planning (movement still executes after).
            try:
                cycle_budget_s = float(os.getenv('SARAH_AUTO_CYCLE_BUDGET_S', '1.2').strip() or '1.2')
            except Exception:
                cycle_budget_s = 1.2
            cycle_budget_s = max(0.4, float(cycle_budget_s))
            
            try:
                decision_count += 1
                dprint(SARAH_DEBUG, f"\n[AUTO] *** DECISION CYCLE #{decision_count} ***")

                # Explicit explore request clears the exhausted flag (but keeps memory/history so we can still return home).
                if EXPLORE_REQUEST_EVENT.is_set():
                    try:
                        EXPLORE_REQUEST_EVENT.clear()
                    except Exception:
                        pass
                    self._explore_exhausted = False
                    self._exhausted_cycles = 0
                
                # 1. Capture latest frame for visual analysis (PRIORITY: use camera actively)
                latest_img_b64 = None
                camera_analysis = None
                
                # Get the most recent frame without long blocking waits. Encoding is cached in CameraThread.
                latest_img_b64 = None
                if self.camera_thread and self.camera_thread.camera_enabled:
                    latest_img_b64 = self.camera_thread.get_frame_base64()
                    if not latest_img_b64:
                        # Short wait budget to allow first frame / slow camera startup without stalling the mode loop.
                        if self.camera_thread.wait_for_frame(timeout=0.2):
                            latest_img_b64 = self.camera_thread.get_frame_base64()
                
                # 1.5. Read sensors early (needed for safety + decision-making)
                # DIAGNOSTIC: Force verbose reading for first 10 cycles to diagnose sensor issues
                verbose_sensors = (decision_count <= 10)
                sensor_readings_raw = read_all_sensors(self.sensors, verbose=verbose_sensors) if self.sensors else []
                sensor_readings_raw_ts = time.time()
                
                # DIAGNOSTIC: Check if sensors are working
                if decision_count % 5 == 0 or decision_count <= 3:  # First 3 cycles + every 5 cycles
                    valid_sensor_count = sum(1 for d in sensor_readings_raw if d is not None and d > 0)
                    if verbose_sensors or valid_sensor_count == 0:
                        print(f"[AUTO-DIAGNOSTIC] Cycle {decision_count}: {valid_sensor_count}/{len(self.sensors) if self.sensors else 0} sensors reading valid data")
                        if valid_sensor_count == 0 and self.sensors:
                            print(f"[AUTO-WARNING] ⚠️ NO VALID ULTRASONIC READINGS - Robot is BLIND!")
                            print(f"[AUTO-WARNING] Check: 1) Power to sensors, 2) Wiring on pins {ULTRASONIC_PINS}, 3) GPIO initialization")
                            for i, s in enumerate(self.sensors):
                                label = ['left', 'center', 'right'][i] if i < 3 else f'sensor{i}'
                                print(f"[AUTO-DIAGNOSTIC] {label}: enabled={s.enabled}, trigger=GPIO{s.trigger_pin}, echo=GPIO{s.echo_pin}")

                # SAFETY: Multi-tier ultrasonic protection.
                # IMPORTANT: Sensor 1 (center) is on the BACK - only use left/right (0,2) for forward obstacle detection
                try:
                    # Only use valid positive readings from working sensors
                    valid = [float(d) for d in (sensor_readings_raw or []) if d is not None and float(d) > 0]
                    # For FORWARD movement safety, only check FRONT sensors (left=0, right=2), ignore back sensor (center=1)
                    front_sensors = []
                    if sensor_readings_raw and isinstance(sensor_readings_raw, list) and len(sensor_readings_raw) >= 3:
                        # Safer array access with explicit bounds checking
                        left_val = sensor_readings_raw[0] if len(sensor_readings_raw) > 0 else None
                        right_val = sensor_readings_raw[2] if len(sensor_readings_raw) > 2 else None
                        if left_val is not None and float(left_val) > 0:
                            front_sensors.append(float(left_val))  # Left
                        if right_val is not None and float(right_val) > 0:
                            front_sensors.append(float(right_val))  # Right
                except Exception:
                    valid = []
                    front_sensors = []
                
                min_dist = min(valid) if valid else None
                min_front = min(front_sensors) if front_sensors else None  # Front obstacle distance only

                # If we recently hit an obstacle mid-move, treat it as still present for a short time.
                try:
                    global _LAST_AUTO_FORWARD_MIDSTOP_TS, _LAST_AUTO_FORWARD_MIDSTOP_CM
                    if _LAST_AUTO_FORWARD_MIDSTOP_CM is not None and (time.time() - float(_LAST_AUTO_FORWARD_MIDSTOP_TS)) < 2.0:
                        try:
                            remembered = float(_LAST_AUTO_FORWARD_MIDSTOP_CM)
                            if min_front is None:
                                min_front = remembered
                            else:
                                min_front = min(float(min_front), remembered)
                        except Exception:
                            pass
                except Exception:
                    pass
                
                # If we have at least 1 working sensor, use it. If 0 sensors work, min_dist will be None
                working_sensor_count = len(valid)
                if decision_count % 10 == 1:  # Periodic reminder
                    if working_sensor_count == 0:
                        dprint(True, f"[AUTO-WARNING] No ultrasonic readings - navigating with vision only")
                    elif working_sensor_count < 3:
                        dprint(True, f"[AUTO-INFO] Operating with {working_sensor_count}/3 sensors - back={sensor_readings_raw[1] if sensor_readings_raw and len(sensor_readings_raw) > 1 else 'N/A'}cm, front L/R={front_sensors}")
                
                # EMERGENCY: Very close obstacle IN FRONT - stop immediately and back up.
                if min_front is not None and min_front <= self._emergency_stop_cm:
                    cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Emergency stop - very close obstacle"}
                    print(f"[AUTO] EMERGENCY STOP: front obstacle at {min_front:.1f}cm <= {self._emergency_stop_cm:.1f}cm")
                    execute_command(self.drive, cmd, silent=True)
                    
                    # Extract back sensor and front sensors
                    back_cm = None
                    try:
                        if sensor_readings_raw and len(sensor_readings_raw) >= 2:
                            back_cm = float(sensor_readings_raw[1]) if (sensor_readings_raw[1] is not None and float(sensor_readings_raw[1]) > 0) else None
                    except Exception:
                        back_cm = None

                    left_cm = None
                    right_cm = None
                    try:
                        if sensor_readings_raw and len(sensor_readings_raw) >= 3:
                            left_cm = float(sensor_readings_raw[0]) if (sensor_readings_raw[0] is not None and float(sensor_readings_raw[0]) > 0) else None
                            right_cm = float(sensor_readings_raw[2]) if (sensor_readings_raw[2] is not None and float(sensor_readings_raw[2]) > 0) else None
                    except Exception:
                        left_cm = right_cm = None

                    # Remember blocked directions so we don't keep targeting the same wall.
                    try:
                        with self._pose_lock:
                            self._mark_blocked_from_sensors_locked(min_front, left_cm, right_cm)
                    except Exception:
                        pass

                    # Update persistent relative map from this near-collision event.
                    try:
                        self._update_relative_map(sensor_readings_raw, min_front, None)
                        self._maybe_persist()
                    except Exception:
                        pass

                    # ALWAYS back up first when stuck at obstacle (helps escape repeated blocking)
                    backup_cmd = None
                    if back_cm is not None and back_cm > 20.0:
                        print(f"[AUTO] Backing up (back clear: {back_cm:.1f}cm)")
                        backup_cmd = {"command": "BACKWARD", "duration": 0.25, "speed": 70, "speak": ""}
                    else:
                        # Back sensor blocked or uncertain - do minimal backup
                        print(f"[AUTO] Minimal backup (back: {back_cm})")
                        backup_cmd = {"command": "BACKWARD", "duration": 0.1, "speed": 60, "speak": ""}
                    
                    execute_command(self.drive, backup_cmd, silent=True)
                    # Record backup movement for pose tracking (critical for exploration memory)
                    try:
                        self._record_motion(backup_cmd['command'], backup_cmd['duration'], backup_cmd['speed'])
                    except Exception:
                        pass

                    # Turn away (if right is close, turn left; if left close, turn right)
                    # Hysteresis: if we just evaded, keep the same direction briefly to avoid spinning.
                    g = globals()
                    last_dir = g.get('_LAST_EVADE_DIR', None)
                    try:
                        last_ts = float(g.get('_LAST_EVADE_TS', 0.0) or 0.0)
                    except Exception:
                        last_ts = 0.0

                    chosen_dir = None
                    try:
                        if last_dir in ("LEFT", "RIGHT") and (time.time() - last_ts) < 1.5:
                            chosen_dir = str(last_dir)
                    except Exception:
                        chosen_dir = None

                    if chosen_dir is None:
                        if right_cm is not None and (left_cm is None or right_cm <= left_cm):
                            chosen_dir = "LEFT"
                        else:
                            chosen_dir = "RIGHT"
                        try:
                            g['_LAST_EVADE_DIR'] = chosen_dir
                            g['_LAST_EVADE_TS'] = time.time()
                        except Exception:
                            pass

                    evade = {"command": chosen_dir, "duration": 0.9, "speed": 100, "speak": ""}
                    execute_command(self.drive, evade, silent=True)
                    # Record turn for pose tracking
                    try:
                        self._record_motion(evade['command'], evade['duration'], evade['speed'])
                    except Exception:
                        pass

                    # Persist after escape motion.
                    try:
                        self._maybe_persist()
                    except Exception:
                        pass
                    self.stop_event.wait(0.05)
                    continue

                # Debounced readings for planning (reduces twitchy decisions); still preserve raw for hard safety.
                try:
                    sensor_readings = self._filter_ultrasonic_readings(sensor_readings_raw)
                    if not sensor_readings or all(v is None for v in sensor_readings):
                        sensor_readings = sensor_readings_raw
                except Exception:
                    sensor_readings = sensor_readings_raw

                sensor_text = ""
                if sensor_readings:
                    try:
                        sensor_text = (f"Ultrasonic sensors (cm) - left(front)={sensor_readings[0]}, "
                                       f"center(back)={sensor_readings[1]}, right(front)={sensor_readings[2]}. ")
                    except Exception:
                        sensor_text = ""
                
                # REGULAR: Close obstacle IN FRONT - stop and back up before turning.
                if min_front is not None and min_front <= ULTRASONIC_STOP_DISTANCE_CM:
                    print(f"[AUTO] Safety STOP: front obstacle at {min_front:.1f}cm <= {ULTRASONIC_STOP_DISTANCE_CM:.1f}cm")
                    execute_command(self.drive, {"command": "STOP", "duration": 0, "speed": 0, "speak": ""}, silent=True)

                    # Extract back and front sensor readings
                    back_cm = None
                    try:
                        if sensor_readings_raw and len(sensor_readings_raw) >= 2:
                            back_cm = float(sensor_readings_raw[1]) if (sensor_readings_raw[1] is not None and float(sensor_readings_raw[1]) > 0) else None
                    except Exception:
                        back_cm = None

                    left_cm = None
                    right_cm = None
                    try:
                        if sensor_readings_raw and len(sensor_readings_raw) >= 3:
                            left_cm = float(sensor_readings_raw[0]) if (sensor_readings_raw[0] is not None and float(sensor_readings_raw[0]) > 0) else None
                            right_cm = float(sensor_readings_raw[2]) if (sensor_readings_raw[2] is not None and float(sensor_readings_raw[2]) > 0) else None
                    except Exception:
                        left_cm = right_cm = None

                    # Remember blocked directions.
                    try:
                        with self._pose_lock:
                            self._mark_blocked_from_sensors_locked(min_front, left_cm, right_cm)
                    except Exception:
                        pass

                    # Update persistent relative map from this blocked-front event.
                    try:
                        self._update_relative_map(sensor_readings_raw, min_front, None)
                        self._maybe_persist()
                    except Exception:
                        pass

                    # Back up before turning (helps when repeatedly hitting same obstacle)
                    if back_cm is not None and back_cm > 20.0:
                        print(f"[AUTO] Backing up (back clear: {back_cm:.1f}cm)")
                        backup_cmd = {"command": "BACKWARD", "duration": 0.2, "speed": 70, "speak": ""}
                        execute_command(self.drive, backup_cmd, silent=True)
                        # Record backup for pose tracking
                        try:
                            self._record_motion(backup_cmd['command'], backup_cmd['duration'], backup_cmd['speed'])
                        except Exception:
                            pass

                    # Hysteresis: if we just evaded, keep the same direction briefly.
                    g = globals()
                    last_dir = g.get('_LAST_EVADE_DIR', None)
                    try:
                        last_ts = float(g.get('_LAST_EVADE_TS', 0.0) or 0.0)
                    except Exception:
                        last_ts = 0.0

                    chosen_dir = None
                    try:
                        if last_dir in ("LEFT", "RIGHT") and (time.time() - last_ts) < 1.5:
                            chosen_dir = str(last_dir)
                    except Exception:
                        chosen_dir = None

                    if chosen_dir is None:
                        if right_cm is not None and (left_cm is None or right_cm <= left_cm):
                            chosen_dir = "LEFT"
                        else:
                            chosen_dir = "RIGHT"
                        try:
                            g['_LAST_EVADE_DIR'] = chosen_dir
                            g['_LAST_EVADE_TS'] = time.time()
                        except Exception:
                            pass

                    evade = {"command": chosen_dir, "duration": 0.7, "speed": 100, "speak": ""}
                    execute_command(self.drive, evade, silent=True)
                    # Record turn for pose tracking
                    try:
                        self._record_motion(evade['command'], evade['duration'], evade['speed'])
                    except Exception:
                        pass

                    # Persist after escape motion.
                    try:
                        self._maybe_persist()
                    except Exception:
                        pass
                    self.stop_event.wait(0.05)
                    continue

                # RETURN-TO-START: retrace path without querying the model.
                if RETURN_TO_START_EVENT.is_set():
                    step = self._next_return_step()
                    if not step:
                        RETURN_TO_START_EVENT.clear()
                        cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Arrived at start.", "emotion": "happy"}
                        execute_command(self.drive, cmd, silent=True)
                        self.stop_event.wait(0.1)
                        continue

                    # Basic safety: if we're about to go FORWARD and something is close, stop and wait.
                    if str(step.get('command', '')).upper() == 'FORWARD' and min_front is not None and min_front <= self._safety_margin_cm:
                        execute_command(self.drive, {"command": "STOP", "duration": 0, "speed": 0, "speak": "Blocked while returning.", "emotion": "concerned"}, silent=True)
                        self.stop_event.wait(AUTONOMOUS_OBSTACLE_PAUSE_SECONDS)
                        continue

                    execute_command(self.drive, step, silent=True)
                    # Apply motion update without re-recording (we popped history already).
                    with self._pose_lock:
                        self._apply_motion_locked(str(step.get('command', 'STOP')), float(step.get('duration', 0) or 0), int(step.get('speed', 100) or 100))
                    self.stop_event.wait(0.05)
                    continue
                
                if latest_img_b64:
                    if last_frame_ok is not True:
                        print("[AUTO] Camera frame acquired.")
                    else:
                        dprint(SARAH_DEBUG, "[AUTO] Frame acquired - running Llava analysis then Llama decision")
                    last_frame_ok = True
                    # Smart two-stage pipeline:
                    # 1) Llava analyzes the image (structured safety fields)
                    # 2) Llama3.1 chooses the movement using sensors + llava analysis + memory
                    # OPTIMIZATION: Skip slow vision analysis if front sensors show clear path
                    # STRATEGY: Use AI every 2nd cycle only for 3-second fluid movement
                    # SAFETY: Never skip vision if we have no valid front sensor data
                    skip_vision = False

                    # If we are already over budget this cycle, skip heavy work and keep moving.
                    try:
                        if (time.time() - float(cycle_start_time)) >= float(cycle_budget_s):
                            # Never skip vision when front sensors are unavailable; otherwise we'd be blind.
                            # In that case we will either use cached vision or fall back to a cautious turn.
                            if min_front is not None:
                                skip_vision = True
                                dprint(SARAH_DEBUG, f"[AUTO-BUDGET] Over cycle budget ({cycle_budget_s:.1f}s) - skipping vision")
                            else:
                                skip_vision = False
                                dprint(SARAH_DEBUG, f"[AUTO-BUDGET] Over cycle budget but front sensors missing - keeping vision enabled")
                    except Exception:
                        pass
                    
                    # SAFETY FIRST: Always use camera vision for obstacle detection
                    # DO NOT skip vision - camera is critical for detecting walls and obstacles
                    # that ultrasonic sensors might miss (e.g., objects between sensors, thin obstacles)
                    skip_vision = False
                    
                    # Only use cached vision if very recent (< 2 seconds) AND sensors agree
                    if min_front is not None and min_front > 40.0:
                        # Check if we have very recent cached vision
                        cache_max_age_s = 2.0  # Reduced from 5s for safety
                        cache_valid = False
                        if last_vision_analysis is not None and isinstance(last_vision_analysis, dict):
                            try:
                                cache_age = time.time() - float(last_vision_time)
                                if cache_age < cache_max_age_s:
                                    # Only use cache if it also showed clear path
                                    if not last_vision_analysis.get('obstacles', True):
                                        cache_valid = True
                                        skip_vision = True
                                        dprint(SARAH_DEBUG, f"[AUTO-CACHE] Using recent cache ({cache_age:.1f}s old, sensors clear)")
                            except Exception:
                                cache_valid = False
                    
                    if not skip_vision:
                        dprint(SARAH_DEBUG, f"[AUTO-SAFETY] Using camera vision for obstacle detection (front: {min_front}cm)")
                    
                    try:
                        if skip_vision:
                            # Use fast default analysis when sensors show clear
                            # First check if we have a recent cached vision result (< 5 seconds old)
                            cache_max_age_s = 5.0
                            cache_valid = False
                            if last_vision_analysis is not None and isinstance(last_vision_analysis, dict):
                                try:
                                    cache_age = time.time() - float(last_vision_time)
                                    if cache_age < cache_max_age_s:
                                        cache_valid = True
                                except Exception:
                                    cache_valid = False
                            
                            if cache_valid:
                                camera_analysis = last_vision_analysis
                                dprint(SARAH_DEBUG, f"[AUTO-CACHE] Using cached vision analysis ({cache_age:.1f}s old)")
                            else:
                                # Use sensor-based default analysis
                                camera_analysis = {
                                    "obstacles": False,
                                    "clear_left": True,
                                    "clear_right": True,
                                    "clear_forward": True,
                                    "distance_estimate": "clear",
                                    "distance_cm": min_front if min_front is not None else 100,
                                    "confidence": 0.9,
                                    "recommendation": "PROCEED",
                                    "description": "Sensors show clear path"
                                }
                        else:
                            # Throttle vision frequency so Llava can't make motion jerky.
                            try:
                                min_interval = float(os.getenv('SARAH_AUTO_VISION_MIN_INTERVAL_S', '0.8').strip() or '0.8')
                            except Exception:
                                min_interval = 0.8
                            min_interval = max(0.0, float(min_interval))

                            now_ts = time.time()
                            recent_ok = (last_vision_analysis is not None) and ((now_ts - float(last_vision_ts)) < float(min_interval))

                            # If we're time-constrained, prefer cached analysis.
                            time_left = float(cycle_budget_s) - float(now_ts - float(cycle_start_time))
                            if recent_ok and time_left < 0.6:
                                camera_analysis = last_vision_analysis
                                dprint(SARAH_DEBUG, f"[AUTO-BUDGET] Using cached vision (age={(now_ts - float(last_vision_ts)):.2f}s)")
                            else:
                                # If we are out of time and have no cache, do NOT run blind forward.
                                # We'll allow the rest of the pipeline to pick a turn if vision can't run.
                                if time_left < 0.25 and last_vision_analysis is not None:
                                    camera_analysis = last_vision_analysis
                                    dprint(SARAH_DEBUG, f"[AUTO-BUDGET] Very low time_left; using cached vision")
                                elif time_left < 0.25 and last_vision_analysis is None:
                                    camera_analysis = None
                                    dprint(SARAH_DEBUG, f"[AUTO-BUDGET] Very low time_left; skipping vision call (no cache)")
                                else:
                                    camera_analysis = self.camera_analyzer.analyze_scene(img_b64=latest_img_b64) if self.camera_analyzer else None
                                    last_vision_ts = float(now_ts)
                                    last_vision_analysis = camera_analysis

                        # Normalize camera_analysis for downstream safety logic.
                        try:
                            if camera_analysis is not None and isinstance(camera_analysis, dict):
                                # Confidence
                                try:
                                    conf = float(camera_analysis.get('confidence', 0.0) or 0.0)
                                except Exception:
                                    conf = 0.0
                                camera_analysis['confidence'] = max(0.0, min(1.0, float(conf)))

                                # Distance
                                try:
                                    dcm = camera_analysis.get('distance_cm', None)
                                    dcm = float(dcm) if dcm is not None else None
                                except Exception:
                                    dcm = None
                                if dcm is not None and dcm <= 0:
                                    dcm = None
                                if dcm is None:
                                    # Keep a sane default; sensors will override if present.
                                    dcm = 60.0
                                camera_analysis['distance_cm'] = float(dcm)

                                # Ensure forward clearance matches obstacle flags.
                                dist_est = str(camera_analysis.get('distance_estimate', '') or '').lower()
                                obstacles = bool(camera_analysis.get('obstacles', False))
                                if obstacles or dist_est in ('very_close', 'close') or float(dcm) < 35.0:
                                    camera_analysis['clear_forward'] = False
                        except Exception:
                            pass
                    except Exception:
                        camera_analysis = None
                else:
                    if last_frame_ok is not False:
                        print("[AUTO] WARNING: No camera frame (using sensor data only).")
                    else:
                        dprint(SARAH_DEBUG, "[AUTO] No camera frame - using sensor data only")
                    last_frame_ok = False
                    camera_analysis = None
                    # Even without camera, store sensor context in memory
                    if sensor_readings:
                        try:
                            sensor_context = (f"Detected obstacles - left={sensor_readings[0]}cm, "
                                            f"center={sensor_readings[1]}cm, right={sensor_readings[2]}cm")
                            exploration_memory.append({
                                'cycle': decision_count,
                                'observation': sensor_context,
                                'timestamp': time.time(),
                                'vision_available': False,
                                'sensor_readings': {
                                    'left_cm': sensor_readings[0] if len(sensor_readings) > 0 else None,
                                    'center_cm': sensor_readings[1] if len(sensor_readings) > 1 else None,
                                    'right_cm': sensor_readings[2] if len(sensor_readings) > 2 else None,
                                    'min_front_cm': min_front
                                }
                            })
                            if len(exploration_memory) > memory_max_size:
                                exploration_memory.pop(0)
                        except Exception:
                            pass

                # Option J: update relative map (ultrasonic + vision) and persist periodically.
                try:
                    self._update_relative_map(sensor_readings_raw, min_front, camera_analysis if isinstance(camera_analysis, dict) else None)
                except Exception:
                    pass
                try:
                    self._maybe_persist()
                except Exception:
                    pass
                
                # Periodic map stats logging (every 50 cycles) for monitoring.
                try:
                    if decision_count % 50 == 0 and bool(getattr(self, '_persist_enabled', False)):
                        with self._pose_lock:
                            v_count = len(self._visited)
                            b_count = len(self._blocked)
                            l_count = len(self._persist_landmarks)
                            # Periodic trimming of _occ_score to prevent unbounded growth
                            max_cells = int(getattr(self, '_persist_max_cells', 12000) or 12000)
                            if len(self._occ_score) > max_cells:
                                self._occ_score = self._persist__limit_occ_score(self._occ_score, max_cells)
                        dprint(True, f"[AUTO-MAP] Cycle {decision_count}: visited={v_count}, blocked={b_count}, landmarks={l_count}")
                except Exception:
                    pass
                
                # 2. Build enhanced prompt with vision + sensors + exploration context
                user_prompt = sensor_text
                if camera_analysis:
                    user_prompt += f"Camera available with live image for analysis. "
                
                # Add recent exploration context to guide decisions with RICH MEMORY DATA
                if exploration_memory:
                    # Include observations/reasoning
                    recent_observations = " | ".join([m['observation'][:80] for m in exploration_memory[-3:]])
                    user_prompt += f"Recent observations: {recent_observations}. "

                # Add long-term map memory (persistent) to guide decisions away from known bad cells.
                # Keep this very short to avoid prompt bloat (only include when relevant).
                try:
                    with self._pose_lock:
                        neigh = self._neighbor_cells_from_pose_locked()
                        blocked_dirs = []
                        if neigh.get('F') in self._blocked:
                            blocked_dirs.append('F')
                        if neigh.get('L') in self._blocked:
                            blocked_dirs.append('L')
                        if neigh.get('R') in self._blocked:
                            blocked_dirs.append('R')
                        if blocked_dirs:
                            user_prompt += f"Map: next-blocked={','.join(blocked_dirs)}. "
                        # Only mention map size if it's grown significantly (avoid prompt bloat).
                        if len(self._visited) > 30 or len(self._blocked) > 10:
                            user_prompt += f"Map: {len(self._visited)}v,{len(self._blocked)}b. "
                except Exception:
                    pass
                    
                    # Include vision history for pattern recognition
                    try:
                        recent_with_vision = [m for m in exploration_memory[-3:] if m.get('vision', {}).get('available', False)]
                        if recent_with_vision:
                            vision_summary = []
                            for mem in recent_with_vision:
                                v = mem.get('vision', {})
                                dist = v.get('distance_estimate', 'unknown')
                                obst = 'obstacles' if v.get('obstacles', False) else 'clear'
                                vision_summary.append(f"{dist}/{obst}")
                            user_prompt += f"Vision history: {' → '.join(vision_summary)}. "
                    except Exception:
                        pass
                    user_prompt += f"Recent observations: {recent_observations}. "
                
                # Add exploration guidance - ENCOURAGE forward movement
                exploration_hint = ""
                if decision_count > 1 and exploration_memory:
                    # Check recent movement patterns
                    recent_commands = [m.get('command', 'UNKNOWN') for m in exploration_memory[-5:]]
                    forward_count = recent_commands.count('FORWARD')
                    turn_count = recent_commands.count('LEFT') + recent_commands.count('RIGHT')
                    # Only warn if we're turning too much - we WANT forward movement
                    if turn_count > forward_count and turn_count >= 3:
                        exploration_hint = "TIP: Path seems clear - move FORWARD more to make progress! "
                
                # Add navigation guidance - prefer forward
                navigation_hint = ""
                if consecutive_forwards >= 6:
                    # Only suggest turns after many consecutive forwards
                    navigation_hint = "Consider a slight turn to explore different areas. "
                elif consecutive_forwards == 0 and decision_count > 2:
                    # Encourage forward when we haven't moved forward recently
                    navigation_hint = "IMPORTANT: Move FORWARD when path is clear to explore efficiently. "
                
                user_prompt += f"{exploration_hint}{navigation_hint}PRIORITY: Move FORWARD when path is clear. Use camera and ultrasonic sensors to detect obstacles. Only turn when obstacles block forward path. Goal: Explore by moving forward as much as possible, turning only to avoid obstacles and continue forward. Remember: large objects in frame = very close, turn and continue forward!"
                
                # Merge ultrasonic sensor readings into camera_analysis so Llama3.1 sees them.
                try:
                    if camera_analysis is None or not isinstance(camera_analysis, dict):
                        # Initialize with safe defaults
                        camera_analysis = {
                            "obstacles": False,
                            "clear_forward": True,
                            "distance_estimate": "medium",
                            "distance_cm": min_front if min_front is not None else 50.0,
                            "confidence": 0.5,
                            "recommendation": "PROCEED"
                        }
                    if sensor_readings and len(sensor_readings) >= 3:
                        # FIXED: Only inject valid positive readings
                        camera_analysis['ultrasonic_left_cm'] = sensor_readings[0] if (sensor_readings[0] is not None and sensor_readings[0] > 0) else None
                        camera_analysis['ultrasonic_center_cm'] = sensor_readings[1] if (sensor_readings[1] is not None and sensor_readings[1] > 0) else None
                        camera_analysis['ultrasonic_right_cm'] = sensor_readings[2] if (sensor_readings[2] is not None and sensor_readings[2] > 0) else None
                        try:
                            front_vals = []
                            if camera_analysis.get('ultrasonic_left_cm') is not None and float(camera_analysis.get('ultrasonic_left_cm')) > 0:
                                front_vals.append(float(camera_analysis.get('ultrasonic_left_cm')))
                            if camera_analysis.get('ultrasonic_right_cm') is not None and float(camera_analysis.get('ultrasonic_right_cm')) > 0:
                                front_vals.append(float(camera_analysis.get('ultrasonic_right_cm')))
                            camera_analysis['ultrasonic_front_min_cm'] = min(front_vals) if front_vals else None
                        except Exception:
                            camera_analysis['ultrasonic_front_min_cm'] = None
                except Exception:
                    pass
                
                # 4. Get decision from Llama3.1 using the structured Llava analysis (no image here on purpose).
                # This ensures we truly use BOTH models: Llava for perception, Llama3.1 for planning.
                # OPTIMIZATION: If vision was skipped AND sensors show clear path, use fast reactive decision
                # CRITICAL: Only use fast path when we have valid front sensor data (min_front must exist)
                if skip_vision and min_front is not None and float(min_front) > 25.0:
                    # Fast sensor-based decision: strongly prefer forward movement
                    # CRITICAL: Ensure robot can FIT forward - check both front sensors, not just minimum
                    left_clear = (left_cm is None) or (float(left_cm) >= float(self._min_proceed_cm))
                    right_clear = (right_cm is None) or (float(right_cm) >= float(self._min_proceed_cm))
                    robot_can_fit_forward = left_clear and right_clear and min_front >= float(self._min_proceed_cm)
                    
                    # CAMERA SAFETY CHECK: Also verify camera shows clear path (if available)
                    camera_allows_forward = True
                    if camera_analysis and isinstance(camera_analysis, dict):
                        # Check camera analysis for obstacles
                        vision_obstacles = camera_analysis.get('obstacles', False)
                        vision_clear_forward = camera_analysis.get('clear_forward', True)
                        vision_distance_cm = camera_analysis.get('distance_cm', 100)
                        
                        # Block forward if camera detects obstacles
                        if vision_obstacles or not vision_clear_forward:
                            camera_allows_forward = False
                            print(f"[CAMERA-SAFETY] Camera detected obstacle - blocking FORWARD (obstacles={vision_obstacles}, clear_forward={vision_clear_forward})")
                        elif vision_distance_cm is not None and vision_distance_cm < 30:
                            camera_allows_forward = False
                            print(f"[CAMERA-SAFETY] Camera shows close obstacle ({vision_distance_cm:.0f}cm) - blocking FORWARD")
                    
                    if robot_can_fit_forward and camera_allows_forward:
                        # FORWARD PREFERENCE: Always move forward when path is clear
                        # Use longer durations for smoother, more confident movement
                        duration = random.uniform(2.5, 4.0)  # Longer forward bursts
                        cmd = {"command": "FORWARD", "duration": duration, "speed": 100, "speak": ""}
                        consecutive_forwards += 1
                        lawnmower_depth += 1
                        dprint(SARAH_DEBUG, f"[AUTO-FAST] Clear path detected: FORWARD {duration:.1f}s (sensors: L={left_cm}, R={right_cm}, min={min_front:.1f}cm)")
                    elif sensor_readings_raw and len(sensor_readings_raw) >= 3:
                        # IMPROVED OBSTACLE HANDLING: Turn toward clearer side and continue forward
                        # No backing up - just turn and keep moving forward
                        left_cm = None
                        right_cm = None
                        try:
                            left_cm = float(sensor_readings_raw[0]) if (sensor_readings_raw[0] is not None and float(sensor_readings_raw[0]) > 0) else None
                            right_cm = float(sensor_readings_raw[2]) if (sensor_readings_raw[2] is not None and float(sensor_readings_raw[2]) > 0) else None
                        except Exception:
                            left_cm = right_cm = None

                        # Determine which direction has more clearance
                        if left_cm is None and right_cm is None:
                            # No side sensor data - pick random turn
                            turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                            turn_duration = 0.9
                        elif right_cm is None:
                            # Only left sensor working - turn left if it's clearer
                            turn_dir = 'LEFT'
                            turn_duration = 1.2 if left_cm > 30 else 0.9
                        elif left_cm is None:
                            # Only right sensor working - turn right if it's clearer
                            turn_dir = 'RIGHT'
                            turn_duration = 1.2 if right_cm > 30 else 0.9
                        else:
                            # Both sensors working - turn toward the clearer side
                            turn_dir = 'LEFT' if float(left_cm) >= float(right_cm) else 'RIGHT'
                            # Longer turn if the clear side has lots of space
                            max_clear = max(float(left_cm), float(right_cm))
                            turn_duration = 1.2 if max_clear > 30 else 0.9

                        # Turn in place, then immediately continue forward
                        cmd = {
                            "command": turn_dir,
                            "duration": turn_duration,
                            "speed": 100,
                            "speak": "",
                            "_followup": {"command": "FORWARD", "duration": 2.0, "speed": 100}
                        }
                        dprint(SARAH_DEBUG, f"[AUTO-FAST] Obstacle detected - TURN {turn_dir} then FORWARD (L={left_cm}, R={right_cm})")
                    else:
                        # No sensor data - default cautious turn
                        cmd = {"command": ("LEFT" if random.random() < 0.5 else "RIGHT"), "duration": 0.8, "speed": 100, "speak": ""}
                else:
                    # Use AI for decision (either vision was used, or sensors show obstacle)
                    decision_timeout = 10 if (min_front is not None and min_front > 40.0) else AUTONOMOUS_DECISION_TIMEOUT
                    try:
                        hard_budget = float(os.getenv('SARAH_AUTO_AI_BUDGET_S', '2.0').strip() or '2.0')
                    except Exception:
                        hard_budget = 2.0
                    decision_timeout = max(1.0, min(float(decision_timeout), float(hard_budget)))

                    # If we're out of time budget for this cycle, don't block on Llama.
                    try:
                        time_left = float(cycle_budget_s) - float(time.time() - float(cycle_start_time))
                    except Exception:
                        time_left = 0.0
                    if time_left < 0.25:
                        dprint(SARAH_DEBUG, f"[AUTO-BUDGET] Skipping Llama decision (time_left={time_left:.2f}s)")
                        # Deterministic fallback: if front looks blocked by vision, turn; otherwise forward a short burst.
                        turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                        try:
                            if camera_analysis and isinstance(camera_analysis, dict):
                                if bool(camera_analysis.get('obstacles', False)) or (not bool(camera_analysis.get('clear_forward', True))):
                                    cmd = {"command": turn_dir, "duration": 0.9, "speed": 100, "speak": ""}
                                else:
                                    cmd = {"command": "FORWARD", "duration": 1.8, "speed": 100, "speak": ""}
                            else:
                                cmd = {"command": turn_dir, "duration": 0.9, "speed": 100, "speak": ""}
                        except Exception:
                            cmd = {"command": turn_dir, "duration": 0.9, "speed": 100, "speak": ""}
                    else:
                        cmd = query_ollama_for_command(self.system_prompt, user_prompt, image_data=None, camera_analysis=camera_analysis, timeout=decision_timeout)
                
                # CRITICAL SAFETY: If we have NO valid sensor data, only allow turns and backward, NEVER forward
                safety_override_active = False
                if min_front is None and cmd.get('command') == 'FORWARD':
                    print("[AUTO-SAFETY] No sensor data - blocking FORWARD, turning instead")
                    turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                    cmd = {"command": turn_dir, "duration": 0.8, "speed": 100, "speak": "Sensors offline, turning cautiously."}
                    safety_override_active = True

                # If the model provided any optional fields (distance/clear/etc), merge them into camera_analysis
                # so downstream safety + duration logic can use them deterministically.
                try:
                    if cmd and isinstance(cmd, dict):
                        if camera_analysis is None or not isinstance(camera_analysis, dict):
                            camera_analysis = {}
                        for k in (
                            'obstacles',
                            'recommendation',
                            'distance_estimate',
                            'distance_cm',
                            'clear_left',
                            'clear_right',
                            'clear_forward',
                            'obstacle_size',
                        ):
                            if k in cmd and cmd.get(k) is not None:
                                camera_analysis[k] = cmd.get(k)
                except Exception:
                    pass
                
                # Store the command AND comprehensive vision/sensor data in memory for exploration tracking
                if cmd:
                    # Build comprehensive memory entry with all available data
                    memory_entry = {
                        'cycle': decision_count,
                        'command': cmd.get('command', 'UNKNOWN'),
                        'observation': cmd.get('reasoning', 'No reasoning provided'),
                        'timestamp': time.time(),
                        'duration': cmd.get('duration', 0),
                        'speed': cmd.get('speed', 0)
                    }
                    
                    # Add comprehensive vision analysis from Llava if available
                    if camera_analysis and isinstance(camera_analysis, dict):
                        memory_entry['vision'] = {
                            'available': True,
                            'distance_estimate': camera_analysis.get('distance_estimate', 'unknown'),
                            'distance_cm': camera_analysis.get('distance_cm', None),
                            'obstacles': camera_analysis.get('obstacles', False),
                            'clear_forward': camera_analysis.get('clear_forward', None),
                            'clear_left': camera_analysis.get('clear_left', None),
                            'clear_right': camera_analysis.get('clear_right', None),
                            'confidence': camera_analysis.get('confidence', 0.0),
                            'recommendation': camera_analysis.get('recommendation', 'UNKNOWN'),
                            'description': camera_analysis.get('description', '')[:100],  # Truncate for memory
                            'obstacle_size': camera_analysis.get('obstacle_size', None)
                        }
                    else:
                        memory_entry['vision'] = {'available': False}
                    
                    # Add sensor readings
                    if sensor_readings:
                        memory_entry['sensors'] = {
                            'left_cm': sensor_readings[0] if len(sensor_readings) > 0 else None,
                            'center_cm': sensor_readings[1] if len(sensor_readings) > 1 else None,
                            'right_cm': sensor_readings[2] if len(sensor_readings) > 2 else None,
                            'min_front_cm': min_front
                        }
                    
                    # Add robot pose (dead reckoning position)
                    try:
                        with self._pose_lock:
                            memory_entry['pose'] = {
                                'x_cm': float(self._x_cm),
                                'y_cm': float(self._y_cm),
                                'heading_deg': float(self._heading_deg),
                                'grid_cell': self._grid_cell_from_xy(self._x_cm, self._y_cm)
                            }
                    except Exception:
                        pass
                    
                    exploration_memory.append(memory_entry)
                    if len(exploration_memory) > memory_max_size:
                        exploration_memory.pop(0)
                    
                    # LAWNMOWER PATTERN: Track depth and switch phases for systematic coverage
                    cmd_type = cmd.get('command', 'UNKNOWN')
                    if cmd_type == 'FORWARD':
                        consecutive_forwards += 1
                        lawnmower_depth += 1
                    elif cmd_type in ('LEFT', 'RIGHT'):
                        consecutive_forwards = 0
                        last_turn_direction = cmd_type
                        # After going deep, retreat back; after retreating, explore again
                        if lawnmower_depth >= 4 and lawnmower_phase == "explore":
                            lawnmower_phase = "retreat"
                            dprint(SARAH_DEBUG, f"[LAWNMOWER] Switching to RETREAT phase (depth {lawnmower_depth})")
                        elif lawnmower_depth <= -2 and lawnmower_phase == "retreat":
                            lawnmower_phase = "explore"
                            lawnmower_depth = 0
                            dprint(SARAH_DEBUG, f"[LAWNMOWER] Switching to EXPLORE phase")
                    elif cmd_type == 'BACKWARD':
                        consecutive_forwards = 0
                        lawnmower_depth -= 1

                # If an emergency stop was triggered (or mode changed) while we were waiting
                # on the model, do not execute any movement.
                if self.stop_event.is_set():
                    try:
                        if self.drive:
                            self.drive.stop()
                    except Exception:
                        pass
                    break
                if get_current_mode() != "autonomous":
                    try:
                        if self.drive:
                            self.drive.stop()
                    except Exception:
                        pass
                    continue
                
                # SMART VISION-TO-MOVEMENT MAPPING with distance estimation
                # ENHANCED SAFETY: Camera vision takes priority for detecting walls and obstacles
                # Vision can detect obstacles that ultrasonic sensors miss (thin objects, gaps between sensors)
                if latest_img_b64 and camera_analysis and not safety_override_active:
                    vision_rec = camera_analysis.get('recommendation', 'PROCEED')
                    vision_obstacles = camera_analysis.get('obstacles', False)
                    distance_est = camera_analysis.get('distance_estimate', 'medium')
                    distance_cm = camera_analysis.get('distance_cm', 50)
                    clear_left = camera_analysis.get('clear_left', True)
                    clear_right = camera_analysis.get('clear_right', True)
                    clear_forward = camera_analysis.get('clear_forward', True)
                    
                    # Format distance_cm safely for logging
                    dist_str = f"{distance_cm:.0f}" if distance_cm is not None else "unknown"
                    print(f"[VISION] Distance: {distance_est} (~{dist_str}cm), Obstacles: {vision_obstacles}, Clear L/R/F: {clear_left}/{clear_right}/{clear_forward}")
                    
                    # ENHANCED SENSOR FUSION: Camera takes priority for forward obstacle detection
                    # Ultrasonic sensors can miss walls if angled or between sensors
                    vision_confidence = camera_analysis.get('confidence', 0.5)
                    
                    # If camera detects obstacle ahead, TRUST IT even if ultrasonics seem clear
                    # This prevents running into walls that sensors miss
                    camera_detects_obstacle = vision_obstacles or (not clear_forward) or \
                                            (distance_cm is not None and distance_cm < 35)
                    
                    # OVERRIDE FORWARD COMMANDS if camera sees obstacle
                    if cmd and cmd.get('command') == 'FORWARD' and camera_detects_obstacle:
                        print(f"[CAMERA-OVERRIDE] Camera detected obstacle ahead - changing FORWARD to TURN")
                        print(f"[CAMERA-OVERRIDE] Details: obstacles={vision_obstacles}, clear_fwd={clear_forward}, dist={distance_cm}cm")
                        
                        # Choose turn direction based on camera clearance data
                        if clear_left and not clear_right:
                            turn_dir = 'LEFT'
                        elif clear_right and not clear_left:
                            turn_dir = 'RIGHT'
                        else:
                            # Both sides similar - pick based on distance or random
                            turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                        
                        cmd = {
                            "command": turn_dir,
                            "duration": 1.0,
                            "speed": 100,
                            "speak": f"Camera detected wall ahead, turning {turn_dir.lower()}.",
                            "emotion": "concerned"
                        }
                    
                    # SENSOR FUSION VALIDATION for existing obstacle detections
                    ultrasonic_confirms_obstacle = False
                    if min_front is not None and min_front <= 25.0:  # Front ultrasonics detect something close
                        ultrasonic_confirms_obstacle = True
                    
                    # High confidence camera + close distance = override even if sensors disagree
                    high_confidence_override = (vision_confidence >= 0.75 and distance_est in ['very_close', 'close'] and 
                                              distance_cm is not None and distance_cm < 25)
                    vision_override_allowed = camera_detects_obstacle and (ultrasonic_confirms_obstacle or (min_front is None) or high_confidence_override)
                    
                    if not vision_override_allowed and camera_detects_obstacle and min_front and min_front > 30:
                        print(f"[SENSOR-FUSION] Camera reports obstacle but front ultrasonics clear (min_front={min_front}cm)")
                        print(f"[SENSOR-FUSION] Camera confidence={vision_confidence:.2f}, distance={distance_cm}cm - trusting camera for safety")
                        vision_override_allowed = True  # Trust camera to prevent wall collisions
                    
                    if vision_override_allowed:
                        # CRITICAL: Very close obstacles - must turn immediately
                        if distance_est == 'very_close' or (distance_cm is not None and distance_cm < 22):
                            turn_dir = 'LEFT' if (clear_left or random.random() < 0.5) else 'RIGHT'
                            print(f"[VISION-SAFETY] VERY CLOSE obstacle detected - turning {turn_dir}")
                            cmd = {"command": turn_dir, "duration": 1.0, "speed": 100, "speak": f"Too close! Turning {turn_dir.lower()}.", "emotion": "concerned"}
                        
                        # Close obstacles - prefer turning
                        elif distance_est == 'close' or (distance_cm is not None and distance_cm < 35):
                            if clear_left and not clear_right:
                                cmd = {"command": "LEFT", "duration": 1.0, "speed": 100, "speak": "Turning left to avoid obstacle.", "emotion": "thinking"}
                            elif clear_right and not clear_left:
                                cmd = {"command": "RIGHT", "duration": 1.0, "speed": 100, "speak": "Turning right to avoid obstacle.", "emotion": "thinking"}
                            elif not clear_forward:
                                turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                                cmd = {"command": turn_dir, "duration": 1.0, "speed": 100, "speak": f"Obstacle close, turning {turn_dir.lower()}.", "emotion": "concerned"}

                        # Recommend turns from vision
                        elif vision_rec == 'TURN_LEFT':
                            cmd = {"command": "LEFT", "duration": 1.0, "speed": 100, "speak": "Vision suggests turning left.", "emotion": "thinking"}
                        elif vision_rec == 'TURN_RIGHT':
                            cmd = {"command": "RIGHT", "duration": 1.0, "speed": 100, "speak": "Vision suggests turning right.", "emotion": "thinking"}

                        # Stop command from vision
                        elif vision_rec == 'STOP' or (vision_obstacles and distance_cm is not None and distance_cm < 22):
                            print(f"[VISION-SAFETY] Obstacle detected - stopping")
                            cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Obstacle ahead - stopping.", "emotion": "concerned"}

                # FRONTIER EXPLORATION (Option C): deterministic target selection toward unvisited adjacent cells.
                # This runs after vision safety overrides, and before the old random post-policy.
                # IMPORTANT: Skip if safety_override_active flag is set
                try:
                    if self._frontier_enabled and (not RETURN_TO_START_EVENT.is_set()) and (not self._explore_exhausted) and not safety_override_active:
                        planned = self._frontier_plan_command(sensor_readings, camera_analysis)
                        if planned is not None:
                            try:
                                planned["_planner"] = "frontier"
                            except Exception:
                                pass
                            cmd = planned
                except Exception:
                    pass

                # EXPLORATION POST-POLICY: reduce forward bias when it is safe to turn.
                # This is a deterministic safety layer on top of the model output.
                # IMPORTANT: Skip if safety_override_active flag is set
                try:
                    if safety_override_active or str(cmd.get("_planner", "")) == "frontier":
                        raise RuntimeError("skip post-policy")

                    def _safe_float(v):
                        try:
                            x = float(v)
                            return x if x > 0 else None
                        except Exception:
                            return None

                    left_cm = _safe_float(sensor_readings[0]) if sensor_readings and len(sensor_readings) > 0 else None
                    center_cm = _safe_float(sensor_readings[1]) if sensor_readings and len(sensor_readings) > 1 else None
                    right_cm = _safe_float(sensor_readings[2]) if sensor_readings and len(sensor_readings) > 2 else None

                    front_vals = [v for v in (left_cm, right_cm) if v is not None]
                    front_min = min(front_vals) if front_vals else None

                    # Prefer model-provided clear_* if present; otherwise infer from ultrasonic.
                    clear_left_pol = bool(cmd.get('clear_left', (left_cm is None or left_cm >= self._clear_side_cm)))
                    clear_right_pol = bool(cmd.get('clear_right', (right_cm is None or right_cm >= self._clear_side_cm)))
                    clear_forward_pol = bool(cmd.get('clear_forward', (front_min is not None and front_min >= self._clear_forward_cm)))

                    recent_cmds = [m.get('command', '') for m in exploration_memory[-5:]] if exploration_memory else []
                    forward_streak = 0
                    for c in reversed(recent_cmds):
                        if str(c).upper() == 'FORWARD':
                            forward_streak += 1
                        else:
                            break

                    chosen = None
                    current_cmd = str(cmd.get('command', 'STOP')).upper()

                    # Only override when it's safe to turn.
                    can_turn = clear_left_pol or clear_right_pol
                    if current_cmd == 'FORWARD' and can_turn and clear_forward_pol:
                        force_turn = forward_streak >= self._max_forward_streak
                        maybe_turn = (random.random() < self._turn_when_clear_prob)

                        # If we've likely already visited the next forward grid cell, strongly prefer turning.
                        try:
                            with self._pose_lock:
                                x0, y0, h0 = float(self._x_cm), float(self._y_cm), float(self._heading_deg)
                                g = float(self._grid_cm) if float(self._grid_cm) > 0 else 50.0
                                f_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0)) * g, y0 + math.sin(math.radians(h0)) * g)
                                l_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0 + 90.0)) * g, y0 + math.sin(math.radians(h0 + 90.0)) * g)
                                r_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0 - 90.0)) * g, y0 + math.sin(math.radians(h0 - 90.0)) * g)
                                if f_cell in self._visited and (l_cell not in self._visited or r_cell not in self._visited):
                                    force_turn = True
                        except Exception:
                            pass
                        if force_turn or maybe_turn:
                            left_w = self._explore_left_w if clear_left_pol else 0.0
                            right_w = self._explore_right_w if clear_right_pol else 0.0

                            # If we can infer an unvisited side, bias toward it.
                            try:
                                with self._pose_lock:
                                    x0, y0, h0 = float(self._x_cm), float(self._y_cm), float(self._heading_deg)
                                    g = float(self._grid_cm) if float(self._grid_cm) > 0 else 50.0
                                    l_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0 + 90.0)) * g, y0 + math.sin(math.radians(h0 + 90.0)) * g)
                                    r_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0 - 90.0)) * g, y0 + math.sin(math.radians(h0 - 90.0)) * g)
                                    if l_cell not in self._visited and r_cell in self._visited:
                                        left_w *= 1.4
                                    elif r_cell not in self._visited and l_cell in self._visited:
                                        right_w *= 1.4
                            except Exception:
                                pass

                            # Prefer the more open side if we have ultrasonic data.
                            if left_cm is not None and right_cm is not None and (left_w > 0 or right_w > 0):
                                if left_cm > right_cm:
                                    left_w *= 1.2
                                elif right_cm > left_cm:
                                    right_w *= 1.2

                            total = left_w + right_w
                            if total > 0:
                                r = random.random() * total
                                chosen = 'LEFT' if r < left_w else 'RIGHT'

                    # If model said STOP but we appear clear, allow exploration move occasionally.
                    if chosen is None and current_cmd == 'STOP' and clear_forward_pol and can_turn:
                        # Never override a vision STOP into movement.
                        # Only treat it as a vision stop if BOTH obstacles are detected AND recommendation is STOP.
                        # This prevents false stops when camera is unavailable (which returns PROCEED).
                        vision_stop = False
                        try:
                            if camera_analysis and isinstance(camera_analysis, dict):
                                has_obstacles = bool(camera_analysis.get('obstacles', False))
                                rec_stop = str(camera_analysis.get('recommendation', '')).upper() == 'STOP'
                                if has_obstacles and rec_stop:
                                    vision_stop = True
                        except Exception:
                            # On exception, err on the side of caution but don't freeze exploration.
                            # Ultrasonic sensors + execution safety still provide hard stops.
                            vision_stop = False

                        if not vision_stop:
                            f_w = self._explore_forward_w if clear_forward_pol else 0.0
                            l_w = self._explore_left_w if clear_left_pol else 0.0
                            r_w = self._explore_right_w if clear_right_pol else 0.0
                            total = f_w + l_w + r_w
                            if total > 0:
                                rr = random.random() * total
                                chosen = 'FORWARD' if rr < f_w else ('LEFT' if rr < (f_w + l_w) else 'RIGHT')

                    if chosen in ('LEFT', 'RIGHT', 'FORWARD') and chosen != current_cmd:
                        dprint(
                            SARAH_DEBUG,
                            f"[AUTO-POLICY] Override {current_cmd} -> {chosen} (streak={forward_streak}, clear L/R/F={clear_left_pol}/{clear_right_pol}/{clear_forward_pol})",
                        )
                        cmd['command'] = chosen
                        cmd['duration'] = 1.0
                        cmd['speed'] = 100
                        if not cmd.get('speak'):
                            cmd['speak'] = f"Exploring: turning {chosen.lower()}." if chosen in ('LEFT', 'RIGHT') else "Exploring ahead."
                except Exception:
                    pass

                # Clamp turn durations to short bursts.
                try:
                    if str(cmd.get('command', '')).upper() in ("LEFT", "RIGHT"):
                        cmd['duration'] = min(float(cmd.get('duration', 0) or 0), float(self._turn_duration))
                except Exception:
                    pass

                # Exploration exhaustion: if all nearby clear directions are already visited, stop (once) until user asks.
                try:
                    if not RETURN_TO_START_EVENT.is_set():
                        def _safe_float(v):
                            try:
                                x = float(v)
                                return x if x > 0 else None
                            except Exception:
                                return None

                        left_cm = _safe_float(sensor_readings[0]) if sensor_readings and len(sensor_readings) > 0 else None
                        center_cm = _safe_float(sensor_readings[1]) if sensor_readings and len(sensor_readings) > 1 else None
                        right_cm = _safe_float(sensor_readings[2]) if sensor_readings and len(sensor_readings) > 2 else None

                        # BUGFIX: Use the authoritative min_front from earlier sensor processing
                        # This was already calculated from the filtered readings (lines ~6940)
                        # Recalculating here from left_cm/right_cm can give wrong results
                        front_min_exh = min_front  # Use the value from earlier in the cycle
                        
                        # Safe clearance checks with None handling
                        try:
                            clear_left_pol = bool(cmd.get('clear_left', (left_cm is None or float(left_cm) >= self._clear_side_cm) if left_cm is not None else True))
                        except Exception:
                            clear_left_pol = True
                        
                        try:
                            clear_right_pol = bool(cmd.get('clear_right', (right_cm is None or float(right_cm) >= self._clear_side_cm) if right_cm is not None else True))
                        except Exception:
                            clear_right_pol = True
                        
                        # CRITICAL: Forward is only clear if robot can FIT (both sensors show clearance)
                        # The min_front value is minimum of left/right, but we need BOTH to be clear
                        try:
                            left_ok = (left_cm is None) or (float(left_cm) >= self._clear_forward_cm)
                            right_ok = (right_cm is None) or (float(right_cm) >= self._clear_forward_cm)
                            both_clear = left_ok and right_ok and (front_min_exh is not None and float(front_min_exh) >= self._clear_forward_cm)
                            clear_forward_pol = bool(cmd.get('clear_forward', both_clear))
                        except Exception:
                            clear_forward_pol = False

                        # Access visited cells with proper locking (prevents race conditions)
                        with self._pose_lock:
                            x0, y0, h0 = float(self._x_cm), float(self._y_cm), float(self._heading_deg)
                            g = float(self._grid_cm) if float(self._grid_cm) > 0 else 50.0
                            f_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0)) * g, y0 + math.sin(math.radians(h0)) * g)
                            l_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0 + 90.0)) * g, y0 + math.sin(math.radians(h0 + 90.0)) * g)
                            r_cell = self._grid_cell_from_xy(x0 + math.cos(math.radians(h0 - 90.0)) * g, y0 + math.sin(math.radians(h0 - 90.0)) * g)

                            # Check visited status while holding lock
                            forward_new = (f_cell not in self._visited) and clear_forward_pol
                            left_new = (l_cell not in self._visited) and clear_left_pol
                            right_new = (r_cell not in self._visited) and clear_right_pol

                            # If frontier exploration is enabled and there are still frontier targets,
                            # do NOT mark exploration as exhausted just because adjacent cells are visited.
                            has_frontier = False
                            try:
                                if self._frontier_enabled:
                                    has_frontier = self._choose_frontier_target_locked() is not None
                            except Exception:
                                has_frontier = False

                        if has_frontier:
                            self._exhausted_cycles = 0
                            self._explore_exhausted = False
                        elif not (forward_new or left_new or right_new):
                            self._exhausted_cycles += 1
                        else:
                            self._exhausted_cycles = 0
                            self._explore_exhausted = False

                        # Require a few consecutive confirmations to avoid stopping due to transient sensor noise.
                        if self._exhausted_cycles >= 3 and not self._explore_exhausted:
                            self._explore_exhausted = True
                            cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "I think I've explored everything nearby. Say 'return home' or 'explore' to continue.", "emotion": "thinking"}
                except Exception:
                    pass
                
                # Display decision with timing and detailed information
                elapsed_so_far = time.time() - cycle_start_time
                command = cmd.get('command', 'STOP').upper()
                duration = float(cmd.get('duration', 0) or 0)
                speed = int(cmd.get('speed', DEFAULT_SPEED))
                
                # SMOOTH VARIATION: Add subtle weaving to forward commands for natural movement
                if command == 'FORWARD' and duration > 1.5:
                    if 'weave' not in cmd:
                        # 30% chance of adding subtle drift for variation
                        if random.random() < 0.3:
                            cmd['weave'] = random.choice([-8, -5, 5, 8])
                            dprint(SARAH_DEBUG, f"[VARIATION] Adding subtle weave {cmd['weave']} to forward movement")

                # Per-command caps (forward can be long; turns/backward should remain short).
                if command in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
                    try:
                        cap = float(self._max_forward_sec) if command == 'FORWARD' else (float(self._max_backward_sec) if command == 'BACKWARD' else float(self._max_turn_sec))
                        cap = max(0.0, min(float(AUTONOMOUS_MAX_MOVE_DURATION), float(cap)))
                        duration = max(0.0, min(float(duration), float(cap)))
                        cmd["duration"] = duration
                        
                        # SAFETY: Reduce forward speed if obstacles are moderately close (20-40cm)
                        # (Front-only; the center sensor is mounted on the back.)
                        if command == 'FORWARD' and min_front is not None and 20.0 < float(min_front) <= 40.0:
                            original_speed = speed
                            speed = 60  # Reduce to 60% when getting close
                            cmd["speed"] = speed
                            dprint(SARAH_DEBUG, f"[AUTO-SAFETY] Reducing forward speed {original_speed}% -> {speed}% (front obstacle at {float(min_front):.1f}cm)")
                    except Exception:
                        duration = min(duration, AUTONOMOUS_MAX_MOVE_DURATION)
                        cmd["duration"] = duration

                # Adaptive duration: apply a final forward-duration safety clamp.
                # (Frontier planner already uses this, but keep it for any other FORWARD sources.)
                try:
                    if command == "FORWARD":
                        desired_cm = self._desired_forward_cm(sensor_readings, camera_analysis)
                        safe_dur = self._safe_forward_duration_for_cm(desired_cm, sensor_readings, camera_analysis, speed_pct=float(cmd.get('speed', 100) or 100))
                        if safe_dur <= 0:
                            cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Too close - stopping.", "emotion": "concerned"}
                            command = "STOP"
                            duration = 0.0
                        else:
                            cmd["duration"] = float(safe_dur)
                            duration = float(safe_dur)
                except Exception:
                    pass

                # Debounce/hysteresis gate: if ultrasonic was recently too close, don't immediately resume FORWARD.
                try:
                    if command == "FORWARD" and bool(self._ultra_forward_blocked):
                        cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Waiting for clearance.", "emotion": "concerned"}
                        command = "STOP"
                        duration = 0.0
                except Exception:
                    pass

                # If we're trying to go forward but front clearance is below minimum, block it.
                # CRITICAL: Also check if robot can actually FIT (both sensors must be clear)
                try:
                    if command == "FORWARD" and min_front is not None:
                        # Check if robot can fit forward (both sensors clear enough)
                        left_ok_fit = True
                        right_ok_fit = True
                        try:
                            if sensor_readings_raw and len(sensor_readings_raw) >= 3:
                                left_cm_fit = float(sensor_readings_raw[0]) if (sensor_readings_raw[0] is not None and float(sensor_readings_raw[0]) > 0) else None
                                right_cm_fit = float(sensor_readings_raw[2]) if (sensor_readings_raw[2] is not None and float(sensor_readings_raw[2]) > 0) else None
                                if left_cm_fit is not None and float(left_cm_fit) < float(self._min_proceed_cm):
                                    left_ok_fit = False
                                if right_cm_fit is not None and float(right_cm_fit) < float(self._min_proceed_cm):
                                    right_ok_fit = False
                        except Exception:
                            pass
                        
                        blocked = (float(min_front) < float(self._min_proceed_cm)) or (not left_ok_fit) or (not right_ok_fit)
                        if blocked:
                            reason = "front too close" if float(min_front) < float(self._min_proceed_cm) else "robot won't fit (side sensor blocked)"
                            print(f"[AUTO] Blocking FORWARD ({reason}: min_front={float(min_front):.1f}cm, L/R fit={left_ok_fit}/{right_ok_fit})")
                        try:
                            roomba_style = str(os.getenv('SARAH_AUTO_ROOMBA_STYLE', '1')).strip().lower() not in ('0', 'false', 'no', 'off')
                        except Exception:
                            roomba_style = True

                        if roomba_style:
                            left_cm = None
                            right_cm = None
                            back_cm = None
                            try:
                                if sensor_readings_raw and len(sensor_readings_raw) >= 3:
                                    left_cm = float(sensor_readings_raw[0]) if (sensor_readings_raw[0] is not None and float(sensor_readings_raw[0]) > 0) else None
                                    right_cm = float(sensor_readings_raw[2]) if (sensor_readings_raw[2] is not None and float(sensor_readings_raw[2]) > 0) else None
                                if sensor_readings_raw and len(sensor_readings_raw) >= 2:
                                    back_cm = float(sensor_readings_raw[1]) if (sensor_readings_raw[1] is not None and float(sensor_readings_raw[1]) > 0) else None
                            except Exception:
                                left_cm = right_cm = back_cm = None

                            if left_cm is None and right_cm is None:
                                turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                            elif right_cm is None:
                                turn_dir = 'LEFT'
                            elif left_cm is None:
                                turn_dir = 'RIGHT'
                            else:
                                turn_dir = 'LEFT' if float(left_cm) >= float(right_cm) else 'RIGHT'

                            # Check if we have room to back up (allow with 5cm margin)
                            back_ok = (back_cm is None) or (float(back_cm) >= (float(self._min_proceed_cm) + 5.0))
                            if back_ok:
                                avoid_back = 1.1 if float(min_front) < (float(self._min_proceed_cm) * 0.75) else 0.9
                                try:
                                    avoid_back = float(os.getenv('SARAH_AUTO_AVOID_BACK_SEC', str(avoid_back)).strip() or str(avoid_back))
                                except Exception:
                                    pass
                                avoid_back = max(0.2, min(float(avoid_back), float(self._max_backward_sec)))

                                avoid_turn = 1.0
                                try:
                                    avoid_turn = float(os.getenv('SARAH_AUTO_AVOID_TURN_SEC', str(avoid_turn)).strip() or str(avoid_turn))
                                except Exception:
                                    pass
                                avoid_turn = max(0.2, min(float(avoid_turn), float(self._max_turn_sec)))

                                try:
                                    with self._pose_lock:
                                        self._mark_blocked_from_sensors_locked(min_front, left_cm, right_cm)
                                except Exception:
                                    pass

                                print(f"[AUTO-AVOID] BACKWARD {avoid_back:.1f}s then {turn_dir} {avoid_turn:.1f}s")
                                backup_cmd = {"command": "BACKWARD", "duration": avoid_back, "speed": 75, "speak": ""}
                                turn_cmd = {"command": turn_dir, "duration": avoid_turn, "speed": 100, "speak": ""}
                                execute_command(self.drive, backup_cmd, silent=True)
                                try:
                                    self._record_motion('BACKWARD', float(backup_cmd.get('duration', 0) or 0), int(backup_cmd.get('speed', 75) or 75))
                                except Exception:
                                    pass
                                execute_command(self.drive, turn_cmd, silent=True)
                                try:
                                    self._record_motion(turn_dir, float(turn_cmd.get('duration', 0) or 0), int(turn_cmd.get('speed', 100) or 100))
                                except Exception:
                                    pass

                                try:
                                    self._maybe_persist()
                                except Exception:
                                    pass
                                continue

                        # Fallback: can't reverse safely -> turn longer instead of STOP.
                        turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                        cmd = {"command": turn_dir, "duration": 1.0, "speed": 100, "speak": ""}
                        command = str(cmd.get('command', 'STOP')).upper()
                        duration = float(cmd.get('duration', 0) or 0)
                except Exception:
                    pass

                # One concise line per cycle by default
                print(f"[AUTO] #{decision_count} cmd={command} dur={duration:.1f}s speed={speed}% t={elapsed_so_far:.1f}s")

                # Full detail only when debugging
                if SARAH_DEBUG:
                    print("\n" + "="*80)
                    print(f"  [ROBOT] DECISION #{decision_count} | Movement: {command}")
                    print(f"  ├─ Vision→Action: {f'Clear path → FORWARD' if latest_img_b64 and camera_analysis and not camera_analysis.get('obstacles', False) else f'Obstacles → STOP' if latest_img_b64 and camera_analysis and camera_analysis.get('obstacles', False) else 'No vision data'}")
                    print(f"  ├─ Duration: {duration}s | Speed: {speed}%")
                    if camera_analysis:
                        vision_text = str(camera_analysis)[:60] + "..." if len(str(camera_analysis)) > 60 else str(camera_analysis)
                        print(f"  ├─ Vision: {vision_text}")
                    if sensor_readings:
                        print(f"  ├─ Sensors: L={sensor_readings[0]}cm C={sensor_readings[1]}cm R={sensor_readings[2]}cm")
                    if exploration_memory:
                        print(f"  ├─ Memory: {len(exploration_memory)} observations stored")
                        # Display rich memory details
                        for i, mem in enumerate(exploration_memory[-3:]):  # Last 3 entries
                            cycle_num = mem.get('cycle', '?')
                            cmd_type = mem.get('command', 'UNKNOWN')
                            print(f"  │  └─ [{cycle_num}] {cmd_type}: ", end='')
                            # Vision info
                            if mem.get('vision', {}).get('available', False):
                                v = mem['vision']
                                dist_est = v.get('distance_estimate', '?')
                                dist_cm = v.get('distance_cm', '?')
                                obst = 'OBS!' if v.get('obstacles', False) else 'clear'
                                print(f"Vision={dist_est}(~{dist_cm}cm,{obst}) ", end='')
                            # Sensor info
                            if 'sensors' in mem:
                                s = mem['sensors']
                                print(f"Sensors=L:{s.get('left_cm', '?')} R:{s.get('right_cm', '?')} Front:{s.get('min_front_cm', '?')}", end='')
                            print()  # Newline
                    print(f"  └─ Rationale: {cmd.get('speak', 'No explanation')}")
                    print("="*80 + "\n")
                
                # EXECUTE MOVEMENT
                if self.stop_event.is_set() or get_current_mode() != "autonomous":
                    try:
                        if self.drive:
                            self.drive.stop()
                    except Exception:
                        pass
                    continue
                
                # SAFETY: Final pre-execution check - verify sensors before forward movement
                if cmd.get('command') == 'FORWARD':
                    # Avoid an extra sensor read if we just sampled them very recently in this cycle.
                    final_check = []
                    try:
                        if (time.time() - float(sensor_readings_raw_ts)) <= 0.25:
                            final_check = list(sensor_readings_raw or [])
                        else:
                            final_check = read_all_sensors(self.sensors) if self.sensors else []
                    except Exception:
                        final_check = read_all_sensors(self.sensors) if self.sensors else []
                    try:
                        # Forward safety must use FRONT sensors only (left/right). Center sensor is mounted on the BACK.
                        front_vals = []
                        if final_check and len(final_check) >= 3:
                            if final_check[0] is not None and float(final_check[0]) > 0:
                                front_vals.append(float(final_check[0]))
                            if final_check[2] is not None and float(final_check[2]) > 0:
                                front_vals.append(float(final_check[2]))
                        final_min_front = min(front_vals) if front_vals else None
                        try:
                            pre_exec_min = float(os.getenv('SARAH_AUTO_PRE_EXEC_FORWARD_MIN_CM', '22.0').strip() or '22.0')
                        except Exception:
                            pre_exec_min = 22.0
                        pre_exec_min = max(float(self._min_proceed_cm), float(pre_exec_min))
                        
                        # CRITICAL: Check if robot can FIT forward (both sensors must show clearance)
                        left_fit = True
                        right_fit = True
                        if final_check and len(final_check) >= 3:
                            if final_check[0] is not None and float(final_check[0]) > 0 and float(final_check[0]) < pre_exec_min:
                                left_fit = False
                            if final_check[2] is not None and float(final_check[2]) > 0 and float(final_check[2]) < pre_exec_min:
                                right_fit = False
                        
                        obstacle_or_wont_fit = (final_min_front is not None and final_min_front <= float(pre_exec_min)) or (not left_fit) or (not right_fit)
                        if obstacle_or_wont_fit:
                            try:
                                roomba_style = str(os.getenv('SARAH_AUTO_ROOMBA_STYLE', '1')).strip().lower() not in ('0', 'false', 'no', 'off')
                            except Exception:
                                roomba_style = True

                            if roomba_style:
                                left_cm = None
                                right_cm = None
                                back_cm = None
                                try:
                                    if final_check and len(final_check) >= 3:
                                        left_cm = float(final_check[0]) if (final_check[0] is not None and float(final_check[0]) > 0) else None
                                        right_cm = float(final_check[2]) if (final_check[2] is not None and float(final_check[2]) > 0) else None
                                    if final_check and len(final_check) >= 2:
                                        back_cm = float(final_check[1]) if (final_check[1] is not None and float(final_check[1]) > 0) else None
                                except Exception:
                                    left_cm = right_cm = back_cm = None

                                if left_cm is None and right_cm is None:
                                    turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                                elif right_cm is None:
                                    turn_dir = 'LEFT'
                                elif left_cm is None:
                                    turn_dir = 'RIGHT'
                                else:
                                    turn_dir = 'LEFT' if float(left_cm) >= float(right_cm) else 'RIGHT'

                                # Check if we have room to back up (allow with 5cm margin)
                                back_ok = (back_cm is None) or (float(back_cm) >= (float(self._min_proceed_cm) + 5.0))
                                if back_ok:
                                    print(f"[AUTO-ROOMBA] Pre-exec: obstacle detected (min_front={final_min_front:.1f}cm, fit=L:{left_fit}/R:{right_fit}) - BACKWARD then {turn_dir}")
                                    print(f"[AUTO-SAFETY] Pre-exec: obstacle at {final_min_front:.1f}cm - BACKWARD then {turn_dir}")
                                    cmd = {
                                        "command": "BACKWARD",
                                        "duration": 0.9,
                                        "speed": 75,
                                        "speak": "",
                                        "_followup": {"command": turn_dir, "duration": 0.9, "speed": 100},
                                    }
                                else:
                                    print(f"[AUTO-SAFETY] Pre-exec: obstacle at {final_min_front:.1f}cm - turning (back constrained)")
                                    cmd = {"command": turn_dir, "duration": 1.0, "speed": 100, "speak": ""}
                            else:
                                print(f"[AUTO-SAFETY] Pre-execution check: front obstacle at {final_min_front:.1f}cm - turning instead")
                                turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                                cmd = {"command": turn_dir, "duration": 0.6, "speed": 100, "speak": "Obstacle detected!"}
                    except Exception:
                        pass

                # SAFETY: Pre-execution check for backward (uses back/center sensor).
                # IMPORTANT: Use a tighter threshold than min_proceed to allow escape maneuvers
                # Only block BACKWARD if we're about to hit something (< emergency stop distance)
                if cmd.get('command') == 'BACKWARD':
                    try:
                        back_cm = None
                        if sensor_readings_raw and len(sensor_readings_raw) > 1:
                            back_cm = float(sensor_readings_raw[1]) if (sensor_readings_raw[1] is not None and float(sensor_readings_raw[1]) > 0) else None
                        # Only block if VERY close - allow backing up for escape even with limited clearance
                        back_too_close = back_cm is not None and float(back_cm) < float(self._emergency_stop_cm)
                        if back_too_close:
                            print(f"[AUTO-SAFETY] Blocking BACKWARD (back={float(back_cm):.1f}cm < emergency={float(self._emergency_stop_cm):.1f}cm) - too close to wall")
                            cmd = {"command": "STOP", "duration": 0, "speed": 0, "speak": "Can't reverse, obstacle behind."}
                        elif back_cm is not None and float(back_cm) < float(self._min_proceed_cm):
                            # Reduce speed and duration when backing with limited clearance
                            if isinstance(cmd, dict) and 'duration' in cmd:
                                original_dur = float(cmd.get('duration', 0))
                                cmd['duration'] = min(float(original_dur), 0.5)  # Cap at 0.5s
                                cmd['speed'] = min(int(cmd.get('speed', 75)), 60)  # Reduce to 60%
                                print(f"[AUTO-SAFETY] Limited back clearance ({float(back_cm):.1f}cm) - reducing BACKWARD to {cmd['duration']:.1f}s at {cmd['speed']}%")
                    except Exception:
                        pass

                # CORNER ESCAPE: If the front stays blocked for multiple cycles, do a deterministic escape.
                # This addresses the common failure mode where the robot ends up wedged in a corner and keeps
                # turning/stopping without actually clearing.
                try:
                    enable_corner_escape = str(os.getenv('SARAH_AUTO_CORNER_ESCAPE', '1')).strip().lower() not in ('0', 'false', 'no', 'off')
                    if enable_corner_escape and get_current_mode() == 'autonomous' and (not self.stop_event.is_set()):
                        front_blocked = (min_front is not None and float(min_front) < float(self._min_proceed_cm))
                        if front_blocked:
                            blocked_front_cycles += 1
                        else:
                            blocked_front_cycles = 0

                        try:
                            cooldown_s = float(os.getenv('SARAH_AUTO_CORNER_ESCAPE_COOLDOWN_S', '2.0').strip() or '2.0')
                        except Exception:
                            cooldown_s = 2.0

                        if front_blocked and blocked_front_cycles >= 2 and (time.time() - float(last_corner_escape_ts)) >= float(cooldown_s):
                            # Determine turn direction using the more open front sensor.
                            left_cm = None
                            right_cm = None
                            back_cm = None
                            try:
                                if sensor_readings_raw and len(sensor_readings_raw) >= 3:
                                    left_cm = float(sensor_readings_raw[0]) if (sensor_readings_raw[0] is not None and float(sensor_readings_raw[0]) > 0) else None
                                    right_cm = float(sensor_readings_raw[2]) if (sensor_readings_raw[2] is not None and float(sensor_readings_raw[2]) > 0) else None
                                if sensor_readings_raw and len(sensor_readings_raw) >= 2:
                                    back_cm = float(sensor_readings_raw[1]) if (sensor_readings_raw[1] is not None and float(sensor_readings_raw[1]) > 0) else None
                            except Exception:
                                left_cm = right_cm = back_cm = None

                            if left_cm is None and right_cm is None:
                                turn_dir = 'LEFT' if random.random() < 0.5 else 'RIGHT'
                            elif right_cm is None:
                                turn_dir = 'LEFT'
                            elif left_cm is None:
                                turn_dir = 'RIGHT'
                            else:
                                turn_dir = 'LEFT' if float(left_cm) >= float(right_cm) else 'RIGHT'

                            # Prefer backing up first if possible; if not, do a longer in-place turn.
                            backup_ok = (back_cm is None) or (float(back_cm) >= (float(self._min_proceed_cm) + 5.0))
                            if backup_ok:
                                backup_dur = 0.8
                                try:
                                    backup_dur = float(os.getenv('SARAH_AUTO_CORNER_BACK_SEC', '0.8').strip() or '0.8')
                                except Exception:
                                    backup_dur = 0.8
                                backup_dur = max(0.1, min(float(backup_dur), float(self._max_backward_sec)))

                                turn_dur = 1.1
                                try:
                                    turn_dur = float(os.getenv('SARAH_AUTO_CORNER_TURN_SEC', '1.1').strip() or '1.1')
                                except Exception:
                                    turn_dur = 1.1
                                turn_dur = max(0.2, min(float(turn_dur), float(self._max_turn_sec)))

                                print(f"[AUTO-CORNER] Front blocked ({float(min_front):.1f}cm). Escaping: BACKWARD {backup_dur:.1f}s then {turn_dir} {turn_dur:.1f}s")
                                backup_cmd = {"command": "BACKWARD", "duration": backup_dur, "speed": 75, "speak": ""}
                                turn_cmd = {"command": turn_dir, "duration": turn_dur, "speed": 100, "speak": ""}
                                execute_command(self.drive, backup_cmd, silent=True)
                                try:
                                    self._record_motion('BACKWARD', float(backup_cmd.get('duration', 0) or 0), int(backup_cmd.get('speed', 75) or 75))
                                except Exception:
                                    pass
                                execute_command(self.drive, turn_cmd, silent=True)
                                try:
                                    self._record_motion(turn_dir, float(turn_cmd.get('duration', 0) or 0), int(turn_cmd.get('speed', 100) or 100))
                                except Exception:
                                    pass

                                try:
                                    self._maybe_persist()
                                except Exception:
                                    pass

                                last_corner_escape_ts = time.time()
                                blocked_front_cycles = 0
                                continue
                            else:
                                # Back is blocked too; do a longer turn to try to re-orient.
                                turn_dur = 1.2
                                try:
                                    turn_dur = float(os.getenv('SARAH_AUTO_CORNER_TURN_SEC', '1.2').strip() or '1.2')
                                except Exception:
                                    turn_dur = 1.2
                                turn_dur = max(0.2, min(float(turn_dur), float(self._max_turn_sec)))
                                print(f"[AUTO-CORNER] Front+back constrained. Forcing {turn_dir} {turn_dur:.1f}s")
                                cmd = {"command": turn_dir, "duration": turn_dur, "speed": 100, "speak": ""}
                                # fall through to normal execution
                except Exception:
                    pass
                
                # Optional follow-up maneuver (used for Roomba-style BACKWARD then TURN).
                followup = None
                try:
                    if isinstance(cmd, dict):
                        followup = cmd.get('_followup')
                except Exception:
                    followup = None

                execute_command(self.drive, cmd, silent=True)

                # Record executed motion for dead-reckoning + return-to-start.
                try:
                    executed_cmd = str(cmd.get('command', '')).upper()
                    executed_dur = float(cmd.get('duration', 0) or 0)
                    executed_spd = int(cmd.get('speed', DEFAULT_SPEED) or DEFAULT_SPEED)
                    self._record_motion(executed_cmd, executed_dur, executed_spd)
                except Exception:
                    pass

                # Execute follow-up immediately (no extra decision cycle), and record it as well.
                try:
                    if followup and isinstance(followup, dict) and get_current_mode() == 'autonomous' and (not self.stop_event.is_set()):
                        f_cmd = {
                            'command': str(followup.get('command', 'STOP')).upper(),
                            'duration': float(followup.get('duration', 0) or 0),
                            'speed': int(followup.get('speed', 100) or 100),
                            'speak': '',
                        }
                        execute_command(self.drive, f_cmd, silent=True)
                        try:
                            self._record_motion(str(f_cmd.get('command', '')).upper(), float(f_cmd.get('duration', 0) or 0), int(f_cmd.get('speed', 100) or 100))
                        except Exception:
                            pass
                except Exception:
                    pass

                # Persist after motion/pose updates.
                try:
                    self._maybe_persist()
                except Exception:
                    pass

                # If we stopped due to an obstacle, pause briefly to avoid immediately nudging forward again.
                if cmd.get("command", "").upper() == "STOP":
                    # If vision explicitly said STOP (with obstacles), or if we have any close ultrasonic reading, pause.
                    should_pause = False
                    try:
                        if (
                            camera_analysis
                            and str(camera_analysis.get('recommendation', '')).upper() == 'STOP'
                            and bool(camera_analysis.get('obstacles', False))
                        ):
                            should_pause = True
                    except Exception:
                        pass
                    if min_dist is not None and min_dist <= ULTRASONIC_STOP_DISTANCE_CM:
                        should_pause = True
                    if should_pause:
                        self.stop_event.wait(AUTONOMOUS_OBSTACLE_PAUSE_SECONDS)
                
                # Display movement status with VISION CORRESPONDENCE
                # Optional detailed status logging only when debugging
                if SARAH_DEBUG:
                    if latest_img_b64 and camera_analysis:
                        vision_obstacles = camera_analysis.get('obstacles', False)
                        vision_rec = camera_analysis.get('recommendation', 'PROCEED')
                        if vision_obstacles or vision_rec == 'STOP':
                            print("[VISION->MOVEMENT] OK: OBSTACLE DETECTED -> STOP COMMAND EXECUTED")
                        elif not vision_obstacles or vision_rec == 'PROCEED':
                            if command in ['FORWARD', 'LEFT', 'RIGHT', 'BACKWARD']:
                                print(f"[VISION->MOVEMENT] OK: PATH CLEAR ({vision_rec}) -> {command} COMMAND EXECUTED")
                            else:
                                print(f"[VISION→MOVEMENT] Path clear but robot is {command}")
                    else:
                        print(f"[MOVEMENT] Executing {command} (no vision data available)")

                    print(f"\n[EXECUTING] {command}")
                    if duration > 0:
                        print(f"[EXECUTING] Duration: {duration:.1f}s | Speed: {speed}%")
                    else:
                        print(f"[EXECUTING] Speed: {speed}%")
                
            except Exception as e:
                print(f"[AUTO] Cycle {decision_count} error: {e}")
                try:
                    if self.drive:
                        self.drive.stop()
                except (AttributeError, RuntimeError):
                    pass

                # Persist even if the cycle failed.
                try:
                    self._maybe_persist(force=True)
                except Exception:
                    pass
            
                # ENFORCE 2-SECOND CYCLE TIME: Sleep only as much as needed
            elapsed = time.time() - cycle_start_time
            remaining = max(0, AUTONOMOUS_LOOP_INTERVAL - elapsed)
            
            # VERIFY GPIO CONTINUITY: Check PWM state before sleep
            if remaining > 0 and self.drive and self.drive.use_gpio and hasattr(self.drive, 'pwm_objects'):
                try:
                    left_pwm = self.drive.pwm_objects.get('left')
                    right_pwm = self.drive.pwm_objects.get('right')
                    if left_pwm and right_pwm:
                        left_val = left_pwm.value * 100 if hasattr(left_pwm, 'value') else 0
                        right_val = right_pwm.value * 100 if hasattr(right_pwm, 'value') else 0
                        dprint(SARAH_DEBUG, f"[GPIO-CHECK] Before sleep: PWM LEFT={left_val:.1f}%, RIGHT={right_val:.1f}% (during {remaining:.2f}s)")
                except Exception as e:
                    dprint(SARAH_DEBUG, f"[GPIO-CHECK] Could not verify PWM state: {e}")
            
            if remaining > 0:
                dprint(SARAH_DEBUG, f"[AUTO] Waiting {remaining:.1f}s for next cycle...")
                self.stop_event.wait(remaining)
                
                # VERIFY GPIO CONTINUITY: Check PWM state after sleep to ensure signal maintained
                if self.drive and self.drive.use_gpio and hasattr(self.drive, 'pwm_objects'):
                    try:
                        left_pwm = self.drive.pwm_objects.get('left')
                        right_pwm = self.drive.pwm_objects.get('right')
                        if left_pwm and right_pwm:
                            left_val = left_pwm.value * 100 if hasattr(left_pwm, 'value') else 0
                            right_val = right_pwm.value * 100 if hasattr(right_pwm, 'value') else 0
                            dprint(SARAH_DEBUG, f"[GPIO-CHECK] After sleep: PWM LEFT={left_val:.1f}%, RIGHT={right_val:.1f}%")
                    except Exception as e:
                        dprint(SARAH_DEBUG, f"[GPIO-CHECK] Could not verify PWM state: {e}")
            else:
                print(f"[AUTO] WARNING: Decision took {elapsed:.1f}s (no sleep, exceeds {AUTONOMOUS_LOOP_INTERVAL}s target)")
            
            # MEMORY OPTIMIZATION: Periodic garbage collection for 4GB RAM constraint
            if decision_count % 6 == 0:  # Every 30 seconds
                gc.collect()
                mem_status = get_memory_status()
                if mem_status['percent'] > 75:
                    Logger.log("MEMORY", f"Memory at {mem_status['percent']}% - triggering cleanup", "WARN")

        print("[AUTO] Autonomous thread exiting.")
        try:
            self._maybe_persist(force=True)
        except Exception:
            pass


###############################################
# Camera Thread
###############################################
class CameraThread(threading.Thread):
    def __init__(self, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.cap = None
        self.picam2 = None
        self.use_rpicam = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.camera_enabled = True
        self.rpicam_process = None
        self.temp_jpeg_path = "/tmp/rpicam_frame.jpg"

        # Cache for JPEG/base64 encoding so multiple consumers (or repeated calls)
        # don't re-encode the exact same latest frame.
        self._cached_b64_frame_id = None
        self._cached_b64 = None

    def _try_usb_camera(self):
        """
        Try to initialize USB camera (works on Windows, Linux, and Mac).
        Returns True if successful, False otherwise.
        """
        dprint(CAMERA_DEBUG, "[CAM] Attempting USB camera (Arducam, standard webcam, or integrated)...")
        camera_indices = [0, 1, 2]
        
        # Use appropriate backend for the OS
        system_os = platform.system()
        if system_os == "Windows":
            # DirectShow backend for Windows
            camera_backend = cv2.CAP_DSHOW
            dprint(CAMERA_DEBUG, "[CAM-DEBUG] Using DirectShow backend for Windows")
        elif system_os == "Darwin":
            # macOS uses AVFoundation backend
            camera_backend = cv2.CAP_AVFOUNDATION
            dprint(CAMERA_DEBUG, "[CAM-DEBUG] Using AVFoundation backend for macOS")
        else:
            # V4L2 backend for Linux
            camera_backend = cv2.CAP_V4L2
            dprint(CAMERA_DEBUG, "[CAM-DEBUG] Using V4L2 backend for Linux")
        
        for idx in camera_indices:
            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Trying camera index {idx}...")
            try:
                self.cap = cv2.VideoCapture(idx, camera_backend)
                
                if self.cap.isOpened():
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Camera {idx} opened successfully")
                    
                    # Configure camera for better performance
                    try:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        # Try to set buffer size, but don't fail if not supported
                        try:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for fresh frames
                        except Exception:
                            pass  # Not all cameras support buffer size setting
                    except Exception as config_e:
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Warning: could not set all camera properties: {config_e}")
                    
                    # Give camera time to initialize
                    time.sleep(1)
                    
                    # Verify it can actually read frames
                    max_read_attempts = 5
                    ret = False
                    test_frame = None
                    for attempt in range(max_read_attempts):
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            break
                        time.sleep(0.2)
                    
                    if ret and test_frame is not None:
                        frame_shape = test_frame.shape
                        device_info = f"Camera {idx} ({frame_shape[1]}x{frame_shape[0]})"
                        print(f"[CAM] [OK] Camera ready (USB) - {device_info}")
                        
                        with self.frame_lock:
                            self.latest_frame = test_frame
                            self.frame_ready.set()
                        
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Camera thread started successfully (USB camera)")
                        
                        # Main capture loop - USB camera
                        frame_count = 0
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Entering main capture loop...")
                        
                        while not self.stop_event.is_set():
                            try:
                                ret, frame = self.cap.read()
                                if ret and frame is not None:
                                    with self.frame_lock:
                                        self.latest_frame = frame
                                        frame_count += 1
                                        if frame_count % 30 == 0:
                                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Captured {frame_count} frames from USB camera")
                                else:
                                    # Try to reconnect on frame read failure
                                    try:
                                        self.cap.release()
                                        # Brief delay before reconnecting
                                        time.sleep(0.5)
                                        self.cap = cv2.VideoCapture(idx, camera_backend)
                                        if self.cap.isOpened():
                                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Reconnected to camera {idx}")
                                    except Exception as reconnect_e:
                                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Reconnection failed: {reconnect_e}")
                                        break
                                
                                time.sleep(0.033)  # ~30 FPS
                            except Exception as e:
                                dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Frame capture error: {e}")
                                time.sleep(0.5)
                        
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Exit: captured {frame_count} total frames from USB camera")
                        return True
                    else:
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Camera {idx} opened but could not read frames")
                        self.cap.release()
                else:
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Camera {idx} not available")
            except Exception as e:
                dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Error with camera {idx}: {e}")
        
        return False

    def run(self):
        print(f"[CAM] Initializing camera ({'Raspberry Pi' if IS_RASPBERRY_PI else platform.system()})")
        
        try:
            # Windows: Try USB camera first
            if platform.system() == "Windows":
                dprint(CAMERA_DEBUG, "[CAM] Windows detected - attempting USB camera...")
                if self._try_usb_camera():
                    return
                # If USB camera fails on Windows, disable camera and continue
                print("[CAM] USB camera failed on Windows. Operating without vision.")
                self.camera_enabled = False
                return
            
            # For Raspberry Pi 5 with ribbon cable camera, verify hardware first
            if IS_RASPBERRY_PI:
                dprint(CAMERA_DEBUG, "[CAM-DEBUG] Starting rpicam hardware verification...")
                # Test if rpicam-hello works (same as user verified)
                try:
                    result = subprocess.run(
                        ["rpicam-hello", "--list-cameras"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        errors='ignore'
                    )
                    if result.returncode == 0:
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] [OK] rpicam-hello --list-cameras output:")
                        for line in result.stdout.split('\n')[:10]:  # Limit output lines
                            dprint(CAMERA_DEBUG, f"  {line}")
                    else:
                        error_msg = result.stderr[:200] if result.stderr else "Unknown error"
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] [WARN] rpicam-hello failed: {error_msg}")
                except subprocess.TimeoutExpired:
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] [WARN] rpicam-hello timeout (hardware may be busy)")
                except FileNotFoundError:
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] [WARN] rpicam-hello not found (libcamera may not be installed)")
                except Exception as e:
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] [WARN] Could not run rpicam-hello: {e}")
                
                dprint(CAMERA_DEBUG, "[CAM] Attempting to use picamera2 for Raspberry Pi camera (CSI ribbon cable via rpicam)...")
                try:
                    # picamera2 already imported at top if available
                    if not PICAMERA2_AVAILABLE:
                        raise ImportError("picamera2 not available")
                    
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Creating Picamera2 instance...")
                    try:
                        if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                            self.picam2 = Picamera2(camera_num=CSI_CAMERA_INDEX)
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Using CSI_CAMERA_INDEX={CSI_CAMERA_INDEX}")
                        else:
                            self.picam2 = Picamera2()
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Picamera2 instance created successfully")
                    except Exception as e:
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Failed to create Picamera2: {e}")
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] This may require: sudo chmod 644 /dev/video*")
                        raise
                    
                    # Configure camera
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Creating camera configuration (320x240, XRGB8888)...")
                    try:
                        config = self.picam2.create_preview_configuration(
                            main={"format": 'XRGB8888', "size": (320, 240)}
                        )
                        self.picam2.configure(config)
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Camera configured")
                    except Exception as e:
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Configuration failed: {e}")
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Trying with default configuration...")
                        config = self.picam2.create_preview_configuration()
                        self.picam2.configure(config)
                    
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Starting camera...")
                    self.picam2.start()
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Camera started")
                    
                    # Warm up camera with multiple captures
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Warming up camera (capturing 3 frames)...")
                    for i in range(3):
                        try:
                            warm_frame = self.picam2.capture_array()
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Warmup frame {i+1}: {warm_frame.shape if warm_frame is not None else 'None'}")
                        except Exception as capture_e:
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Warmup capture {i+1} failed: {capture_e}")
                        time.sleep(0.3)
                    
                    # Test first real frame
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Capturing test frame...")
                    test_frame = self.picam2.capture_array()
                    if test_frame is None or test_frame.size == 0:
                        raise IOError("picamera2 initialized but returned empty frame")
                    
                    print("[CAM] [OK] Camera ready (CSI / picamera2)")
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Got first frame: {test_frame.shape}")
                    
                    with self.frame_lock:
                        # Convert XRGB to BGR for OpenCV compatibility
                        try:
                            if len(test_frame.shape) == 3 and test_frame.shape[2] >= 3:
                                # Handle both RGB and RGBA formats
                                if test_frame.shape[2] == 4:
                                    self.latest_frame = cv2.cvtColor(test_frame[:,:,:3], cv2.COLOR_RGB2BGR)
                                else:
                                    self.latest_frame = cv2.cvtColor(test_frame[:,:,:3], cv2.COLOR_RGB2BGR)
                            else:
                                self.latest_frame = test_frame
                        except Exception as convert_e:
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Frame conversion warning: {convert_e}")
                            self.latest_frame = test_frame
                        self.frame_ready.set()
                    
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Camera thread started successfully (CSI ribbon cable / picamera2 / rpicam)")
                    
                    # Main capture loop - picamera2 native
                    frame_count = 0
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Entering main capture loop...")
                    while not self.stop_event.is_set():
                        try:
                            frame = self.picam2.capture_array()
                            if frame is not None and frame.size > 0:
                                with self.frame_lock:
                                    # Convert XRGB to BGR for OpenCV compatibility
                                    if len(frame.shape) == 3 and frame.shape[2] >= 3:
                                        self.latest_frame = cv2.cvtColor(frame[:,:,:3], cv2.COLOR_RGB2BGR)
                                    else:
                                        self.latest_frame = frame
                                    frame_count += 1
                                    if frame_count % 30 == 0:
                                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Captured {frame_count} frames via picamera2")
                            else:
                                dprint(CAMERA_DEBUG, "[CAM-DEBUG] [WARN] Got empty frame from picamera2")
                            
                            time.sleep(0.067)  # ~15 FPS
                        except Exception as e:
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Frame capture error: {e}")
                            time.sleep(0.5)
                    
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Exit: captured {frame_count} total frames from picamera2")
                    return
                    
                except ImportError:
                    dprint(CAMERA_DEBUG, "[CAM] picamera2 not available")
                except PermissionError as perm_e:
                    print(f"[CAM] Permission denied accessing camera: {perm_e}")
                    print(f"[CAM] Try: sudo usermod -a -G video $(whoami)")
                    print(f"[CAM] Then log out and log back in")
                except Exception as e:
                    error_msg = str(e)[:200]
                    print(f"[CAM] picamera2 initialization failed: {error_msg}")
                    if 'libcamera' in error_msg.lower():
                        print(f"[CAM] libcamera error - try: sudo apt-get install -y libcamera-tools")
                    if CAMERA_DEBUG:
                        import traceback
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Traceback:")
                        traceback.print_exc()
                
                # Fallback: Try rpicam/libcamera if picamera2 fails
                dprint(CAMERA_DEBUG, "[CAM] Attempting rpicam/libcamera as fallback...")
                try:
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Starting rpicam capture process...")
                    # Start a libcamera-based process that continuously captures frames.
                    # First try a single output path overwritten on each timelapse tick.
                    # If that fails (some builds don't overwrite as expected), retry with a numbered pattern.
                    output_path = "/tmp/rpicam_frame.jpg"
                    pattern_output = "/tmp/rpicam_frame_%04d.jpg"
                    latest_output = "/tmp/rpicam_latest.jpg"
                    use_pattern_mode = False
                    use_latest_file = False

                    rpicam_stderr_path = "/tmp/sarah_rpicam_stderr.txt"

                    def _tail_text_file(path: str, max_chars: int = 1200) -> str:
                        try:
                            if not os.path.exists(path):
                                return "(stderr file not created)"
                            with open(path, "rb") as f:
                                f.seek(0, os.SEEK_END)
                                size = f.tell()
                                start = max(0, size - max_chars)
                                f.seek(start)
                                data = f.read().decode(errors="replace")
                            data = data.strip()
                            return data if data else "(stderr empty)"
                        except Exception as _e:
                            return "(failed to read stderr file)"
                    def _start_capture_process(cmd_to_run):
                        # Capture stderr to a file so we can debug even while the process is running.
                        try:
                            with open(rpicam_stderr_path, "w", encoding="utf-8", errors="replace") as _:
                                pass
                        except Exception:
                            pass

                        try:
                            err_fh = open(rpicam_stderr_path, "a", encoding="utf-8", errors="replace")
                        except Exception:
                            err_fh = None

                        try:
                            return subprocess.Popen(
                                cmd_to_run,
                                stdout=subprocess.DEVNULL,
                                stderr=err_fh if err_fh is not None else subprocess.PIPE,
                                text=True,
                                bufsize=1,
                            )
                        finally:
                            # Keep the file handle open while process runs ONLY if it's attached.
                            # When stderr is PIPE, there's nothing to keep.
                            if err_fh is not None:
                                try:
                                    # Do not close; owned by child process while running.
                                    pass
                                except Exception:
                                    pass

                    # Clean up any old numbered frames from previous runs
                    try:
                        for f in os.listdir("/tmp"):
                            if f.startswith("rpicam_frame_") and f.endswith(".jpg"):
                                try:
                                    os.remove(os.path.join("/tmp", f))
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except Exception:
                        pass

                    try:
                        if os.path.exists(latest_output):
                            os.remove(latest_output)
                    except Exception:
                        pass

                    # IMPORTANT: continuous output requires timelapse.
                    # Prefer rpicam-still with --zsl when available (libcamera recommends it).
                    # rpicam-still timelapse is most reliable with a numbered output pattern.
                    if _which("rpicam-still"):
                        tool_name = "rpicam-still"
                        use_pattern_mode = True
                        output_path = pattern_output
                        cmd = [
                            "rpicam-still",
                            "--zsl",
                            "-t", "0",
                            "-n",
                            "--width", "320",
                            "--height", "240",
                            "--framerate", "15",
                            "--timelapse", "100",
                            "-o", output_path,
                        ]
                    else:
                        # rpicam-jpeg flag aliases vary across versions; use -t/-o (most common).
                        tool_name = "rpicam-jpeg"
                        cmd = [
                            "rpicam-jpeg",
                            "-t", "0",
                            "-n",
                            "--width", "320",
                            "--height", "240",
                            "--framerate", "15",
                            "--timelapse", "100",
                            "-o", output_path,
                        ]

                    if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                        cmd.extend(["--camera", str(CSI_CAMERA_INDEX)])

                    try:
                        self.rpicam_process = _start_capture_process(cmd)
                    except FileNotFoundError:
                        tool_name = "libcamera-still"
                        cmd = [
                            "libcamera-still",
                            "--timeout", "0",
                            "--nopreview",
                            "--width", "320",
                            "--height", "240",
                            "--timelapse", "100",
                            "-o", output_path,
                        ]

                        if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                            cmd.extend(["--camera", str(CSI_CAMERA_INDEX)])
                        self.rpicam_process = _start_capture_process(cmd)

                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] {tool_name} process started (PID: {self.rpicam_process.pid})")
                    
                    def _wait_for_first_frame_single_file(path: str):
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Waiting up to {CAMERA_FIRST_FRAME_TIMEOUT_SECONDS} seconds for first camera frame...")
                        for wait_sec in range(CAMERA_FIRST_FRAME_TIMEOUT_SECONDS):
                            time.sleep(1)
                            try:
                                if os.path.exists(path):
                                    size = os.path.getsize(path)
                                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Check {wait_sec+1}: {os.path.basename(path)} Size={size} bytes")
                                    if size > 0:
                                        return path
                                else:
                                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Check {wait_sec+1}: {os.path.basename(path)} not created yet")
                            except Exception as scan_e:
                                dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Check {wait_sec+1}: Error scanning /tmp: {scan_e}")
                        return None

                    def _newest_pattern_frame():
                        try:
                            candidates = []
                            for f in os.listdir("/tmp"):
                                if f.startswith("rpicam_frame_") and f.endswith(".jpg"):
                                    full = os.path.join("/tmp", f)
                                    try:
                                        if os.path.getsize(full) > 0:
                                            candidates.append(full)
                                    except Exception:
                                        pass
                            if not candidates:
                                return None
                            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            return candidates[0]
                        except Exception:
                            return None

                    def _oneshot_capture_jpeg() -> Optional[str]:
                        """Capture a single JPEG to /tmp and return its path on success, else None."""
                        oneshot_path = "/tmp/rpicam_oneshot.jpg"
                        try:
                            if os.path.exists(oneshot_path):
                                os.remove(oneshot_path)
                        except Exception:
                            pass

                        # Prefer rpicam-still --zsl if available; else rpicam-jpeg; else libcamera-still.
                        cmd_oneshot: Optional[list[str]] = None
                        if _which("rpicam-still"):
                            cmd_oneshot = [
                                "rpicam-still",
                                "--zsl",
                                "-t", "200",  # ms
                                "-n",
                                "--width", "320",
                                "--height", "240",
                                "-o", oneshot_path,
                            ]
                            if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                                cmd_oneshot.extend(["--camera", str(CSI_CAMERA_INDEX)])
                        elif _which("rpicam-jpeg"):
                            cmd_oneshot = [
                                "rpicam-jpeg",
                                "-t", "200",  # ms
                                "-n",
                                "--width", "320",
                                "--height", "240",
                                "-o", oneshot_path,
                            ]
                            if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                                cmd_oneshot.extend(["--camera", str(CSI_CAMERA_INDEX)])
                        elif _which("libcamera-still"):
                            cmd_oneshot = [
                                "libcamera-still",
                                "--timeout", "200",
                                "--nopreview",
                                "--width", "320",
                                "--height", "240",
                                "-o", oneshot_path,
                            ]
                            if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                                cmd_oneshot.extend(["--camera", str(CSI_CAMERA_INDEX)])

                        if not cmd_oneshot:
                            return None

                        try:
                            subprocess.run(
                                cmd_oneshot,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=4,
                            )
                        except Exception:
                            return None

                        try:
                            if os.path.exists(oneshot_path) and os.path.getsize(oneshot_path) > 0:
                                return oneshot_path
                        except Exception:
                            pass
                        return None

                    if use_pattern_mode and not use_latest_file:
                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Waiting up to {CAMERA_FIRST_FRAME_TIMEOUT_SECONDS} seconds for numbered frames...")
                        first_frame_path = None
                        for _ in range(CAMERA_FIRST_FRAME_TIMEOUT_SECONDS):
                            time.sleep(1)
                            candidate = _newest_pattern_frame()
                            if candidate:
                                first_frame_path = candidate
                                break
                    else:
                        first_frame_path = _wait_for_first_frame_single_file(output_path)

                    # Retry with numbered pattern output (and --latest if supported) if the single-file mode didn't work.
                    if not first_frame_path or not os.path.exists(first_frame_path):
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Single-file output did not appear; retrying with numbered output pattern...")

                        # Stop the previous capture process before restarting.
                        try:
                            if self.rpicam_process:
                                self.rpicam_process.terminate()
                                self.rpicam_process.wait(timeout=2)
                        except Exception:
                            try:
                                if self.rpicam_process:
                                    self.rpicam_process.kill()
                            except Exception:
                                pass

                        # Clean old outputs
                        try:
                            if os.path.exists(latest_output):
                                os.remove(latest_output)
                        except Exception:
                            pass

                        # Decide if rpicam-jpeg supports --latest
                        use_latest_file = False
                        if tool_name == "rpicam-jpeg":
                            try:
                                help_out = subprocess.run(
                                    ["rpicam-jpeg", "--help"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    timeout=5,
                                ).stdout or ""
                                if "--latest" in help_out:
                                    use_latest_file = True
                            except Exception:
                                use_latest_file = False

                        use_pattern_mode = True

                        if tool_name == "rpicam-still":
                            cmd = [
                                "rpicam-still",
                                "--zsl",
                                "-t", "0",
                                "-n",
                                "--width", "320",
                                "--height", "240",
                                "--framerate", "15",
                                "--timelapse", "100",
                                "-o", pattern_output,
                            ]
                            if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                                cmd.extend(["--camera", str(CSI_CAMERA_INDEX)])
                        elif tool_name == "rpicam-jpeg":
                            cmd = [
                                "rpicam-jpeg",
                                "-t", "0",
                                "-n",
                                "--width", "320",
                                "--height", "240",
                                "--framerate", "15",
                                "--timelapse", "100",
                                "-o", pattern_output,
                            ]
                            if use_latest_file:
                                cmd.extend(["--latest", latest_output])
                            if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                                cmd.extend(["--camera", str(CSI_CAMERA_INDEX)])
                        else:
                            # libcamera-still: try numbered pattern as output
                            cmd = [
                                "libcamera-still",
                                "--timeout", "0",
                                "--nopreview",
                                "--width", "320",
                                "--height", "240",
                                "--timelapse", "100",
                                "-o", pattern_output,
                            ]
                            if CSI_CAMERA_INDEX is not None and CSI_CAMERA_INDEX >= 0:
                                cmd.extend(["--camera", str(CSI_CAMERA_INDEX)])

                        self.rpicam_process = _start_capture_process(cmd)

                        dprint(CAMERA_DEBUG, f"[CAM-DEBUG] {tool_name} pattern process started (PID: {self.rpicam_process.pid})")

                        if use_latest_file:
                            first_frame_path = _wait_for_first_frame_single_file(latest_output)
                            if first_frame_path:
                                output_path = latest_output
                        else:
                            # Wait for any numbered frame to appear
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Waiting up to {CAMERA_FIRST_FRAME_TIMEOUT_SECONDS} seconds for numbered frames...")
                            for _ in range(CAMERA_FIRST_FRAME_TIMEOUT_SECONDS):
                                time.sleep(1)
                                candidate = _newest_pattern_frame()
                                if candidate:
                                    first_frame_path = candidate
                                    break

                    # Verify we can read the first frame
                    if not first_frame_path or not os.path.exists(first_frame_path):
                        # FINAL FALLBACK: single-shot capture (avoids long-lived process issues)
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Continuous capture produced no frames; trying single-shot capture...")
                        oneshot_path = None
                        for _ in range(max(1, int(CAMERA_FIRST_FRAME_TIMEOUT_SECONDS / 2))):
                            oneshot_path = _oneshot_capture_jpeg()
                            if oneshot_path:
                                first_frame_path = oneshot_path
                                break
                            time.sleep(0.25)

                        if not first_frame_path or not os.path.exists(first_frame_path):
                            # Check if process has exited and only then read stderr (avoid blocking read() while running)
                            self.rpicam_process.poll()
                            stderr_output = _tail_text_file(rpicam_stderr_path)
                            raise IOError(
                                f"{tool_name} did not create frame files within {CAMERA_FIRST_FRAME_TIMEOUT_SECONDS} seconds. "
                                f"mode={'pattern' if use_pattern_mode else 'single'} returncode={self.rpicam_process.returncode} "
                                f"stderr_tail: {stderr_output}"
                            )

                    oneshot_mode = (first_frame_path == "/tmp/rpicam_oneshot.jpg")

                    # If we fall back to one-shot capture, we must stop any long-lived rpicam/libcamera process.
                    # Otherwise the camera can remain busy and one-shot calls may fail.
                    if oneshot_mode and self.rpicam_process:
                        try:
                            self.rpicam_process.terminate()
                            self.rpicam_process.wait(timeout=2)
                        except Exception:
                            try:
                                self.rpicam_process.kill()
                            except Exception:
                                pass
                        self.rpicam_process = None
                    
                    # Read and verify first frame
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Attempting to read first frame...")
                    frame = cv2.imread(first_frame_path)
                    if frame is None:
                        file_size = os.path.getsize(first_frame_path)
                        raise IOError(f"Could not read first frame from {tool_name} (file size: {file_size} bytes)")

                    print(f"[CAM] [OK] Camera ready (CSI / {tool_name})")
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Got first frame: {frame.shape}")
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] File size: {os.path.getsize(first_frame_path)} bytes")
                    
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_ready.set()
                    
                    self.use_rpicam = True
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Camera thread started successfully ({tool_name})")
                    
                    # Main capture loop
                    frame_count = 0
                    read_errors = 0
                    last_mtime = None
                    last_path = None
                    dprint(CAMERA_DEBUG, "[CAM-DEBUG] Entering main capture loop...")
                    
                    while not self.stop_event.is_set():
                        try:
                            # If we ended up in one-shot mode, periodically invoke rpicam to refresh the frame.
                            if oneshot_mode:
                                refreshed = _oneshot_capture_jpeg()
                                if refreshed:
                                    current_path = refreshed
                                else:
                                    current_path = first_frame_path
                            else:
                                current_path = output_path
                                if use_pattern_mode and not use_latest_file:
                                    candidate = _newest_pattern_frame()
                                    if candidate:
                                        current_path = candidate

                            if current_path and os.path.exists(current_path):
                                try:
                                    mtime = os.path.getmtime(current_path)
                                    if last_mtime is not None and last_path == current_path and mtime == last_mtime:
                                        time.sleep(0.01)
                                        continue

                                    file_size = os.path.getsize(current_path)
                                    file_age = time.time() - mtime
                                    if file_size > 0 and file_age < 2.0:
                                        frame = cv2.imread(current_path)
                                        if frame is not None:
                                            with self.frame_lock:
                                                self.latest_frame = frame
                                                frame_count += 1
                                                read_errors = 0
                                                last_mtime = mtime
                                                last_path = current_path
                                            if frame_count % 30 == 0:
                                                dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Captured {frame_count} frames from {tool_name} (size: {file_size}, age: {file_age:.2f}s)")
                                        else:
                                            read_errors += 1
                                            if read_errors <= 5:
                                                dprint(CAMERA_DEBUG, f"[CAM-DEBUG] [WARN] Could not decode JPEG {os.path.basename(current_path)} (attempt {read_errors})")
                                except Exception:
                                    pass
                            else:
                                if current_path:
                                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] [WARN] {os.path.basename(current_path)} not found yet")
                            
                            time.sleep(0.067)  # ~15 FPS
                        except Exception as e:
                            dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Frame read error: {e}")
                            time.sleep(0.5)
                    
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Exit: captured {frame_count} total frames from {tool_name}")
                    return
                except Exception as e:
                    print(f"[CAM] rpicam/libcamera initialization failed: {e}")
                    # Ensure the capture process doesn't keep running if first-frame capture failed.
                    if self.rpicam_process:
                        try:
                            self.rpicam_process.terminate()
                            self.rpicam_process.wait(timeout=2)
                        except Exception:
                            try:
                                self.rpicam_process.kill()
                            except Exception:
                                pass

                    if CAMERA_DEBUG:
                        import traceback
                        dprint(CAMERA_DEBUG, "[CAM-DEBUG] Traceback:")
                        traceback.print_exc()
            
            # Fallback on Raspberry Pi: Try USB camera if ribbon cable and rpicam-jpeg failed
            if IS_RASPBERRY_PI:
                dprint(CAMERA_DEBUG, "[CAM] Attempting USB camera as fallback (in case ribbon cable failed)...")
                if self._try_usb_camera():
                    return
            
            # On Raspberry Pi, if we reach here, both picamera2 and rpicam failed
            # This is a ribbon cable camera failure, not USB camera issue
            if IS_RASPBERRY_PI:
                raise IOError("Ribbon cable camera (picamera2/rpicam) initialization failed on Raspberry Pi")
        
        except Exception as e:
            print(f"[CAM] [FAIL] Camera init failed: {e}")
            print("[CAM] Vision disabled (robot will continue without vision).")
            # When libcamera says "no cameras available", it's almost always OS/hardware/config,
            # not a Python/OpenCV issue.
            err_lower = str(e).lower() if e else ""
            if IS_RASPBERRY_PI and ("no cameras available" in err_lower or "no cameras" in err_lower):
                print("[CAM] libcamera reports NO CAMERAS AVAILABLE.")
                print("[CAM] This usually means the Pi does not detect the CSI camera at all.")
                print("[CAM] Quick checks on the Pi:")
                print("[CAM]   1) Reseat ribbon cable (correct orientation + fully latched)")
                print("[CAM]   2) Run: rpicam-hello --list-cameras")
                print("[CAM]   3) Run: libcamera-hello --list-cameras (if installed)")
                print("[CAM]   4) Ensure camera stack installed:")
                print("[CAM]        sudo apt-get update && sudo apt-get install -y rpicam-apps libcamera-tools python3-picamera2")
                print("[CAM]   5) Reboot: sudo reboot")
            dprint(CAMERA_DEBUG, "[CAM-DEBUG] Troubleshooting: rpicam-hello --list-cameras ; rpicam-jpeg -t 5 -o /tmp/test.jpg")
            self.camera_enabled = False
            speak("Ribbon cable camera initialization failed. I'm operating without vision.")
        finally:
            if self.rpicam_process:
                try:
                    self.rpicam_process.terminate()
                    self.rpicam_process.wait(timeout=2)
                except (AttributeError, subprocess.TimeoutExpired):
                    try:
                        self.rpicam_process.kill()
                    except:
                        pass
            if self.picam2:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except (AttributeError, RuntimeError):
                    pass
            if self.cap:
                try:
                    self.cap.release()
                except (AttributeError, RuntimeError):
                    pass
            # Clean up temp file
            try:
                if os.path.exists(self.temp_jpeg_path):
                    os.remove(self.temp_jpeg_path)
            except:
                pass
            print("[CAM] Camera thread stopped.")

    def wait_for_frame(self, timeout=5.0):
        """
        Waits for the camera to capture its first frame.
        Returns True if frame is ready, False on timeout.
        """
        return self.frame_ready.wait(timeout)

    def get_frame_base64(self):
        """
        Returns the latest frame as base64-encoded JPEG string for remote Ollama server.
        Optimizes frame size for network transmission while maintaining vision model quality.
        Returns None if camera is not available or frame encoding fails.
        """
        if not self.camera_enabled or self.latest_frame is None:
            return None
            
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            try:
                frame_id = id(self.latest_frame)
                if self._cached_b64 is not None and self._cached_b64_frame_id == frame_id:
                    return self._cached_b64

                # For remote Ollama server, balance compression with vision model accuracy
                # Lower quality = smaller transmission size but may hurt vision model accuracy
                # Higher quality = larger size but better model accuracy
                # On RPi5 with network transmission, 65 is optimal (better than 60 for network latency)
                quality = 65 if IS_RASPBERRY_PI else 75  # Optimized for remote transmission
                
                # Ensure frame is in BGR format (required by cv2.imencode)
                frame_to_encode = self.latest_frame
                if len(frame_to_encode.shape) == 2:  # Grayscale
                    frame_to_encode = cv2.cvtColor(frame_to_encode, cv2.COLOR_GRAY2BGR)
                
                success, buffer = cv2.imencode('.jpg', frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not success:
                    dprint(CAMERA_DEBUG, "[CAM] Failed to encode frame to JPEG")
                    return None
                
                # Convert buffer to base64 string for transmission to remote server
                jpeg_bytes = buffer.tobytes()
                b64_string = base64.b64encode(jpeg_bytes).decode('utf-8')

                self._cached_b64_frame_id = frame_id
                self._cached_b64 = b64_string
                
                # Log size info for diagnostics (only on first few frames to reduce spam)
                if hasattr(self, '_frame_log_count'):
                    self._frame_log_count += 1
                else:
                    self._frame_log_count = 1
                
                if self._frame_log_count <= 3:
                    dprint(CAMERA_DEBUG, f"[CAM-DEBUG] Encoded frame: {len(jpeg_bytes)} bytes → {len(b64_string)} bytes base64 (quality={quality})")
                
                return b64_string
            except Exception as e:
                dprint(CAMERA_DEBUG, f"[CAM] Error encoding frame: {e}")
                return None

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


###############################################
# SpeechRecognition Microphone helpers
###############################################
_SR_MIC_DEVICE_INDEX_OVERRIDE_SET = False
_SR_MIC_DEVICE_INDEX_OVERRIDE = None


def _preferred_sr_mic_device_index():
    """Return the preferred SpeechRecognition device_index.

    Default behavior (all platforms): auto-detect (`None`).

    If you explicitly set `SARAH_MIC_DEVICE_INDEX` to a non-negative integer,
    we pass it through to `speech_recognition.Microphone(device_index=...)`.

    Note for Raspberry Pi/Linux:
    - ALSA "card numbers" from `arecord -l` are NOT the same as PyAudio device
      indices used by SpeechRecognition.
    - Auto-detect is usually the most reliable choice unless you're sure you
      have the correct PyAudio input index.
    """
    global _SR_MIC_DEVICE_INDEX_OVERRIDE_SET, _SR_MIC_DEVICE_INDEX_OVERRIDE

    if not _SR_MIC_DEVICE_INDEX_OVERRIDE_SET:
        override = os.getenv("SARAH_MIC_DEVICE_INDEX", "").strip()
        if override == "":
            _SR_MIC_DEVICE_INDEX_OVERRIDE = None
        elif override.lower() in ("none", "auto", "default"):
            _SR_MIC_DEVICE_INDEX_OVERRIDE = None
        elif override.isdigit():
            _SR_MIC_DEVICE_INDEX_OVERRIDE = int(override)
        else:
            print(f"[VOICE] [WARN] Invalid SARAH_MIC_DEVICE_INDEX='{override}'. Using auto-detect.")
            _SR_MIC_DEVICE_INDEX_OVERRIDE = None
        _SR_MIC_DEVICE_INDEX_OVERRIDE_SET = True

    return _SR_MIC_DEVICE_INDEX_OVERRIDE


def _preferred_sr_mic_sample_rate():
    """Return preferred sample rate for SpeechRecognition mic.

    Default is `None` (let PortAudio choose the device default), which is the
    most compatible setting across USB mics.

    You can override with `SARAH_MIC_SAMPLE_RATE` (e.g. 16000).
    """
    override = os.getenv("SARAH_MIC_SAMPLE_RATE", "").strip()
    if override.isdigit():
        return int(override)
    return None


_UNSET = object()


def create_sr_microphone(*, sample_rate=None, chunk_size=1024, device_index=_UNSET, allow_env_overrides: bool = True):
    """Create a SpeechRecognition Microphone with safe defaults.

    - By default, uses env overrides (SARAH_MIC_DEVICE_INDEX / SARAH_MIC_SAMPLE_RATE).
    - For recovery fallbacks, set allow_env_overrides=False to let PortAudio pick
      the device-default sample rate (often avoids paInvalidSampleRate).
    """
    if device_index is _UNSET:
        device_index = _preferred_sr_mic_device_index()

    kwargs = {"chunk_size": chunk_size}

    if sample_rate is None and allow_env_overrides:
        sample_rate = _preferred_sr_mic_sample_rate()

    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    if device_index is not None:
        kwargs["device_index"] = device_index
    return sr.Microphone(**kwargs)


@contextmanager
def safe_microphone_source(microphone: "sr.Microphone"):
    """Safely enter/exit a SpeechRecognition `Microphone`.

    On some Windows setups, entering the mic can yield a source with
    `source.stream is None`, which makes `adjust_for_ambient_noise()` raise
    and then `Microphone.__exit__()` can crash with `'NoneType' has no attribute close'`.

    This wrapper:
    - validates `source.stream` before yielding
    - skips the problematic `__exit__` path when the stream is missing
    """
    source = None
    exc_info = (None, None, None)
    try:
        silence = (platform.system() == "Linux") and (not SARAH_DEBUG)
        with _silence_stderr_fd(silence):
            source = microphone.__enter__()
        if getattr(source, "stream", None) is None:
            # Best-effort cleanup without triggering `stream.close()` on None.
            try:
                audio_obj = getattr(source, "audio", None) or getattr(microphone, "audio", None)
                if audio_obj is not None and hasattr(audio_obj, "terminate"):
                    audio_obj.terminate()
            except Exception:
                pass
            raise RuntimeError("Microphone stream not opened (device/PyAudio issue)")

        yield source
    except Exception:
        exc_info = sys.exc_info()
        raise
    finally:
        try:
            if source is not None and getattr(source, "stream", None) is not None:
                microphone.__exit__(*exc_info)
        except Exception:
            pass


###############################################
# Game Movement Thread (for game mode physical robot movement)
###############################################
class GameMovementThread(threading.Thread):
    """Translates game events into robot movements."""
    def __init__(self, drive, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.drive = drive
        self.stop_event = stop_event
        self._stop_event = threading.Event()
        self._last_score_milestone = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        print("[GAME-MOVE] Game movement thread started")
        last_mode = None

        while (not self._stop_event.is_set()) and (not self.stop_event.is_set()):
            try:
                mode = CURRENT_MODE
            except Exception:
                mode = None

            if mode != "game":
                if last_mode == "game":
                    try:
                        if self.drive:
                            self.drive.stop()
                    except Exception:
                        pass
                last_mode = mode
                time.sleep(0.1)
                continue

            if last_mode != "game":
                last_mode = "game"
                self._last_score_milestone = 0
                print("[GAME-MOVE] Game movement active")

            # Check for game events
            events = get_game_events()

            # Jump = quick forward burst
            if events.get('jump', False):
                try:
                    execute_command(self.drive, {"command": "FORWARD", "duration": 0.25, "speed": 55, "speak": ""}, silent=True)
                except Exception:
                    pass

            # Duck = quick backward
            if events.get('duck', False):
                try:
                    execute_command(self.drive, {"command": "BACKWARD", "duration": 0.2, "speed": 50, "speak": ""}, silent=True)
                except Exception:
                    pass

            # Game over = stop
            if events.get('game_over', False):
                try:
                    if self.drive:
                        self.drive.stop()
                except Exception:
                    pass

            # Celebrate score milestones with a spin
            score = events.get('score', 0.0)
            milestone = int(score // 50)
            if milestone > self._last_score_milestone and milestone > 0:
                self._last_score_milestone = milestone
                try:
                    execute_command(self.drive, {"command": "LEFT", "duration": 0.18, "speed": 50, "speak": ""}, silent=True)
                except Exception:
                    pass

            time.sleep(0.05)

        try:
            if self.drive:
                self.drive.stop()
        except Exception:
            pass
        print("[GAME-MOVE] Game movement thread stopped")


###############################################
# Dance Thread (for dance mode physical robot movement + LED effects)
###############################################
class DanceThread(threading.Thread):
    def __init__(self, drive, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.drive = drive
        self.stop_event = stop_event
        self._stop_event = threading.Event()
        self._audio_proc: Optional[subprocess.Popen] = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        print("[DANCE] Thread started")

        # Vivid "party" palette (fast-cycling, high contrast)
        palette: list[tuple[int, int, int]] = [
            (255, 0, 80),
            (255, 60, 0),
            (255, 160, 0),
            (255, 255, 0),
            (60, 255, 0),
            (0, 255, 160),
            (0, 220, 255),
            (40, 160, 255),
            (120, 80, 255),
            (220, 0, 255),
            (255, 0, 180),
        ]

        beat_s = float(os.getenv("SARAH_DANCE_BEAT_S", "0.5"))
        beat_s = max(0.25, min(1.2, beat_s))
        step_ttl = min(0.85, beat_s * 1.2)
        strobe_ttl = min(0.25, max(0.10, beat_s * 0.35))

        last_mode = None
        palette_i = 0
        step_i = 0
        dance_time = 0.0  # Track elapsed time for smooth animations
        update_rate = 0.05  # Update eyes smoothly at 20Hz
        drift_x = 0.0
        drift_y = 0.0
        blink_t = -1.0
        blink_dur = 0.30
        blink_depth = 0.90
        next_blink = dance_time + random.uniform(4.0, 9.0)

        while (not self._stop_event.is_set()) and (not self.stop_event.is_set()):
            try:
                mode = CURRENT_MODE
            except Exception:
                mode = None

            if mode != "dance":
                if last_mode == "dance":
                    _stop_process(self._audio_proc)
                    self._audio_proc = None
                    set_avatar_gaze(None, None, ttl_s=0.01)
                    set_avatar_eye_color(None, ttl_s=0.01)
                    set_avatar_eye_open(None, None, ttl_s=0.01)
                    try:
                        if self.drive:
                            self.drive.stop()
                    except Exception:
                        pass
                last_mode = mode
                dance_time = 0.0
                time.sleep(0.1)
                continue

            if last_mode != "dance":
                song_path = _resolve_dance_song_path()
                self._audio_proc = _start_mp3_playback(song_path)
                last_mode = "dance"
                palette_i = 0
                step_i = 0
                dance_time = 0.0
                drift_x = 0.0
                drift_y = 0.0
                blink_t = -1.0
                next_blink = dance_time + random.uniform(4.0, 9.0)
                print("[DANCE] Dance mode active")

            # If the song finished, automatically return to autonomous.
            try:
                auto_return = os.getenv('SARAH_DANCE_RETURN_AUTONOMOUS', '1').strip().lower() not in ('0', 'false', 'no', 'off')
            except Exception:
                auto_return = True
            try:
                if auto_return and self._audio_proc is not None and (self._audio_proc.poll() is not None):
                    print("[DANCE] Song finished - returning to autonomous")
                    _stop_process(self._audio_proc)
                    self._audio_proc = None
                    # Switch back to autonomous exploration.
                    try:
                        request_explore(source='dance_song_end')
                    except Exception:
                        set_current_mode('autonomous', source='dance_song_end')
                    # Loop will see mode change and clean up.
                    time.sleep(0.05)
                    continue
            except Exception:
                pass

            # Increment time at start of loop for consistent timing
            dance_time += update_rate

            # Determine current phase
            phase = int(dance_time / beat_s) % 4
            phase_time = (dance_time % beat_s) / beat_s  # 0.0 to 1.0 within current beat
            
            # Complex multi-directional eye movement using Lissajous curves
            # Multiple frequencies create varied, organic motion in all directions
            angle1 = dance_time * 3.2
            angle2 = dance_time * 4.7
            angle3 = dance_time * 2.1
            angle4 = dance_time * 5.3
            
            # Combine multiple sine waves with different frequencies for rich motion.
            # Keep amplitudes < 1.0 to reduce clipping (which can look directionally skewed).
            gaze_x = (math.sin(angle1) * 0.35 +
                     math.cos(angle2 * 0.8) * 0.25 +
                     math.sin(angle3 * 1.7) * 0.20 +
                     math.cos(angle4 * 0.5) * 0.15)
            gaze_y = (math.cos(angle1 * 0.9) * 0.33 +
                     math.sin(angle2 * 1.2) * 0.25 +
                     math.cos(angle3 * 1.5) * 0.18 +
                     math.sin(angle4 * 0.7) * 0.14)
            
            # Add slight phase-based bias to emphasize movement direction
            if phase == 0:  # FORWARD
                gaze_y -= 0.10
            elif phase == 1:  # BACKWARD
                gaze_y += 0.10
            elif phase == 2:  # LEFT
                gaze_x -= 0.12
            else:  # RIGHT
                gaze_x += 0.12

            # Drift-centering to prevent long-term bias to one side.
            drift_x = drift_x * 0.98 + gaze_x * 0.02
            drift_y = drift_y * 0.98 + gaze_y * 0.02
            gaze_x -= drift_x
            gaze_y -= drift_y
            
            # Clamp to valid range
            gaze_x = max(-1.0, min(1.0, gaze_x))
            gaze_y = max(-1.0, min(1.0, gaze_y))
            
            # Smooth color transitions
            color = palette[palette_i % len(palette)]

            # Intermittent blinks (both eyes together; rarer + slightly longer)
            def _blink_shape(p: float) -> float:
                # p in [0..1] -> 0..1..0 smooth
                s = math.sin(math.pi * max(0.0, min(1.0, p)))
                return s * s

            open_amt = 1.0
            if blink_t >= 0.0:
                blink_t += update_rate
                open_amt = 1.0 - (blink_depth * _blink_shape(blink_t / max(0.12, blink_dur)))
                if blink_t >= blink_dur:
                    blink_t = -1.0
                    # Partial blinks (eyebrow movements) more frequent, full blinks less frequent
                    if random.random() < 0.7:  # 70% chance of partial blink next
                        next_blink = dance_time + random.uniform(1.5, 3.5)
                    else:  # 30% chance of full blink next
                        next_blink = dance_time + random.uniform(5.0, 10.0)
            elif dance_time >= next_blink:
                blink_t = 0.0
                # Choose between eyebrow movement (partial) or full blink
                if random.random() < 0.7:  # 70% partial (eyebrow-like)
                    blink_dur = random.uniform(0.15, 0.25)  # Faster
                    blink_depth = random.uniform(0.25, 0.50)  # Shallow (eyebrow raise/lower)
                else:  # 30% full blink
                    blink_dur = random.uniform(0.28, 0.45)  # Slower
                    blink_depth = random.uniform(0.85, 0.95)  # Deep (full blink)

            open_amt = max(0.08, min(1.0, open_amt))
            left_open = open_amt
            right_open = open_amt
            
            # Update gaze frequently for fluid motion
            set_avatar_gaze(gaze_x, gaze_y, ttl_s=update_rate * 3)
            set_avatar_eye_color(color, ttl_s=update_rate * 3)
            set_avatar_eye_open(left_open, right_open, ttl_s=update_rate * 3)
            
            # Execute robot movement at phase transitions
            # SAFETY: Short, gentle movements since robot is blind (no sensors in dance mode)
            if phase != step_i % 4:
                step_i = phase + int(dance_time / beat_s)
                palette_i += 1
                
                if phase == 0:
                    # Short forward waddle
                    execute_command(self.drive, {"command": "FORWARD", "duration": min(0.15, beat_s * 0.3), "speed": 40, "speak": ""}, silent=True)
                elif phase == 1:
                    # Short backward waddle
                    execute_command(self.drive, {"command": "BACKWARD", "duration": min(0.15, beat_s * 0.3), "speed": 40, "speak": ""}, silent=True)
                elif phase == 2:
                    # Brief left turn (waddle effect)
                    execute_command(self.drive, {"command": "LEFT", "duration": min(0.12, beat_s * 0.25), "speed": 45, "speak": ""}, silent=True)
                else:
                    # Brief right turn (waddle effect)
                    execute_command(self.drive, {"command": "RIGHT", "duration": min(0.12, beat_s * 0.25), "speed": 45, "speak": ""}, silent=True)
                
                # Brief strobe flash at phase transition
                set_avatar_eye_color((255, 255, 255), ttl_s=strobe_ttl)

            time.sleep(update_rate)

        _stop_process(self._audio_proc)
        self._audio_proc = None
        try:
            if self.drive:
                self.drive.stop()
        except Exception:
            pass
        print("[DANCE] Thread stopped")


class VoiceListenerThread(threading.Thread):
    def __init__(self, drive, stop_event: threading.Event, camera_thread: CameraThread):
        super().__init__(daemon=True)
        self.drive = drive
        self.running = True
        self.stop_event = stop_event
        self.camera_thread = camera_thread
        self.recognizer = sr.Recognizer()
        # Optimize speech recognition for accuracy
        self.recognizer.energy_threshold = 4000  # Sensitivity (lower = more sensitive, but more false positives)
        self.recognizer.dynamic_energy_threshold = True  # Auto-adjust to environment
        self.recognizer.phrase_threshold = 0.3  # More lenient phrase detection
        # Faster end-of-utterance detection for snappier responses.
        try:
            self.recognizer.pause_threshold = 0.6
            self.recognizer.non_speaking_duration = 0.3
        except Exception:
            pass
        try:
            sr_device_index = _preferred_sr_mic_device_index()
            if sr_device_index is not None:
                print(f"[VOICE] Initializing microphone (SpeechRecognition device_index={sr_device_index})...")
            else:
                print("[VOICE] Initializing microphone with auto-detection...")
            self.microphone = create_sr_microphone(sample_rate=None, chunk_size=1024)
        except (OSError, IndexError) as e: # Catch both OSError and IndexError for microphone issues
            print(f"[VOICE] Microphone init failed: {e}")
            print("[VOICE] Trying auto-detection as fallback...")
            try:
                self.microphone = create_sr_microphone(sample_rate=None, chunk_size=1024)
                print("[VOICE] [OK] Auto-detected microphone")
            except Exception as e2:
                print(f"[VOICE] Microphone initialization failed: {e2}")
                print("[VOICE] TROUBLESHOOTING:")
                print("  - Check USB microphone is connected")
                print("  - Run: arecord -l  (to list audio devices)")
                print("  - Run: python3 -m speech_recognition  (to test microphone)")
                print("[VOICE] Audio input will be unavailable")
                self.microphone = None
        # system prompt instructing the model to respond with JSON for movement commands when user speaks
        self.system_prompt = (
            "You are SARAH, a friendly and helpful robot assistant. "
            "The user is speaking to you via voice. Provide concise, friendly, and conversational responses. "
            "Be helpful and natural in your speech. Keep responses brief for audio playback. "
            "If asked to perform a movement, respond with JSON format: "
            "{\\\"command\\\": \\\"FORWARD\\\", \\\"duration\\\": 2.0, \\\"speed\\\": 100, \\\"speak\\\": \\\"Moving forward now.\\\", \\\"emotion\\\": \\\"excited\\\"}. "
            "The 'emotion' field controls your facial expression. Valid emotions: neutral, happy, excited, thinking, surprised, concerned, sad, listening, proud. "
            "Choose emotions that match the situation and your response tone. "
            "For conversational responses (no movement), just speak naturally but still express appropriate emotion through the 'emotion' field if you want. "
            "Example conversational: 'I'm happy to help you!' (emotion: happy). Example movement: JSON format with emotion field."
        )
        self.vision_prompt = (
            "You are SARAH, a robot assistant with vision. The user is asking you to describe what you see.\n"
            "Analyze the image carefully and provide ONE CONCISE SENTENCE describing the scene.\n"
            "Be accurate and factual. Do not hallucinate or invent details.\n"
            "Focus on real obstacles, paths, and objects directly ahead.\n"
            "Example: 'I see a doorway in front of me.' or 'The path ahead is clear.'"
        )

    def run(self):
        if not self.microphone:
            print("[VOICE] No microphone available; voice activation disabled.")
            return

        # Calibrate once up-front.
        # Re-calibrating continuously can make recognition fail in MANUAL mode because
        # motor noise gets treated as "ambient" and pushes the energy threshold too high.
        try:
            with safe_microphone_source(self.microphone) as source:
                print("[VOICE] Calibrating microphone (ambient noise)...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print(f"[VOICE] Calibration complete. energy_threshold={int(getattr(self.recognizer, 'energy_threshold', 0))}")
            # Freeze the threshold after calibration to keep it stable while driving.
            self.recognizer.dynamic_energy_threshold = False
        except Exception as e:
            print(f"[VOICE] [WARN] Calibration failed; continuing without calibration: {e}")

        print("[VOICE] Voice listener started. Say 'Sarah' to issue a command.")
        consecutive_open_failures = 0
        attempted_relaxed_fallback = False
        while not self.stop_event.is_set():
            audio = None
            try:
                # Keep ALL audio capture operations inside an entered AudioSource.
                # Use a safe wrapper to avoid Windows edge cases where `source.stream` is None.
                try:
                    with safe_microphone_source(self.microphone) as source:
                        print("[VOICE] Listening...")
                        audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=8)
                    consecutive_open_failures = 0
                except sr.WaitTimeoutError:
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    consecutive_open_failures += 1
                    print(f"[VOICE] Listen error: {e}")

                    # Recovery strategies: try multiple fallback approaches
                    if consecutive_open_failures >= 3 and (not attempted_relaxed_fallback):
                        attempted_relaxed_fallback = True
                        print("[VOICE] [WARN] Attempting microphone recovery (auto-detect + default sample rate)...")
                        
                        # Strategy 1: Auto-detect without env overrides
                        try:
                            self.microphone = create_sr_microphone(
                                sample_rate=None,
                                chunk_size=1024,
                                device_index=None,
                                allow_env_overrides=False,
                            )
                            print("[VOICE] [OK] Microphone recovery successful (strategy 1: auto-detect)")
                            consecutive_open_failures = 0
                            time.sleep(0.2)
                            continue
                        except Exception as rebuild_e:
                            print(f"[VOICE] [WARN] Strategy 1 failed: {rebuild_e}")
                        
                        # Strategy 2: Try with original settings but smaller chunk size
                        try:
                            self.microphone = create_sr_microphone(
                                sample_rate=None,
                                chunk_size=512,  # Smaller chunks can help with some devices
                                device_index=_preferred_sr_mic_device_index(),
                                allow_env_overrides=True,
                            )
                            print("[VOICE] [OK] Microphone recovery successful (strategy 2: smaller chunks)")
                            consecutive_open_failures = 0
                            time.sleep(0.2)
                            continue
                        except Exception as rebuild_e2:
                            print(f"[VOICE] [WARN] Strategy 2 failed: {rebuild_e2}")

                    if consecutive_open_failures >= 10:
                        print("[VOICE] [WARN] Repeated microphone open failures; disabling voice listener.")
                        return
                    time.sleep(1)
                    continue

                # Audio processing outside the context manager (safe)
                if audio is None:
                    time.sleep(0.1)
                    continue

                try:
                    text = self.recognizer.recognize_google(audio)
                    print("[VOICE] Heard:", text)
                except sr.UnknownValueError:
                    print("[VOICE] Could not understand audio.")
                    time.sleep(0.2)
                    continue
                except sr.RequestError as e:
                    # Check for FLAC error specifically
                    if 'FLAC' in str(e) or 'flac' in str(e):
                        print("[VOICE] Error: FLAC conversion utility not available")
                        print("[VOICE]   Install with: sudo apt-get install flac")
                        print("[VOICE]   Or: brew install flac (on macOS)")
                        # Try to install it automatically on Raspberry Pi
                        if IS_RASPBERRY_PI:
                            try:
                                print("[VOICE] Attempting automatic FLAC installation...")
                                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'flac'],
                                             timeout=60, capture_output=True)
                                print("[VOICE] FLAC installed. Please restart.")
                            except Exception as install_e:
                                print(f"[VOICE] Auto-install failed: {install_e}")
                        time.sleep(2)
                    else:
                        error_msg = f"Speech recognition error: {e}"
                        print(f"[VOICE] {error_msg}")
                        # Do not speak recognition errors.
                        time.sleep(1)
                    continue
                except Exception as e:
                    print(f"[VOICE] Recognition error: {e}")
                    time.sleep(0.5)
                    continue

                text_lower = text.lower()
                activation = (ACTIVATION_WORD.lower() in text_lower)

                # Allow exiting GAME/DANCE without saying the activation word.
                # This is intentionally limited to when we're already in that mode.
                allow_special_exit = False
                try:
                    cur_mode = get_current_mode()
                except Exception:
                    cur_mode = ""
                try:
                    if cur_mode == "game" and any(p in text_lower for p in ("stop game mode", "exit game mode", "quit game mode", "end game mode", "stop game", "exit game")):
                        allow_special_exit = True
                    elif cur_mode == "dance" and any(p in text_lower for p in ("stop dancing", "stop dance", "end dance", "end dancing", "quit dancing", "stop dance mode", "end dance mode")):
                        allow_special_exit = True
                except Exception:
                    allow_special_exit = False

                if activation or allow_special_exit:
                    # Process voice input asynchronously so listening never blocks
                    print("[VOICE] Activation word detected, processing asynchronously...")
                    # Keep this snappy: only wait briefly if TTS is currently speaking.
                    try:
                        if TTS_ENABLED and hasattr(TTS_LOCK, "locked") and TTS_LOCK.locked():
                            time.sleep(0.2)
                        else:
                            time.sleep(0.05)
                    except Exception:
                        time.sleep(0.05)
                    threading.Thread(
                        target=self._handle_voice_input,
                        args=(text,),
                        daemon=True
                    ).start()
            except Exception as e:
                print(f"[VOICE] Unexpected error: {type(e).__name__}: {e}")
                time.sleep(1)
            
            time.sleep(0.1)

    def _handle_voice_input(self, text: str):
        """Handle voice input asynchronously without blocking the listening thread."""
        try:
            user_prompt = text
            text_lower = text.lower()
            
            # Check for visual query keywords (be specific to avoid misrouting movement commands)
            # Vision queries: "what do you see", "describe what you see", "look at this", "show me what you see"
            # NOT vision: "look left", "turn and look", "go forward and look"
            vision_triggers = [
                "what do you see",
                "what can you see",
                "describe what",
                "tell me what you see",
                "show me what",
                "look at this",
                "look at that",
                "view the",
            ]
            is_vision_query = any(trigger in text_lower for trigger in vision_triggers)
            # Also allow natural phrasing like "what's ahead" / "describe the scene",
            # but require a vision-related keyword so we don't misroute general questions.
            if not is_vision_query:
                vision_keywords = [
                    "see",
                    "look",
                    "ahead",
                    "in front",
                    "infront",
                    "camera",
                    "scene",
                    "surroundings",
                    "around you",
                    "view",
                ]
                if ("describe" in text_lower or "what" in text_lower) and any(k in text_lower for k in vision_keywords):
                    is_vision_query = True
            # Exclude if movement command words are present
            movement_words = ["left", "right", "forward", "backward", "turn", "move", "go", "drive"]
            if is_vision_query and any(word in text_lower for word in movement_words):
                is_vision_query = False
            
            if is_vision_query:
                print("[VOICE] Visual query detected.")
                img_b64 = self.camera_thread.get_frame_base64()
                if img_b64:
                    print("[VOICE] Got camera frame, querying vision model...")
                    try:
                        # Handle both remote and local AI modes
                        if AI_MODE == "remote":
                            if hasattr(client, 'chat'):
                                with avatar_ai_activity():
                                    response = client.chat(
                                        model='llava',
                                        messages=[
                                            {'role': 'system', 'content': self.vision_prompt},
                                            {'role': 'user', 'content': user_prompt, 'images': [img_b64]}
                                        ],
                                        stream=False,
                                        options={"temperature": 0.05, "num_predict": 200}
                                    )
                            else:
                                llama_speak("Vision analysis is not available on the remote server.")
                                return
                        else:
                            # Local Ollama with llava
                            with avatar_ai_activity():
                                response = client.chat(
                                    model='llava',
                                    messages=[
                                        {'role': 'system', 'content': self.vision_prompt},
                                        {'role': 'user', 'content': user_prompt, 'images': [img_b64]}
                                    ],
                                    stream=False,
                                    options={"temperature": 0.05, "num_predict": 200}
                                )
                        description = response['message']['content'].strip()
                        print(f"[VOICE] Vision response: '{description[:60]}...'" if len(description) > 60 else f"[VOICE] Vision response: '{description}'")
                        llama_speak(description)
                    except Exception as vision_err:
                        error_msg = "I couldn't analyze the image."
                        print(f"[VOICE] Error: {vision_err}")
                        llama_speak(error_msg)
                else:
                    print("[VOICE] Camera frame unavailable.")
                    llama_speak("I can't see anything right now, my camera might be offline.")
            else: # It's a command or question
                # Check for mode switching commands first
                if any(p in text_lower for p in ("stop game mode", "exit game mode", "quit game mode", "end game mode", "stop game", "exit game")):
                    print("[VOICE] Game mode stop requested")
                    llama_speak("Stopping game mode.")
                    if self.drive:
                        self.drive.stop()
                    exit_game_mode(source="voice")
                    llama_speak(f"Mode is now {get_current_mode()}.")
                    return
                elif ("not" not in text_lower):
                    # Game mode voice trigger: tolerate common phrasing variations.
                    # Examples: "activate game mode", "switch to game mode", "start the game mode", "play dino game".
                    wants_game = False
                    try:
                        if any(k in text_lower for k in ("game mode", "dino game", "dinosaur game", "dino runner")):
                            if any(v in text_lower for v in ("activate", "start", "enter", "switch", "go", "play", "launch")):
                                wants_game = True
                    except Exception:
                        wants_game = False

                    if wants_game:
                        print("[VOICE] Mode switch requested: GAME")
                        if self.drive:
                            self.drive.stop()
                        request_game(source="voice")
                        llama_speak("Starting game mode.")
                        llama_speak("Mode is now game.")
                        return
                if any(p in text_lower for p in ("stop dancing", "stop dance", "end dance", "end dancing", "quit dancing", "stop dance mode", "end dance mode")):
                    print("[VOICE] Dance stop requested")
                    llama_speak("Stopping dance mode.")
                    if self.drive:
                        self.drive.stop()
                    exit_dance_mode(source="voice")
                    return
                elif any(p in text_lower for p in ("dance", "start dancing", "dance mode")) and ("not" not in text_lower and "stop" not in text_lower):
                    print("[VOICE] Mode switch requested: DANCE")
                    if self.drive:
                        self.drive.stop()
                    request_dance(source="voice")
                    llama_speak("Let's dance!")
                    return
                elif "manual" in text_lower or "manual mode" in text_lower:
                    print("[VOICE] Mode switch requested: MANUAL")
                    # Manual requires a controller; otherwise fall back to autonomous.
                    if not is_controller_connected():
                        if self.drive:
                            self.drive.stop()
                        set_current_mode("autonomous", source="voice_no_controller")
                        llama_speak("No controller detected. Switching to autonomous mode instead.")
                        llama_speak("Mode is now autonomous.")
                        return

                    # Stop current operations and switch mode immediately, then speak.
                    if self.drive:
                        self.drive.stop()
                    set_current_mode("manual", source="voice")
                    llama_speak("Switching to manual control mode. Please use your controller.")
                    llama_speak("Mode is now manual.")
                    return
                elif ("autonomous" in text_lower or "autonomous mode" in text_lower) and "not" not in text_lower:
                    print("[VOICE] Mode switch requested: AUTONOMOUS")
                    # Stop current operations and switch mode immediately, then speak.
                    if self.drive:
                        self.drive.stop()
                    set_current_mode("autonomous", source="voice")
                    llama_speak("Switching to autonomous mode.")
                    llama_speak("Mode is now autonomous.")
                    return
                
                # First, check for simple direct voice commands (stop, forward, left, right, etc.)
                # Voice commands work in ALL modes (manual, autonomous, chat)
                simple_cmd, is_simple = parse_simple_voice_command(text_lower)
                
                if is_simple and simple_cmd:
                    # Execute the simple command directly without Llama
                    current_mode = get_current_mode()
                    print(f"[VOICE] Matched simple command: {simple_cmd['command']} (mode: {current_mode})")
                    print("\n" + "="*70)
                    print(f"  [MIC] VOICE COMMAND: {simple_cmd['command']}")
                    print(f"  Speed: {simple_cmd['speed']}% | Duration: {simple_cmd['duration']}s")
                    print("="*70 + "\n")
                    llama_speak(simple_cmd['speak'])
                    # Execute command regardless of mode - voice overrides autonomous
                    execute_command(self.drive, simple_cmd)
                else:
                    # Not a simple command, try parsing as complex movement command
                    cmd = parse_freeform_model_response(text_lower)
                    
                    # Check if it's clearly a movement command
                    is_movement_command = (cmd["command"] != "STOP" or cmd["duration"] > 0) and \
                                        any(word in text_lower for word in ["forward", "backward", "back", "left", "right", "move"])
                    
                    if is_movement_command:
                        # It's a complex movement command (with duration/speed specifications)
                        print(f"[VOICE] Parsed complex movement command: {cmd}")
                        print("\n" + "="*70)
                        print(f"  [MIC] VOICE COMMAND: {cmd['command']}")
                        print(f"  Speed: {cmd['speed']}% | Duration: {cmd['duration']}s")
                        print("="*70 + "\n")
                        if cmd["command"] != "STOP":
                            llama_speak(f"Okay, moving {cmd['command'].lower()} for {cmd['duration']} seconds.")
                        else:
                            llama_speak("Okay, stopping.")
                        execute_command(self.drive, cmd)
                    else:
                        # It's a general question/statement - use Llama3.1 with TTS
                        print("[VOICE] Processing as general question with Llama3.1...")
                        # Activation word 'Sarah' was detected in voice input - allow TTS to respond
                        query_llama_and_speak(
                            prompt=user_prompt,
                            system_prompt=self.system_prompt,
                            speak_response=True,  # Auto-speak response
                            activation_detected=True  # 'Sarah' was detected - allow TTS
                        )
        except Exception as e:
            print(f"[VOICE] Error in async processing: {e}")
            import traceback
            traceback.print_exc()


###############################################
# Controller thread (if joystick present)
###############################################
class ControllerThread(threading.Thread):
    def __init__(self, drive, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.drive = drive
        self.stop_event = stop_event
        self.speed = DEFAULT_SPEED
        self.joystick = None
        self.controller_connected = False
        self.last_connection_attempt = 0
        self.reconnection_delay = 2.0  # Wait 2 seconds before retrying connection
        self._last_dpad = (0, 0)  # (x,y) normalized to -1/0/1
        self._last_disconnect_ts = None

    def _apply_dpad(self, x: int, y: int):
        """Apply D-pad direction to drive outputs (manual) or game inputs (game)."""
        mode = get_current_mode()
        if mode not in ("manual", "game"):
            return
        x = int(max(-1, min(1, x)))
        y = int(max(-1, min(1, y)))

        if (x, y) == self._last_dpad:
            return
        self._last_dpad = (x, y)

        if mode == "game":
            set_game_dpad(x, y)
            return

        if (x, y) == (0, 1):
            print(f"[CTRL] FORWARD @ {self.speed}%")
            if self.drive:
                self.drive.forward(self.speed)
        elif (x, y) == (0, -1):
            print(f"[CTRL] BACKWARD @ {self.speed}%")
            if self.drive:
                self.drive.backward(self.speed)
        elif (x, y) == (-1, 0):
            print(f"[CTRL] LEFT @ {self.speed}%")
            if self.drive:
                self.drive.left(self.speed)
        elif (x, y) == (1, 0):
            print(f"[CTRL] RIGHT @ {self.speed}%")
            if self.drive:
                self.drive.right(self.speed)
        elif (x, y) == (0, 0):
            print("[CTRL] STOP")
            if self.drive:
                self.drive.stop()

    def run(self):
        if not pygame:
            print("[CTRL] pygame not available; controller disabled.")
            set_controller_connected(False)
            return

        last_mode = None
        mode_seq = _get_mode_seq()

        try:
            pygame.init()
            pygame.joystick.init()
            print("[CTRL] Pygame initialized - waiting for controller (USB or Bluetooth)...")

            # We start disconnected until we successfully initialize a joystick.
            set_controller_connected(False)

            while not self.stop_event.is_set():
                try:
                    current_mode = get_current_mode()
                    if last_mode != current_mode:
                        # If we are leaving manual mode, ensure motors are stopped once.
                        if last_mode == "manual" and current_mode != "manual":
                            try:
                                if self.drive:
                                    self.drive.stop()
                            except Exception:
                                pass
                        last_mode = current_mode
                        mode_seq = _get_mode_seq()

                    # Check for controller connection
                    current_count = pygame.joystick.get_count()
                    
                    if current_count == 0:
                        if self.controller_connected:
                            print("[CTRL] Controller disconnected. Waiting for reconnection...")
                            self.controller_connected = False
                            self.joystick = None
                            set_controller_connected(False)
                            self._last_disconnect_ts = time.time()

                            # Safety: stop motors immediately on disconnect.
                            try:
                                if self.drive:
                                    self.drive.stop()
                            except Exception:
                                pass
                        time.sleep(0.5)
                        continue
                    
                    # Controller detected
                    if not self.controller_connected:
                        print(f"[CTRL] Detected {current_count} controller(s)")
                        try:
                            self.joystick = pygame.joystick.Joystick(0)
                            self.joystick.init()
                            controller_name = self.joystick.get_name()
                            print(f"[CTRL] [OK] Controller connected: {controller_name}")
                            
                            # Detect Xbox controller
                            is_xbox = "xbox" in controller_name.lower() or "xinput" in controller_name.lower()
                            is_bluetooth = "bluetooth" in controller_name.lower() or "wireless" in controller_name.lower()
                            
                            if is_xbox:
                                print(f"[CTRL] [OK] Xbox Controller detected!")
                            else:
                                print(f"[CTRL] Generic gamepad detected (Xbox-compatible layout)")
                            
                            print(f"[CTRL] Connection: {'Bluetooth' if is_bluetooth else 'USB/Wired'}")
                            print(f"[CTRL] Buttons: {self.joystick.get_numbuttons()} | Axes: {self.joystick.get_numaxes()} | Hats: {self.joystick.get_numhats()}")
                            print(f"[CTRL] Ready for control: Use D-Pad for movement, A=STOP, LB/RB=Speed modifiers")
                            # Reset last dpad state when (re)connecting.
                            self._last_dpad = (0, 0)
                            self.controller_connected = True
                            set_controller_connected(True)
                            self._last_disconnect_ts = None
                        except Exception as e:
                            print(f"[CTRL] Failed to initialize controller: {e}")
                            self.controller_connected = False
                            set_controller_connected(False)
                            time.sleep(self.reconnection_delay)
                            continue
                    
                    # Process controller input
                    if self.controller_connected and self.joystick:
                        for event in pygame.event.get():
                            # Handle D-Pad (hat) input for movement
                            if event.type == pygame.JOYHATMOTION:
                                # Prefer the event's value (more reliable than querying get_hat during the callback).
                                try:
                                    x, y = event.value
                                except Exception:
                                    try:
                                        x, y = self.joystick.get_hat(getattr(event, "hat", 0))
                                    except Exception:
                                        x, y = (0, 0)
                                self._apply_dpad(int(x), int(y))
                            
                            # Handle button input (Xbox controller buttons)
                            elif event.type == pygame.JOYBUTTONDOWN:
                                # Xbox button mapping: A=0, B=1, X=2, Y=3, LB=4, RB=5, Back=6, Start=7
                                button_map = {
                                    0: "A (STOP)", 
                                    1: "B", 
                                    2: "X", 
                                    3: "Y",
                                    4: "LB",
                                    5: "RB",
                                    6: "Back",
                                    7: "Start"
                                }
                                button_name = button_map.get(event.button, f"Button{event.button}")
                                print(f"[CTRL] Xbox {button_name} pressed")
                                
                                # A button (0) = emergency stop
                                if event.button == 0:
                                    # Emergency stop + terminate (restore original behavior).
                                    print("[CTRL] EMERGENCY STOP (A button) - terminating")
                                    try:
                                        if self.drive:
                                            self.drive.stop()
                                    finally:
                                        self._last_dpad = (0, 0)
                                        self.stop_event.set()
                                # LB (4) and RB (5) can be used for speed control or other functions
                                elif event.button == 4:
                                    print(f"[CTRL] LB pressed - speed modifier available")
                                elif event.button == 5:
                                    print(f"[CTRL] RB pressed - speed modifier available")
                            
                            # Handle button release
                            elif event.type == pygame.JOYBUTTONUP:
                                print(f"[CTRL] Button {event.button} released")
                            
                            # Handle axis input (analog sticks and triggers)
                            elif event.type == pygame.JOYAXISMOTION:
                                # Xbox Controller axis mapping:
                                # 0 = Left stick X, 1 = Left stick Y
                                # 2 = Left trigger (LT), 3 = Right stick X, 4 = Right stick Y, 5 = Right trigger (RT)
                                
                                # Left trigger (LT) - axis 2
                                if event.axis == 2:
                                    normalized_value = (event.value + 1) / 2  # Convert from -1..1 to 0..1
                                    if normalized_value > 0.1:
                                        print(f"[CTRL] LEFT TRIGGER (LT) @ {int(normalized_value * 100)}%")
                                
                                # Right trigger (RT) - axis 5
                                elif event.axis == 5:
                                    normalized_value = (event.value + 1) / 2  # Convert from -1..1 to 0..1
                                    if normalized_value > 0.1:
                                        print(f"[CTRL] RIGHT TRIGGER (RT) @ {int(normalized_value * 100)}%")
                                
                                # Left analog stick
                                elif event.axis in [0, 1]:
                                    pass  # Can be used for advanced analog movement control
                                
                                # Right analog stick
                                elif event.axis in [3, 4]:
                                    pass  # Can be used for camera/aiming control in future

                                # D-Pad fallback: many Xbox controllers expose D-pad as axes 6/7 on SDL.
                                # Axis 6: left/right (-1 left, +1 right). Axis 7: up/down (-1 up, +1 down).
                                elif event.axis in (6, 7):
                                    if get_current_mode() not in ("manual", "game"):
                                        continue
                                    try:
                                        ax6 = float(self.joystick.get_axis(6)) if self.joystick.get_numaxes() > 6 else 0.0
                                    except Exception:
                                        ax6 = 0.0
                                    try:
                                        ax7 = float(self.joystick.get_axis(7)) if self.joystick.get_numaxes() > 7 else 0.0
                                    except Exception:
                                        ax7 = 0.0
                                    dx = -1 if ax6 < -0.5 else (1 if ax6 > 0.5 else 0)
                                    # SDL commonly reports up as -1, down as +1; convert to y=+1 up, y=-1 down.
                                    dy = 1 if ax7 < -0.5 else (-1 if ax7 > 0.5 else 0)
                                    self._apply_dpad(dx, dy)
                            
                            # Handle controller disconnection (JOYDEVICEREMOVED)
                            elif event.type == pygame.JOYDEVICEREMOVED:
                                print(f"[CTRL] Controller device removed (possible Bluetooth disconnect)")
                                self.controller_connected = False
                                self.joystick = None
                                set_controller_connected(False)
                                self._last_disconnect_ts = time.time()
                                try:
                                    if self.drive:
                                        self.drive.stop()
                                except Exception:
                                    pass
                        
                except Exception as e:
                    print(f"[CTRL] Error handling event: {e}")
                    self.controller_connected = False
                    set_controller_connected(False)

                # Polling fallback (covers missed events and some drivers that don't emit hat events reliably).
                # Also important because the avatar thread pumps pygame events.
                if self.controller_connected and self.joystick and get_current_mode() in ("manual", "game"):
                    try:
                        dx, dy = (0, 0)
                        if self.joystick.get_numhats() > 0:
                            hx, hy = self.joystick.get_hat(0)
                            dx, dy = int(hx), int(hy)
                        else:
                            # Axis-based D-pad fallback
                            ax6 = float(self.joystick.get_axis(6)) if self.joystick.get_numaxes() > 6 else 0.0
                            ax7 = float(self.joystick.get_axis(7)) if self.joystick.get_numaxes() > 7 else 0.0
                            dx = -1 if ax6 < -0.5 else (1 if ax6 > 0.5 else 0)
                            dy = 1 if ax7 < -0.5 else (-1 if ax7 > 0.5 else 0)
                        self._apply_dpad(dx, dy)
                    except Exception:
                        pass

                # When not in manual mode, avoid burning CPU; still process events often enough
                # for emergency stop and reconnect handling.
                # If the controller has been disconnected for longer than the grace period and
                # we're still in manual, fall back to autonomous.
                try:
                    if self._last_disconnect_ts is not None:
                        elapsed = time.time() - float(self._last_disconnect_ts)
                        if elapsed >= float(CONTROLLER_DISCONNECT_GRACE_S):
                            # Grace period expired; if still in manual with no controller, fall back.
                            if get_current_mode() == "manual":
                                print(f"[CTRL] Controller disconnect grace period expired ({CONTROLLER_DISCONNECT_GRACE_S:.1f}s). Switching to autonomous.")
                                set_current_mode("autonomous", source="controller_disconnect_grace")
                            # Clear the disconnect timestamp so we don't keep checking.
                            self._last_disconnect_ts = None
                except Exception:
                    pass

                if get_current_mode() == "manual":
                    time.sleep(CONTROLLER_POLL_INTERVAL)
                else:
                    # Prefer event-driven wakeups, but keep a timeout to ensure we still
                    # detect controller disconnects/reconnects and process emergency stop.
                    # Keep this responsive (<= controller poll interval) so emergency stop latency
                    # isn't worse than before.
                    # Note: we still need to wake periodically to pump pygame events;
                    # otherwise emergency stop is only checked on timeout.
                    mode_seq = _wait_for_mode_change(mode_seq, timeout=0.05)
        except Exception as e:
            print(f"[CTRL] Controller initialization error: {e}")
        finally:
            try:
                pygame.quit()
            except (RuntimeError, AttributeError):
                pass
            set_controller_connected(False)
            print("[CTRL] Controller thread exited.")


###############################################
# Voice helper for one-off commands
###############################################
def listen_for_response(duration=5):
    """Listens for a single voice command and returns the text."""
    recognizer = sr.Recognizer()
    # Optimize for accuracy
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.phrase_threshold = 0.3
    try:
        mic = create_sr_microphone(sample_rate=None, chunk_size=1024)
    except OSError as e: # Specifically catch OSError for microphone issues
        print(f"[VOICE] Failed to open microphone for response: {e}")
        return ""

    print(f"[VOICE] Listening for a response for {duration} seconds...")
    try:
        try:
            with safe_microphone_source(mic) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
        except sr.WaitTimeoutError:
            print("[VOICE] No response heard.")
            return ""

        try:
            text = recognizer.recognize_google(audio)
            print(f"[VOICE] Heard response: '{text}'")
            return text.lower()
        except sr.UnknownValueError:
            print("[VOICE] Could not understand audio.")
        except sr.RequestError as e:
            if 'FLAC' in str(e) or 'flac' in str(e):
                print("[VOICE] Error: FLAC conversion utility not available")
                print("[VOICE]   Install with: sudo apt-get install -y flac")
                if IS_RASPBERRY_PI:
                    try:
                        print("[VOICE] Attempting automatic FLAC installation...")
                        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'flac'], timeout=120, capture_output=True)
                        print("[VOICE] FLAC installed. Please restart.")
                    except Exception as install_e:
                        print(f"[VOICE] Auto-install failed: {install_e}")
            else:
                print(f"[VOICE] Speech recognition error: {e}")
        return ""
    except Exception as e:
        print(f"[VOICE] Error listening for response: {e}")
        return ""
###############################################
# Simple Chat Mode (voice in, voice out)
###############################################
def run_simple_chat_mode(stop_event: threading.Event, camera_thread: CameraThread = None):
    """
    Simple chat mode: speak freely, SARAH responds with voice.
    No mode selection, just pure conversation.
    Press Ctrl+C to exit.
    """
    recognizer = sr.Recognizer()
    # Optimize for accuracy
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    recognizer.phrase_threshold = 0.3
    try:
        microphone = create_sr_microphone(sample_rate=None, chunk_size=1024)
    except OSError as e:
        print(f"[CHAT] Microphone error: {e}")
        speak("Microphone not available.")
        return

    system_prompt = (
        "You are SARAH, a friendly and helpful robot assistant. The user is chatting with you. "
        "Keep your responses friendly, concise (1-2 sentences), and natural. "
        "Be conversational and helpful with whatever they ask."
    )
    print("[CHAT] Simple Chat Mode started. Speak freely (Ctrl+C to exit).")
    speak("Chat mode ready. You can talk to me now.")
    
    while not stop_event.is_set():
        try:
            try:
                set_avatar_emotion("listening")
                with safe_microphone_source(microphone) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    print("[CHAT] Listening...")
                    audio = recognizer.listen(source, timeout=33, phrase_time_limit=33)
            except sr.WaitTimeoutError:
                print("[CHAT] No input heard.")
                continue

            try:
                user_text = recognizer.recognize_google(audio)
                print(f"[CHAT] You: {user_text}")
            except sr.UnknownValueError:
                print("[CHAT] Could not understand audio.")
                continue
            except sr.RequestError as e:
                error_msg = f"Speech recognition error: {e}"
                print(f"[CHAT] {error_msg}")
                speak(error_msg)
                continue

            # Get response from llama3.1
            try:
                print(f"[CHAT] Querying {model}...")
                set_avatar_emotion("thinking")
                with avatar_ai_activity():
                    response = client.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_text}
                        ],
                        stream=False,
                        options={"temperature": 0.1, "num_predict": 250}  # OPTIMIZED: Increased to allow longer responses
                    )
                reply = response['message']['content'].strip()
                print(f"[CHAT] SARAH: {reply}")
                set_avatar_emotion("happy")
                speak(reply)
                set_avatar_emotion("neutral")
            except Exception as e:
                error_msg = f"I encountered an error: {e}"
                print(f"[CHAT] {error_msg}")
                print(f"[CHAT] Make sure Ollama is running: ollama serve")
                speak("I'm having trouble right now. Please try again.")
        except Exception as e:
            print(f"[CHAT] Unexpected error: {e}")
            time.sleep(1)

    print("[CHAT] Simple Chat Mode exited.")

###############################################
# Chat Thread (freeform conversation mode)
###############################################
class ChatThread(threading.Thread):
    def __init__(self, stop_event: threading.Event, camera_thread: CameraThread):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.camera_thread = camera_thread
        self.camera_analyzer = CameraAnalyzer(camera_thread)
        self.recognizer = sr.Recognizer()
        # Optimize speech recognition for accuracy
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.phrase_threshold = 0.3
        try:
            sr_device_index = _preferred_sr_mic_device_index()
            if sr_device_index is not None:
                print(f"[CHAT] Using SpeechRecognition device_index={sr_device_index}")
            else:
                print("[CHAT] Using auto-detected microphone")
            self.microphone = create_sr_microphone(sample_rate=None, chunk_size=1024)
        except OSError as e:
            print("[CHAT] Microphone init failed:", e)
            print("[CHAT] Trying default microphone...")
            try:
                self.microphone = create_sr_microphone(sample_rate=None, chunk_size=1024)
            except Exception as e2:
                print("[CHAT] Default microphone also failed:", e2)
                self.microphone = None
        self.system_prompt = (
            "You are SARAH, a friendly robot assistant. The user can ask you questions or just chat with you. "
            "Provide helpful, conversational responses. Keep responses concise and natural (1-2 sentences usually). "
            "You are always friendly and helpful. If the user asks to describe what you see, you can analyze images from your camera."
        )

    def run(self):
        if not self.microphone:
            print("[CHAT] No microphone available; chat mode disabled.")
            return

        print("[CHAT] Chat mode started. Speak freely to chat with SARAH!")
        consecutive_open_failures = 0
        while not self.stop_event.is_set():
            try:
                try:
                    set_avatar_emotion("listening")
                    with safe_microphone_source(self.microphone) as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                        print("[CHAT] Listening for input...")
                        audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)
                    consecutive_open_failures = 0
                except sr.WaitTimeoutError:
                    time.sleep(0.1)
                    continue
                except Exception as e:
                    consecutive_open_failures += 1
                    print(f"[CHAT] Listen error: {e}")
                    if consecutive_open_failures >= 10:
                        print("[CHAT] [WARN] Repeated microphone open failures; disabling chat thread.")
                        return
                    time.sleep(0.5)
                    continue

                try:
                    user_text = self.recognizer.recognize_google(audio)
                    print(f"[CHAT] You said: '{user_text}'")
                except sr.UnknownValueError:
                    print("[CHAT] Could not understand audio. Please try again.")
                    time.sleep(0.2)
                    continue
                except sr.RequestError as e:
                    print(f"[CHAT] Speech recognition error: {e}")
                    time.sleep(0.5)
                    continue

                # Check if user is asking about what the camera sees
                user_lower = user_text.lower()
                if any(word in user_lower for word in ["see", "look", "view", "camera", "ahead", "front"]):
                    print("[CHAT] Vision query detected.")
                    set_avatar_emotion("thinking")
                    # Capture frame once and pass to analyze_scene to avoid duplicate encoding
                    img_b64 = self.camera_thread.get_frame_base64()
                    camera_analysis = self.camera_analyzer.analyze_scene(img_b64=img_b64)
                    print(f"[CHAT] Camera analysis: {camera_analysis}")
                    if img_b64:
                        try:
                            print("[CHAT] Analyzing camera view with vision model...")
                            # Enhance prompt with camera analysis
                            enhanced_prompt = user_text + f"\n[Scene] {camera_analysis.get('recommendation', '?')}"
                            
                            # Handle both remote and local AI modes
                            if AI_MODE == "remote":
                                if hasattr(client, 'chat'):
                                    with avatar_ai_activity():
                                        response = client.chat(
                                            model='llava',
                                            messages=[
                                                {'role': 'system', 'content': self.system_prompt},
                                                {'role': 'user', 'content': enhanced_prompt, 'images': [img_b64]}
                                            ],
                                            stream=False,
                                            options={"temperature": 0.05, "num_predict": 200}
                                        )
                                else:
                                    raise Exception("Remote server does not support vision analysis")
                            else:
                                # Local Ollama with llava
                                with avatar_ai_activity():
                                    response = client.chat(
                                        model='llava',
                                        messages=[
                                            {'role': 'system', 'content': self.system_prompt},
                                            {'role': 'user', 'content': enhanced_prompt, 'images': [img_b64]}
                                        ],
                                        stream=False,
                                        options={"temperature": 0.05, "num_predict": 200}
                                    )
                            reply = response['message']['content'].strip()
                            print(f"[CHAT] Response: {reply}")
                            set_avatar_emotion("happy")
                            speak(reply)
                            set_avatar_emotion("neutral")
                        except Exception as e:
                            print(f"[CHAT] Vision analysis error: {e}")
                            error_msg = "Sorry, I couldn't analyze the camera right now."
                            print(f"[CHAT] {error_msg}")
                            speak(error_msg)
                    else:
                        no_camera_msg = "I don't have a camera view available right now."
                        print(f"[CHAT] {no_camera_msg}")
                        speak(no_camera_msg)
                else:
                    # Regular chat with llama3.1 - uses new query_llama_and_speak
                    # Chat mode doesn't require activation word - TTS always responds
                    query_llama_and_speak(
                        prompt=user_text,
                        system_prompt=self.system_prompt,
                        speak_response=True,  # Automatically speak response
                        activation_detected=True  # Chat mode - no activation requirement
                    )
            except Exception as e:
                print(f"[CHAT] Unexpected error: {e}")
                time.sleep(1)

            time.sleep(0.5)

        print("[CHAT] Chat thread exiting.")


###############################################
# System Health Checks
###############################################
def check_tts_server():
    """Checks if Piper TTS is available and responsive."""
    global TTS_ENABLED
    print(f"[SYS] Checking Piper TTS...")
    try:
        # Piper TTS is used - no initialization needed, just test it
        print("[SYS] Testing Piper TTS...")
        speak("Activating")
        time.sleep(1.5)  # Wait for audio to finish
        
        TTS_ENABLED = True
        print(f"[SYS] Piper TTS is ready")
        return
    except Exception as e:
        print(f"[WARN] Piper TTS check failed: {e}")
        print(f"[HELP] Install with: pip install piper-tts")
        print(f"[HELP] Download voice model: piper --download en_US-amy-medium")
        import traceback
        traceback.print_exc()
    
    TTS_ENABLED = False
    print("[WARN] Voice output will be disabled. Voice commands will still work but no TTS responses.")
    print(f"[HELP] To debug, run: test_tts_server() from Python console")


def initialize_llava_model():
    """Check if Llava vision model is available."""
    print("[SYS] Checking Llava vision model availability...")
    
    # First, check if we're in remote mode and if remote server supports vision
    if AI_MODE == "remote":
        if not hasattr(client, 'chat'):
            print("[SYS] [WARNING]  Remote AI server doesn't support chat/vision")
            return False
        print("[SYS] [INFO] Using remote AI server for vision (Llava)")
        return True
    
    # For local mode, try to access llava model
    try:
        # Query Ollama to list available models
        import urllib.request
        import json
        
        try:
            # Try to check if llava is in the available models
            response = urllib.request.urlopen('http://localhost:11434/api/tags')
            models_data = json.loads(response.read().decode())
            available_models = [m.get('name', '') for m in models_data.get('models', [])]
            
            # Check if any llava model is available
            llava_available = any('llava' in model.lower() for model in available_models)
            
            if llava_available:
                print("[SYS] [OK] Llava vision model is available locally")
                return True
            else:
                print(f"[SYS] [WARNING]  Llava model not found in Ollama")
                print(f"[SYS] Available models: {', '.join(available_models[:3])}...")
                print(f"[SYS] To install Llava, run: ollama pull llava")
                return False
        except Exception as model_check_err:
            print(f"[SYS] Could not check Ollama models: {model_check_err}")
            print(f"[SYS] Vision features will be skipped")
            return False
            
    except Exception as e:
        print(f"[SYS] [WARNING]  Llava check failed: {e}")
        print(f"[SYS] Vision features will be unavailable")
        return False


def verify_system():
    """
    Run all system checks before starting.
    Provides guidance if services are missing.
    """
    print("\n" + "="*60)
    print("[SYS] System Verification")
    print("="*60)
    
    # Check audio system
    print("\n[CHECK 1/4] System Audio...")
    check_system_audio()
    
    # Check TTS
    print("\n[CHECK 2/4] TTS Engine...")
    check_tts_server()
    
    # Check Ollama
    print("\n[CHECK 3/4] Ollama Model...")
    if not verify_ollama_available():
        print("\n" + "!"*60)
        print("[ACTION REQUIRED] Ollama is not running!")
        print("!"*60)
        if IS_RASPBERRY_PI and AI_MODE == "remote":
            print("\nRemote mode (Pi → Windows Ollama) is selected.")
            print("On the Windows PC, start Ollama listening on the network:")
            print("  PowerShell:")
            print("    $env:OLLAMA_HOST='0.0.0.0:11434'")
            print("    ollama serve")
            print("\nThen (still on Windows) pull the model if needed:")
            print(f"  ollama pull {model}")
            print("\nFrom the Pi, test connectivity:")
            print(f"  ping {REMOTE_AI_SERVER_URL.replace('http://','').replace('https://','').split(':')[0]}")
            print(f"  curl {REMOTE_AI_SERVER_URL}/api/tags")
            print("\nIf it times out, check Windows Firewall inbound rule for TCP 11434.")
        else:
            print("\nTo start Ollama, open a new terminal and run:")
            print("  ollama serve")
            print("\nThen in another terminal, pull the model if needed:")
            print(f"  ollama pull {model}")
            print("\nAfter Ollama is running, restart this script.")
        print("="*60)
        return False
    
    # Check Llava vision model
    print("\n[CHECK 4/4] Vision Model (Llava)...")
    llava_available = initialize_llava_model()
    
    if not llava_available:
        print("[SYS] [WARNING]  Vision features will be disabled (camera analysis unavailable)")
        print("[SYS] Robot will operate without computer vision")
    
    print("\n" + "="*60)
    print("[SYS] [OK] Core systems ready. Starting robot...")
    print("="*60 + "\n")
    return True

###############################################
# Main
###############################################
def main():
    print_startup_banner()

    # Early diagnostics for common Pi setup issues
    check_pi_runtime_dependencies()

    # Optional: run GPIO backend diagnostics and exit.
    if os.environ.get("SARAH_GPIO_DIAG", "0").strip().lower() in ("1", "true", "yes"):
        gpio_backend_diagnostics()
        return
    
    # Initialize audio system FIRST (before any TTS output)
    # This maximizes USB speaker volume
    if IS_RASPBERRY_PI:
        print("[INIT] Initializing USB audio system on Raspberry Pi...")
        check_system_audio()  # Maximize volume and configure audio
        time.sleep(0.5)  # Give audio system time to settle
        print("[INIT] [OK] USB audio system initialized")
    
    # Enable GPIO only on Raspberry Pi with motors connected
    # On Windows/development: GPIO will be automatically disabled by Drive class
    use_gpio = IS_RASPBERRY_PI  # Auto-detect and enable on RPi5
    try:
        drive = Drive(use_gpio=use_gpio)
    except RuntimeError as e:
        # If we're on a Pi and the only workable path is native+root (/dev/gpiomem missing),
        # automatically relaunch with sudo to get access to /dev/mem.
        msg = str(e)
        # Default to auto-sudo ON for Raspberry Pi (set SARAH_AUTO_SUDO=0 to disable)
        auto_sudo = os.environ.get("SARAH_AUTO_SUDO", "1" if IS_RASPBERRY_PI else "0").strip().lower() in ("1", "true", "yes")
        already = os.environ.get("SARAH_ALREADY_SUDO", "0").strip().lower() in ("1", "true", "yes")
        is_root = (os.geteuid() == 0) if hasattr(os, "geteuid") else False

        # Check if this is a permissions/module issue that sudo could fix
        needs_sudo = (
            "/dev/gpiomem=False" in msg 
            or "unable to open /dev/gpiomem" in msg
            or "No module named 'lgpio'" in msg
        )

        if (
            IS_RASPBERRY_PI
            and use_gpio
            and auto_sudo
            and (not already)
            and (not is_root)
            and needs_sudo
        ):
            print("[SYS] GPIO init failed - attempting auto-relaunch with sudo (set SARAH_AUTO_SUDO=0 to disable)")
            print(f"[SYS] Command: sudo -E {sys.executable} {' '.join(sys.argv)}")
            try:
                os.environ["SARAH_ALREADY_SUDO"] = "1"
                os.execvp("sudo", ["sudo", "-E", sys.executable] + sys.argv)
            except Exception as ex:
                print(f"[SYS] Sudo relaunch failed: {type(ex).__name__}: {ex}")
                print("[SYS] Please run manually: sudo -E python /home/liamdusanic/Documents/sarah_pi.py")
                raise

        raise
    
    # Initialize sensors with same pin_factory as motors to avoid backend mismatch
    pin_factory_for_sensors = getattr(drive, 'pin_factory', None)
    sensors = init_ultrasonic_sensors(use_gpio=use_gpio, pin_factory=pin_factory_for_sensors)
    try:
        set_active_ultrasonic_sensors(sensors)
    except Exception:
        pass
    
    # TEST SENSORS: Verify they work before entering main loop
    if use_gpio and sensors:
        print("\n" + "="*70)
        print("[SENSOR-TEST] Testing ultrasonic sensors...")
        print("="*70)
        
        # Test each sensor 3 times to verify reliability
        sensor_results = []
        for i, sensor in enumerate(sensors):
            label = ['left', 'center', 'right'][i] if i < 3 else f'sensor{i}'
            readings = []
            for attempt in range(3):
                reading = sensor.read_distance_cm()
                if reading is not None and reading > 0:
                    readings.append(reading)
                time.sleep(0.1)  # Small delay between readings
            
            success_rate = len(readings) / 3.0
            if readings:
                avg = sum(readings) / len(readings)
                print(f"[ULTRA-READ] Sensor {i} ({label}): {avg:.1f}cm (success: {len(readings)}/3 tests)")
                sensor_results.append(True)
            else:
                print(f"[ULTRA-READ] Sensor {i} ({label}): TIMEOUT/ERROR (check wiring on pins {sensor.trigger_pin}/{sensor.echo_pin})")
                sensor_results.append(False)
        
        valid_count = sum(sensor_results)
        print(f"[SENSOR-TEST] Result: {valid_count}/{len(sensors)} sensors operational")
        
        if valid_count == 0:
            print("[SENSOR-TEST] ⚠️  WARNING: NO SENSORS ARE WORKING!")
            print("[SENSOR-TEST] Robot will be BLIND in autonomous mode and may collide!")
            print("[SENSOR-TEST] Check:")
            print("[SENSOR-TEST]   1. Sensor power connections (VCC/GND)")
            print("[SENSOR-TEST]   2. Wiring: " + str(ULTRASONIC_PINS))
            print("[SENSOR-TEST]   3. Run with sudo if needed")
        elif valid_count < len(sensors):
            print(f"[SENSOR-TEST] ⚠️  WARNING: Only {valid_count}/{len(sensors)} sensors working!")
            print("[SENSOR-TEST] Robot CAN operate but with reduced obstacle detection.")
            print("[SENSOR-TEST] Failed sensors:")
            for i, (working, sensor) in enumerate(zip(sensor_results, sensors)):
                if not working:
                    label = ['left', 'center', 'right'][i] if i < 3 else f'sensor{i}'
                    print(f"[SENSOR-TEST]   - {label}: GPIO{sensor.trigger_pin}(trig) / GPIO{sensor.echo_pin}(echo)")
                    print(f"[SENSOR-TEST]     • Check VCC connected to 5V pin")
                    print(f"[SENSOR-TEST]     • Check GND connected to ground")
                    print(f"[SENSOR-TEST]     • Verify trigger wire to GPIO{sensor.trigger_pin}")
                    print(f"[SENSOR-TEST]     • Verify echo wire to GPIO{sensor.echo_pin}")
                    print(f"[SENSOR-TEST]     • Try swapping with working sensor to test if sensor is faulty")
        else:
            print(f"[SENSOR-TEST] ✓ All sensors ready for autonomous mode")
        print("="*70 + "\n")
    
    stop_event = threading.Event()

    # Detect controller BEFORE starting the avatar.
    # The avatar uses pygame in a background thread; calling pygame.quit() after
    # starting it can tear down the avatar display.
    controller_present = False
    if pygame:
        try:
            # Init only the joystick subsystem. Calling pygame.init() can
            # implicitly initialize SDL video with a forwarded/remote DISPLAY,
            # which then makes it harder for the avatar to retarget the local
            # HDMI display when launched via SSH.
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                controller_present = True
        except Exception:
            controller_present = False
        finally:
            try:
                pygame.joystick.quit()
            except Exception:
                pass

    # Seed global connection state. If a controller is present at startup, we'll
    # mark it connected; ControllerThread will keep this updated afterwards.
    try:
        set_controller_connected(bool(controller_present))
    except Exception:
        pass

    # Start the Pi-screen avatar (never starts on Windows).
    global _AVATAR
    if AVATAR_ENABLED and pygame:
        try:
            _AVATAR = AvatarDisplay()
            _AVATAR.start()
            Logger.log("AVATAR", "Avatar display started", "INFO")
        except Exception as e:
            # Try again without fullscreen if first attempt failed
            try:
                sys.modules.pop('pygame', None)  # Force reimport
                import pygame as pg
                globals()['pygame'] = pg
                old_fullscreen = globals().get('AVATAR_FULLSCREEN')
                globals()['AVATAR_FULLSCREEN'] = False
                _AVATAR = AvatarDisplay()
                _AVATAR.start()
                Logger.log("AVATAR", "Avatar display started (windowed mode)", "INFO")
                globals()['AVATAR_FULLSCREEN'] = old_fullscreen
            except Exception as e2:
                _AVATAR = None
                Logger.log("AVATAR", f"Avatar disabled after 2 attempts: {type(e).__name__}: {e} | {type(e2).__name__}: {e2}", "ERROR")
                Logger.log("AVATAR", "Try: export DISPLAY=:0 before running script", "ERROR")

    # Optional: console-triggered motor wiring verification (exits after running)
    if os.environ.get("SARAH_MOTOR_SMOKE_TEST", "0").strip().lower() in ("1", "true", "yes"):
        run_motor_smoke_test(drive)
        return

    controller_thread = None
    autonomous_thread = None
    voice_thread = None
    camera_thread = None
    chat_thread = None
    game_movement_thread = None
    dance_thread = None

    def shutdown(sig=None, frame=None):
        Logger.log("SYS", "Shutdown signal received. Cleaning up...", "WARN")
        stop_event.set()
        cleanup_temp_files()  # Clean up temp audio files

        if _AVATAR is not None:
            try:
                _AVATAR.stop()
            except Exception:
                pass
        if controller_thread and controller_thread.is_alive():
            controller_thread.join(timeout=1.0)
        if autonomous_thread and autonomous_thread.is_alive():
            autonomous_thread.join(timeout=1.0)
        if voice_thread and voice_thread.is_alive():
            voice_thread.join(timeout=1.0)
        if chat_thread and chat_thread.is_alive():
            chat_thread.join(timeout=1.0)
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=1.0)
        # Clean up GPIO (motors and sensors)
        if drive:
            drive.cleanup()
        if sensors:
            for sensor in sensors:
                try:
                    sensor.cleanup()
                except Exception:
                    pass
        Logger.log("SYS", "Cleanup complete", "SUCCESS")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Memory check and optimization (for 4GB RPi5 constraint)
    print_memory_status("Initial")
    if get_memory_status()['total_gb'] < 3.5:
        Logger.log("SYS", "WARNING: System has less than 4GB RAM!", "WARN")
        Logger.log("SYS", "Recommend using lightweight model: ollama pull phi", "WARN")
    optimize_memory()  # Clear any startup garbage

    # Piper TTS is ready - no initialization needed
    Logger.log("SYS", "TTS: Piper (ready)", "INFO")
    
    # Perform comprehensive system checks before starting threads
    if not verify_system():
        Logger.log("SYS", "System verification failed. Continuing with limited functionality...", "WARN")
        speak("System verification incomplete. Some features may be unavailable.")
    
    # Warm up model to avoid first-query delay.
    # Only do this when Ollama is reachable; otherwise avoid a long hang on startup.
    if verify_ollama_available():
        warmup_model()

    # Start camera thread first (non-critical if it fails)
    try:
        camera_thread = CameraThread(stop_event)
        camera_thread.start()
        time.sleep(1)  # Give camera time to initialize
    except Exception as e:
        print(f"[SYS] Failed to start camera thread: {e}")
        print(f"[SYS] Creating dummy camera thread for compatibility...")
        # Create a minimal camera thread that will gracefully handle missing camera
        camera_thread = CameraThread(stop_event)
        camera_thread.camera_enabled = False
        camera_thread.start()
        speak("Camera not available. I'll operate without vision.")

    # If controller is present, offer mode selection
    if controller_present:
        try:
            speak("Controller detected. Say 'autonomous' for autonomous mode, 'manual' for manual RC, or 'chat' to talk with me.")
            response = listen_for_response(duration=MODE_SELECT_LISTEN_SECONDS)
        except Exception as e:
            print(f"[SYS] Error during mode selection: {e}")
            response = ""

        if "chat" in response:
            print("[MAIN] User selected CHAT mode.")
            speak("Starting chat mode. Speak freely to chat with me!")
            set_current_mode("chat", source="startup")
            run_simple_chat_mode(stop_event, camera_thread)
        else:
            # Start both controller + autonomous threads so voice switching always works.
            controller_thread = ControllerThread(drive, stop_event)
            controller_thread.start()
            autonomous_thread = AutonomousThread(drive, stop_event, camera_thread, sensors=sensors)
            autonomous_thread.start()
            voice_thread = VoiceListenerThread(drive, stop_event, camera_thread)
            voice_thread.start()
            dance_thread = DanceThread(drive, stop_event)
            dance_thread.start()
            game_movement_thread = GameMovementThread(drive, stop_event)
            game_movement_thread.start()

            if ("autonomous" in response) or ("unrestricted" in response):
                print("[MAIN] User selected AUTONOMOUS mode.")
                speak("Starting autonomous mode.")
                set_current_mode("autonomous", source="startup")
            else:
                # Attempt manual mode (will fall back to autonomous if no controller).
                target_mode = "manual" if is_controller_connected() else "autonomous"
                if target_mode == "manual":
                    print("[MAIN] Defaulting to manual RC mode.")
                    speak("Starting manual control mode.")
                else:
                    print("[MAIN] Controller not ready. Starting autonomous mode.")
                    speak("Controller not ready. Starting autonomous mode.")
                set_current_mode(target_mode, source="startup")
    else:
        try:
            speak("No controller found. Say 'chat' to talk with me, or 'autonomous' for autonomous mode.")
            response = listen_for_response(duration=MODE_SELECT_LISTEN_SECONDS)
        except Exception as e:
            print(f"[SYS] Error during mode selection: {e}")
            response = ""

        if "chat" in response:
            print("[MAIN] User selected CHAT mode.")
            speak("Starting chat mode. Speak freely to chat with me!")
            set_current_mode("chat", source="startup")
            run_simple_chat_mode(stop_event, camera_thread)
        else:
            print("[MAIN] Starting AUTONOMOUS mode.")
            speak("Starting autonomous mode.")
            set_current_mode("autonomous", source="startup")
            autonomous_thread = AutonomousThread(drive, stop_event, camera_thread, sensors=sensors)
            autonomous_thread.start()
            voice_thread = VoiceListenerThread(drive, stop_event, camera_thread)
            voice_thread.start()
            dance_thread = DanceThread(drive, stop_event)
            dance_thread.start()
            game_movement_thread = GameMovementThread(drive, stop_event)
            game_movement_thread.start()

    # Main loop: keep alive and process selected mode
    try:
        gc_counter = 0  # Counter for periodic garbage collection
        memory_check_counter = 0  # Counter for memory status checks every 30 seconds
        battery_check_counter = 0  # Counter for battery checks
        last_battery_level = "unknown"
        last_battery_announce_ts = 0.0
        while True:
            time.sleep(0.5)  # Balanced sleep to give camera thread CPU time
            
            # Periodic garbage collection (every 5 seconds)
            gc_counter += 1
            memory_check_counter += 1
            if gc_counter >= 10:  # 10 × 0.5s = 5 seconds
                gc_counter = 0
                optimize_memory()
            
            # Check memory status every 30 seconds (60 × 0.5s = 30 seconds)
            if memory_check_counter >= 60:
                memory_check_counter = 0
                print_memory_status("Periodic check")
                if check_memory_critical():
                    print("[SYS] [WARNING]  Memory critical - consider freeing resources")

            # Check battery periodically (default: every 30 seconds)
            battery_check_counter += 1
            try:
                bat_period_s = float(os.getenv("SARAH_BATTERY_CHECK_SECONDS", "30") or "30")
            except Exception:
                bat_period_s = 30.0
            ticks = max(1, int(round(bat_period_s / 0.5)))
            if battery_check_counter >= ticks:
                battery_check_counter = 0
                percent = _read_battery_percent()
                level = _battery_level_from_percent(percent)

                # Update avatar mood overlay
                if level == "critical":
                    set_avatar_battery_mood("exhausted")
                elif level == "low":
                    set_avatar_battery_mood("tired")
                elif level == "ok":
                    set_avatar_battery_mood(None)

                # Notify (rate-limited) when entering low/critical
                now = time.time()
                try:
                    remind_s = float(os.getenv("SARAH_BATTERY_REMIND_SECONDS", "300") or "300")
                except Exception:
                    remind_s = 300.0

                entered_worse = (last_battery_level in ("unknown", "ok") and level in ("low", "critical")) or (last_battery_level == "low" and level == "critical")
                should_remind = (level in ("low", "critical")) and ((now - last_battery_announce_ts) >= max(30.0, remind_s))

                if entered_worse or should_remind:
                    if level == "critical":
                        msg = "My battery is critically low. Please recharge me soon."
                    elif level == "low":
                        msg = "My battery is getting low. Please plan to recharge me."
                    else:
                        msg = ""
                    if msg:
                        try:
                            if percent is not None:
                                msg = f"{msg} Battery at {int(round(percent))} percent."
                        except Exception:
                            pass
                        try:
                            speak(msg)
                        except Exception:
                            pass
                        last_battery_announce_ts = now

                last_battery_level = level
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
