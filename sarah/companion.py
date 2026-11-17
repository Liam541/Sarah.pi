#!/usr/bin/env python3
"""Retained avatar, voice, speech, chat, dance, game and battery features.
Driving is owned exclusively by sarah.motion through scoped adapters.
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
from .hardware import CameraThread
from .controls import execute_timed
_MOTION = None
_APP_STOP = None

# Human-readable build stamp to help confirm which script copy is running on the Pi.
SARAH_BUILD = "2026-11-17"

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
    raise ImportError("Companion chat requires the ollama package")

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
                pygame.display.quit()
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
# GPIO is initialized only by sarah.hardware, never by companion features.

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

# Companion modes delegate every physical movement to the central controller.
DEFAULT_SPEED = 40
LLAMA_CACHE_SIZE = 20
try:
    LLAMA_MAX_OUTPUT_TOKENS_CHAT = int(os.getenv("SARAH_CHAT_MAX_OUTPUT_TOKENS", "250"))
except ValueError:
    LLAMA_MAX_OUTPUT_TOKENS_CHAT = 250
CURRENT_MODE = "manual"
_CONTROLLER_LOCK = threading.RLock()
_CONTROLLER_CONNECTED = False
_MODE_LOCK = threading.RLock()
_MODE_COND = threading.Condition(_MODE_LOCK)
_MODE_SEQ = 0


def set_controller_connected(is_connected):
    global _CONTROLLER_CONNECTED
    with _CONTROLLER_LOCK:
        _CONTROLLER_CONNECTED = bool(is_connected)


def get_current_mode():
    with _MODE_LOCK:
        return CURRENT_MODE


def set_current_mode(mode, source=""):
    global CURRENT_MODE, _MODE_SEQ
    if mode not in ("manual", "autonomous", "chat", "dance", "game"):
        return
    with _MODE_COND:
        if _MOTION is not None:
            _MOTION.set_mode(mode)
        CURRENT_MODE = mode
        _MODE_SEQ += 1
        _MODE_COND.notify_all()
    print(f"[MODE] {mode.upper()} ({source})")


def request_return_to_start(source=""):
    set_current_mode("manual", source=source)
    speak("Return to start needs localization or wheel encoders. I have stopped for manual control.")


def request_reset_map_memory(source=""):
    set_current_mode("manual", source=source)
    speak("Navigation stopped. Select autonomous again to begin a fresh exploration session.")


def request_explore(source=""):
    set_current_mode("autonomous", source=source)


def request_dance(source=""):
    set_current_mode("dance", source=source)


def request_game(source=""):
    set_current_mode("game", source=source)


def exit_game_mode(source=""):
    set_current_mode("manual", source=source)


def exit_dance_mode(source=""):
    set_current_mode("manual", source=source)


# Optional conversational AI configuration, independent of autonomous planning.
AI_MODE = "local"
REMOTE_AI_SERVER_URL = os.getenv("SARAH_REMOTE_OLLAMA_URL", "http://127.0.0.1:11434")

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
model = os.getenv("SARAH_CHAT_MODEL", "llama3.2:1b")  # Model name - MUST be defined before verify_ollama_available

# Auto-start Ollama service on Windows

# Auto-start AI server if in remote mode and Windows

# Initialize AI client (local Ollama or remote HTTP server)
client = None
CLIENT_TYPE = None  # Track which client type we're using

client = ollama.Client(host=OLLAMA_SERVER_URL, timeout=45)
CLIENT_TYPE = "Ollama chat client"

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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


###############################################
# Model / Ollama Command Distribution
###############################################






def parse_simple_voice_command(text):
    clean = re.sub(r"^sarah[ ,:]*", "", text.lower().strip()).rstrip(".!?")
    match = re.fullmatch(r"(?:go |move |turn )?(forward|backward|back|left|right|stop)(?: for (\d+(?:\.\d+)?) seconds?)?", clean)
    if not match:
        return None, False
    action = {"back": "BACKWARD"}.get(match[1], match[1].upper())
    duration = min(2.0, float(match[2])) if match[2] else 0.4
    return {"command": action, "duration": duration, "speed": 40, "speak": ""}, True


###############################################
# Llama3.1 Query with Automatic TTS
###############################################
def query_llama_and_speak(prompt: str, system_prompt: str = None, speak_response: bool = True, activation_detected: bool = False):
    """
    Query Llama3.1 directly and automatically speak the response via TTS.
    OPTIMIZED: Uses caching, timeout, and output limits for faster responses.
    TTS only responds if activation_detected=True (after 'Sarah' is said) or if TTS_NEEDS_ACTIVATION is False.
    
    Args:
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
    if drive is None or _APP_STOP is None:
        return
    token = drive.motion.token()
    if token[0] != drive.mode:
        return
    set_emotion_for_movement(cmd.get("command", "STOP"))
    # Motion completes (or is canceled) before potentially blocking speech.
    execute_timed(drive, cmd, _APP_STOP, token=token)
    if not silent and cmd.get("speak"):
        speak(cmd["speak"])


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

            # If the song finished, return to manual.
            try:
                auto_return = os.getenv('SARAH_DANCE_END_ON_SONG_FINISH', '1').strip().lower() not in ('0', 'false', 'no', 'off')
            except Exception:
                auto_return = True
            try:
                if auto_return and self._audio_proc is not None and (self._audio_proc.poll() is not None):
                    print("[DANCE] Song finished - returning to manual")
                    _stop_process(self._audio_proc)
                    self._audio_proc = None
                    # End the performance in stopped manual mode.
                    try:
                        exit_dance_mode(source='dance_song_end')
                    except Exception:
                        set_current_mode('manual', source='dance_song_end')
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
        self.system_prompt = (
            "You are SARAH, a friendly robot assistant. Reply concisely for speech. "
            "Movement and mode changes are handled by explicit local commands. "
            "Do not claim that you moved, mapped an area, or changed a mode. "
            "Use plain conversational text, not movement-command JSON."
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
                                print('Install FLAC with: sudo apt install flac')
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
                activation = (ACTIVATION_WORD.lower() in text_lower) or get_current_mode() == "chat"

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
        clean = re.sub(r"^sarah[ ,:]*", "", text.lower().strip()).rstrip(".!?")
        try:
            if clean in ("stop", "emergency stop", "stop moving"):
                self.drive.motion.emergency_stop("Voice stop")
                set_current_mode("manual", source="voice_stop")
                speak("Stopped. Say Sarah manual or press Start to reset.")
                return
            if clean in ("manual", "manual mode", "remote control", "resume manual"):
                self.drive.motion.resume_manual()
                set_current_mode("manual", source="voice")
                speak("Manual control. Hold the left bumper to drive.")
                return
            modes = {"autonomous": "autonomous", "autonomous mode": "autonomous",
                     "explore": "autonomous", "start exploring": "autonomous",
                     "dance": "dance", "dance mode": "dance", "start dancing": "dance",
                     "game": "game", "game mode": "game", "activate game mode": "game",
                     "chat": "chat", "chat mode": "chat"}
            if clean in modes:
                if self.drive.motion.estopped:
                    speak("Emergency stop is active. Select manual to reset it first.")
                    return
                set_current_mode(modes[clean], source="voice")
                speak(f"Mode is now {modes[clean]}.")
                return
            if clean in ("stop dancing", "stop dance", "end dance", "stop game", "exit game", "end game mode", "exit chat"):
                set_current_mode("manual", source="voice")
                return
            if clean in ("go home", "return home", "return to start"):
                request_return_to_start(source="voice")
                return
            if clean in ("reset map", "clear map", "reset navigation memory"):
                request_reset_map_memory(source="voice")
                return
            cmd, matched = parse_simple_voice_command(text)
            if matched:
                # A movement request explicitly takes control away from autonomy.
                set_current_mode("manual", source="voice_movement")
                execute_command(self.drive, cmd)
                return
            if any(p in clean for p in ("what do you see", "what can you see", "describe the scene", "what is ahead")):
                image = self.camera_thread.get_frame_base64()
                if image:
                    with avatar_ai_activity():
                        result = client.chat(model=os.getenv("SARAH_CHAT_VISION_MODEL", "moondream"),
                                             messages=[{"role": "user", "content": clean, "images": [image]}])
                    speak(result['message']['content'])
                else:
                    speak("My camera is unavailable.")
                return
            query_llama_and_speak(text, system_prompt=self.system_prompt, speak_response=True, activation_detected=True)
        except Exception as exc:
            print(f"[VOICE] {exc}")


###############################################
# Controller thread (if joystick present)
###############################################


###############################################
# Voice helper for one-off commands
###############################################
###############################################
# Simple Chat Mode (voice in, voice out)
###############################################

###############################################
# Chat Thread (freeform conversation mode)
###############################################


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





###############################################
# Main
###############################################


