#!/usr/bin/env python3
"""Compatibility entry point for the IDE's original script location.

Keep the full repository together; driving and companion features live in sarah/.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sarah.app import main

if __name__ == "__main__":
    raise SystemExit(main())
