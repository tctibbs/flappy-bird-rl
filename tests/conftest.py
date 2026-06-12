"""Test fixtures and headless-safe defaults."""

import os

# Belt and braces: nothing in the headless paths should touch SDL devices,
# but if something regresses, fail softly to dummy drivers instead of CI.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
