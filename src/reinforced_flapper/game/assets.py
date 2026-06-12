"""Sprite and sound loading, usable with or without a display.

Sprites in this repo are 8-bit palettized PNGs whose transparency is stored
as a colorkey. The original code relied on Surface.convert_alpha(), which
requires an initialized display. Blitting the loaded surface onto a fresh
SRCALPHA surface produces an identical per-pixel alpha result (verified
against convert_alpha for every sprite class) without needing a display,
which is what keeps headless simulation exact.
"""

from pathlib import Path
import random
import sys
from typing import Protocol

import pygame

from reinforced_flapper.game.constants import BACKGROUNDS, PIPES, PLAYERS

# Repo-root assets directory. This project runs from a checkout (uv run),
# so the path is anchored to the source tree.
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

if not pygame.image.get_extended():
    raise ImportError(
        "This pygame build lacks extended image support and cannot load PNG "
        "sprites. Use a Python version covered by requires-python (see "
        ".python-version), where pygame wheels ship with SDL_image."
    )


def load_sprite(name: str, *, convert: bool) -> pygame.Surface:
    """Load a sprite with per-pixel alpha.

    Args:
        name: Path relative to the assets directory.
        convert: Use display-dependent convert_alpha (faster blits when a
            display exists). When False, convert via an SRCALPHA surface,
            which needs no display and yields identical alpha.

    Returns:
        Surface with per-pixel alpha.
    """
    raw = pygame.image.load(str(ASSETS_DIR / name))
    if convert:
        return raw.convert_alpha()
    surface = pygame.Surface(raw.get_size(), pygame.SRCALPHA, 32)
    surface.blit(raw, (0, 0))
    return surface


class Images:
    """Game sprite images.

    Attributes:
        numbers: Digit sprites for the score display.
        base: Floor sprite.
        background: Background sprite.
        player: Three flap-position sprites for the bird.
        pipe: Upper (flipped) and lower pipe sprites.
    """

    def __init__(
        self,
        *,
        convert: bool,
        rng: random.Random | None = None,
        player_idx: int = 2,
        background_idx: int = 0,
        pipe_idx: int = 0,
    ) -> None:
        """Load all sprites.

        Args:
            convert: Use display-dependent conversion (requires a display).
            rng: When given, sprite variants are sampled from it and the
                index arguments are ignored.
            player_idx: Bird variant (0 red, 1 blue, 2 yellow).
            background_idx: Background variant (0 day, 1 night).
            pipe_idx: Pipe variant (0 green, 1 red).
        """
        if rng is not None:
            player_idx = rng.randrange(len(PLAYERS))
            background_idx = rng.randrange(len(BACKGROUNDS))
            pipe_idx = rng.randrange(len(PIPES))

        self.numbers = [
            load_sprite(f"sprites/{num}.png", convert=convert) for num in range(10)
        ]
        self.base = load_sprite("sprites/base.png", convert=convert)
        self.background = load_sprite(BACKGROUNDS[background_idx], convert=convert)
        self.player = tuple(
            load_sprite(name, convert=convert) for name in PLAYERS[player_idx]
        )
        lower_pipe = load_sprite(PIPES[pipe_idx], convert=convert)
        self.pipe = (pygame.transform.flip(lower_pipe, False, True), lower_pipe)


class SoundLike(Protocol):
    """Anything that can be played as a sound effect."""

    def play(self) -> object:
        """Play the sound."""
        ...


class SoundsLike(Protocol):
    """The set of sound effects the game uses."""

    die: SoundLike
    hit: SoundLike
    point: SoundLike
    swoosh: SoundLike
    wing: SoundLike


class _NullSound:
    """A sound that does nothing when played."""

    def play(self) -> None:
        """Do nothing."""


class NullSounds:
    """Silent sound set for headless and offscreen runs."""

    die: SoundLike
    hit: SoundLike
    point: SoundLike
    swoosh: SoundLike
    wing: SoundLike

    def __init__(self) -> None:
        """Initialize all effects as no-ops."""
        self.die = _NullSound()
        self.hit = _NullSound()
        self.point = _NullSound()
        self.swoosh = _NullSound()
        self.wing = _NullSound()


class Sounds:
    """Real sound effects backed by the pygame mixer.

    Requires pygame.mixer to be initialized.
    """

    die: SoundLike
    hit: SoundLike
    point: SoundLike
    swoosh: SoundLike
    wing: SoundLike

    def __init__(self) -> None:
        """Load all sound files."""
        ext = "wav" if "win" in sys.platform else "ogg"
        self.die = pygame.mixer.Sound(str(ASSETS_DIR / f"audio/die.{ext}"))
        self.hit = pygame.mixer.Sound(str(ASSETS_DIR / f"audio/hit.{ext}"))
        self.point = pygame.mixer.Sound(str(ASSETS_DIR / f"audio/point.{ext}"))
        self.swoosh = pygame.mixer.Sound(str(ASSETS_DIR / f"audio/swoosh.{ext}"))
        self.wing = pygame.mixer.Sound(str(ASSETS_DIR / f"audio/wing.{ext}"))
