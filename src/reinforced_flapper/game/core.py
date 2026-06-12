"""Shared runtime context for game entities: window geometry and services."""

import random

import pygame

from reinforced_flapper.game.assets import Images, SoundsLike


class Window:
    """Game window geometry.

    Attributes:
        width: Window width in pixels.
        height: Window height in pixels.
        ratio: Width to height ratio.
        viewport_height: Height of the playable area above the floor.
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize the window geometry.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
        """
        self.width = width
        self.height = height
        self.ratio = width / height
        self.viewport_width = width
        self.viewport_height = height * 0.79
        # Short aliases used by entities.
        self.w = width
        self.h = height
        self.vw = self.viewport_width
        self.vh = self.viewport_height

    @property
    def viewport_ratio(self) -> float:
        """Width to height ratio of the playable area."""
        return self.viewport_width / self.viewport_height


class GameConfig:
    """Runtime context shared by all game entities.

    The screen and clock are None when the game runs headless. Entities only
    touch them inside render paths, which are never called headless.

    Attributes:
        screen: Render target, or None when headless.
        clock: Pygame clock for FPS capping, or None when headless.
        fps: Target frames per second when rendering.
        window: Window geometry.
        images: Loaded sprite images.
        sounds: Sound effects (a silent implementation when headless).
        rng: Random source for game randomness (pipe gap positions).
        debug: Draw debug overlays when rendering.
    """

    def __init__(
        self,
        *,
        window: Window,
        images: Images,
        sounds: SoundsLike,
        rng: random.Random,
        screen: pygame.Surface | None = None,
        clock: pygame.time.Clock | None = None,
        fps: int = 30,
        debug: bool = False,
    ) -> None:
        """Initialize the game context.

        Args:
            window: Window geometry.
            images: Loaded sprite images.
            sounds: Sound effects implementation.
            rng: Random source for game randomness.
            screen: Render target, or None when headless.
            clock: Pygame clock, or None when headless.
            fps: Target frames per second when rendering.
            debug: Draw debug overlays when rendering.
        """
        self.window = window
        self.images = images
        self.sounds = sounds
        self.rng = rng
        self.screen = screen
        self.clock = clock
        self.fps = fps
        self.debug = debug

    def tick(self) -> None:
        """Advance the render clock, capping at the configured FPS."""
        if self.clock is not None:
            self.clock.tick(self.fps)

    def require_screen(self) -> pygame.Surface:
        """Return the screen, which must exist on render paths.

        Returns:
            The render target.

        Raises:
            RuntimeError: If called while running headless.
        """
        if self.screen is None:
            raise RuntimeError("render path reached without a screen")
        return self.screen
