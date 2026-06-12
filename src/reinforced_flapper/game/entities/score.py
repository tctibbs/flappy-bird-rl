"""Score entity."""

from typing import TYPE_CHECKING

import pygame

from reinforced_flapper.game.entities.entity import Entity

if TYPE_CHECKING:
    from reinforced_flapper.game.core import GameConfig


class Score(Entity):
    """Score tracking and display.

    Attributes:
        score: Current score (pipes passed).
    """

    def __init__(self, config: "GameConfig") -> None:
        """Initialize the score.

        Args:
            config: Shared game context.
        """
        super().__init__(config)
        self.y = self.config.window.height * 0.1
        self.score = 0

    def reset(self) -> None:
        """Reset the score to zero."""
        self.score = 0

    def add(self) -> None:
        """Add one to the score."""
        self.score += 1
        self.config.sounds.point.play()

    @property
    def rect(self) -> pygame.Rect:
        """Bounding rect of the rendered score digits."""
        score_digits = [int(x) for x in list(str(self.score))]
        images = [self.config.images.numbers[digit] for digit in score_digits]
        w = sum(image.get_width() for image in images)
        x = (self.config.window.width - w) / 2
        h = max(image.get_height() for image in images)
        return pygame.Rect(x, self.y, w, h)

    def tick(self) -> None:
        """Update the score (no-op)."""

    def render(self) -> None:
        """Draw the score centered near the top of the screen."""
        score_digits = [int(x) for x in list(str(self.score))]
        images = [self.config.images.numbers[digit] for digit in score_digits]
        digits_width = sum(image.get_width() for image in images)
        x_offset = (self.config.window.width - digits_width) / 2

        screen = self.config.require_screen()
        for image in images:
            screen.blit(image, (x_offset, self.y))
            x_offset += image.get_width()
