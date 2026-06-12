"""Base class for game entities."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pygame

from reinforced_flapper.game.utils import get_hit_mask, pixel_collision

if TYPE_CHECKING:
    from reinforced_flapper.game.core import GameConfig


class Entity(ABC):
    """Game entity with position, image, and pixel-perfect collision.

    Attributes:
        config: Shared game context.
        image: Image of the entity, or None for imageless entities.
        x: X coordinate.
        y: Y coordinate.
        w: Width in pixels.
        h: Height in pixels.
        hit_mask: Per-pixel opacity mask used for collision.
    """

    def __init__(
        self,
        config: "GameConfig",
        image: pygame.Surface | None = None,
        x: float = 0,
        y: float = 0,
        w: int | None = None,
        h: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the entity with context, image, and position.

        Args:
            config: Shared game context.
            image: Image of the entity.
            x: X coordinate.
            y: Y coordinate.
            w: Width override; image is scaled when given.
            h: Height override; image is scaled when given.
            **kwargs: Extra attributes set on the instance.
        """
        self.config = config
        self.x = x
        self.y = y
        if image is not None and (w is not None or h is not None):
            width = w if w is not None else config.window.ratio * (h or 0)
            height = h if h is not None else (w or 0) / config.window.ratio
            self.w: float = width
            self.h: float = height
            self.image: pygame.Surface | None = pygame.transform.scale(
                image, (width, height)
            )
        else:
            self.image = image
            self.w = image.get_width() if image else 0
            self.h = image.get_height() if image else 0

        self.hit_mask = get_hit_mask(image) if image else None
        self.__dict__.update(kwargs)

    def update_image(
        self, image: pygame.Surface, w: int | None = None, h: int | None = None
    ) -> None:
        """Replace the entity's image and refresh its hit mask.

        Args:
            image: The new image.
            w: Width override.
            h: Height override.
        """
        self.image = image
        self.hit_mask = get_hit_mask(image)
        self.w = w or (image.get_width() if image else 0)
        self.h = h or (image.get_height() if image else 0)

    @property
    def cx(self) -> float:
        """Center x coordinate."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """Center y coordinate."""
        return self.y + self.h / 2

    @property
    def rect(self) -> pygame.Rect:
        """Bounding rect."""
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def collide(self, other: "Entity") -> bool:
        """Check pixel-perfect collision with another entity.

        Args:
            other: The entity to test against.

        Returns:
            True if opaque pixels overlap (or rects, when masks are absent).
        """
        if not self.hit_mask or not other.hit_mask:
            return self.rect.colliderect(other.rect)
        return pixel_collision(self.rect, other.rect, self.hit_mask, other.hit_mask)

    @abstractmethod
    def tick(self) -> None:
        """Update the entity for one frame."""

    def render(self) -> None:
        """Draw the entity on the screen."""
        screen = self.config.require_screen()
        if self.image:
            screen.blit(self.image, self.rect)
        if self.config.debug:
            pygame.draw.rect(screen, (255, 0, 0), self.rect, 1)
            font = pygame.font.SysFont("Arial", 13, True)
            text = font.render(
                f"{self.x:.1f}, {self.y:.1f}, {self.w:.1f}, {self.h:.1f}",
                True,
                (255, 255, 255),
            )
            screen.blit(
                text,
                (
                    self.rect.x + self.rect.w / 2 - text.get_width() / 2,
                    self.rect.y - text.get_height(),
                ),
            )
