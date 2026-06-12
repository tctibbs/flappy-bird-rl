"""Scrolling floor entity."""

from typing import TYPE_CHECKING

from reinforced_flapper.game.entities.entity import Entity

if TYPE_CHECKING:
    from reinforced_flapper.game.core import GameConfig


class Floor(Entity):
    """Scrolling floor.

    Attributes:
        vel_x: Horizontal scroll speed.
        x_extra: Extra sprite width beyond the window, used for wrapping.
    """

    def __init__(self, config: "GameConfig") -> None:
        """Initialize the floor.

        Args:
            config: Shared game context.
        """
        super().__init__(config, config.images.base, 0, config.window.vh)
        self.vel_x = 4
        self.x_extra = self.w - config.window.w

    def stop(self) -> None:
        """Stop the floor movement."""
        self.vel_x = 0

    def tick(self) -> None:
        """Scroll the floor one frame."""
        self.x = -((-self.x + self.vel_x) % self.x_extra)
