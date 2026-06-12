"""Background entity."""

from typing import TYPE_CHECKING

from reinforced_flapper.game.entities.entity import Entity

if TYPE_CHECKING:
    from reinforced_flapper.game.core import GameConfig


class Background(Entity):
    """Static background scaled to the window."""

    def __init__(self, config: "GameConfig") -> None:
        """Initialize the background.

        Args:
            config: Shared game context.
        """
        super().__init__(
            config,
            config.images.background,
            0,
            0,
            config.window.width,
            config.window.height,
        )

    def tick(self) -> None:
        """Update the background (static, no-op)."""
