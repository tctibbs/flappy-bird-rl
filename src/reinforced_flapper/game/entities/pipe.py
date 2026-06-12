"""Pipe entities."""

from typing import TYPE_CHECKING, Any

from reinforced_flapper.game.entities.entity import Entity

if TYPE_CHECKING:
    from reinforced_flapper.game.core import GameConfig


class Pipe(Entity):
    """A single pipe.

    Attributes:
        vel_x: Horizontal velocity.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the pipe."""
        super().__init__(*args, **kwargs)
        self.vel_x = -5

    def tick(self) -> None:
        """Move the pipe one frame."""
        self.x += self.vel_x

    def render(self) -> None:
        """Draw the pipe."""
        return super().render()


class Pipes(Entity):
    """The set of pipe pairs on screen.

    Gap positions are sampled from the game context's rng, so pipe layouts
    are reproducible from the environment seed.

    Attributes:
        pipe_gap: Vertical gap between an upper and lower pipe.
        upper: Upper pipes, leftmost first.
        lower: Lower pipes, leftmost first.
    """

    upper: list[Pipe]
    lower: list[Pipe]

    def __init__(self, config: "GameConfig") -> None:
        """Initialize and spawn the first pipes.

        Args:
            config: Shared game context.
        """
        super().__init__(config)
        self.pipe_gap = 120
        self.top = 0
        self.bottom = self.config.window.viewport_height
        self.upper = []
        self.lower = []
        self.spawn_initial_pipes()

    def tick(self) -> None:
        """Spawn, advance, and prune pipes for one frame."""
        if self.can_spawn_pipes():
            self.spawn_new_pipes()
        self.remove_old_pipes()

        for up_pipe, low_pipe in zip(self.upper, self.lower, strict=False):
            up_pipe.tick()
            low_pipe.tick()

    def stop(self) -> None:
        """Stop all pipe movement."""
        for pipe in self.upper + self.lower:
            pipe.vel_x = 0

    def can_spawn_pipes(self) -> bool:
        """Check whether a new pipe pair should spawn."""
        last = self.upper[-1]
        if not last:
            return True

        return self.config.window.width - (last.x + last.w) > last.w * 2.5

    def spawn_new_pipes(self) -> None:
        """Append a new randomly placed pipe pair."""
        upper, lower = self.make_random_pipes()
        self.upper.append(upper)
        self.lower.append(lower)

    def remove_old_pipes(self) -> None:
        """Drop pipes that have scrolled off the left edge."""
        for pipe in self.upper:
            if pipe.x < -pipe.w:
                self.upper.remove(pipe)

        for pipe in self.lower:
            if pipe.x < -pipe.w:
                self.lower.remove(pipe)

    def spawn_initial_pipes(self) -> None:
        """Spawn the two initial pipe pairs."""
        upper_1, lower_1 = self.make_random_pipes()
        upper_1.x = self.config.window.width + upper_1.w * 3
        lower_1.x = self.config.window.width + upper_1.w * 3
        self.upper.append(upper_1)
        self.lower.append(lower_1)

        upper_2, lower_2 = self.make_random_pipes()
        upper_2.x = upper_1.x + upper_1.w * 3.5
        lower_2.x = upper_1.x + upper_1.w * 3.5
        self.upper.append(upper_2)
        self.lower.append(lower_2)

    def make_random_pipes(self) -> tuple[Pipe, Pipe]:
        """Create a pipe pair with a randomly placed gap.

        Returns:
            The upper and lower pipe.
        """
        base_y = self.config.window.viewport_height

        gap_y = self.config.rng.randrange(0, int(base_y * 0.6 - self.pipe_gap))
        gap_y += int(base_y * 0.2)
        pipe_height = self.config.images.pipe[0].get_height()
        pipe_x = self.config.window.width + 10

        upper_pipe = Pipe(
            self.config,
            self.config.images.pipe[0],
            pipe_x,
            gap_y - pipe_height,
        )

        lower_pipe = Pipe(
            self.config,
            self.config.images.pipe[1],
            pipe_x,
            gap_y + self.pipe_gap,
        )

        return upper_pipe, lower_pipe

    def render(self) -> None:
        """Draw all pipes."""
        for up_pipe, low_pipe in zip(self.upper, self.lower, strict=False):
            up_pipe.render()
            low_pipe.render()
