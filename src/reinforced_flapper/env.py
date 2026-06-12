"""Gymnasium environment for Flappy Bird with feature observations.

Simulation is fully decoupled from rendering: with render_mode=None nothing
touches the display or audio device, so training runs headless and fast.
With render_mode="rgb_array" frames are drawn to an offscreen surface (still
no display needed), and with render_mode="human" a window opens with sound.
"""

import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from reinforced_flapper.config import EnvConfig
from reinforced_flapper.game.assets import Images, NullSounds, Sounds, SoundsLike
from reinforced_flapper.game.core import GameConfig, Window
from reinforced_flapper.game.entities import (
    Background,
    Floor,
    Pipes,
    Player,
    PlayerMode,
    Score,
)

WINDOW_WIDTH = 288
WINDOW_HEIGHT = 512
FPS = 30


def _make_human_sounds() -> SoundsLike:
    """Load real sounds, falling back to silence if the mixer is unavailable.

    Returns:
        A sound set.
    """
    try:
        return Sounds()
    except pygame.error:
        return NullSounds()


class FlappyBirdEnv(gym.Env):
    """Flappy Bird as a Gymnasium environment.

    Action space: Discrete(2), 0 = no flap, 1 = flap.

    Observation space: 4 normalized features:
        bird y position [0, 1], bird y velocity [-1, 1], horizontal distance
        to the next pipe [0, 1], vertical distance to the gap center [-1, 1].

    Reward: env_config.reward_alive per surviving step,
    env_config.reward_death on collision. Episodes truncate at
    env_config.max_episode_steps because a mastered agent never dies.
    """

    metadata: dict[str, Any] = {  # noqa: RUF012 (matches gym.Env's declaration)
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        render_mode: str | None = None,
        fast_render: bool = False,
    ) -> None:
        """Initialize the environment.

        Args:
            env_config: Environment settings; defaults are the baseline.
            render_mode: None (headless), "human", or "rgb_array".
            fast_render: Skip the FPS cap in human mode.
        """
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.env_config = env_config or EnvConfig()
        self.render_mode = render_mode
        self.fast_render = fast_render

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        window = Window(WINDOW_WIDTH, WINDOW_HEIGHT)
        self._rng = random.Random()

        screen: pygame.Surface | None = None
        clock: pygame.time.Clock | None = None
        sounds: SoundsLike = NullSounds()
        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Flappy Bird")
            screen = pygame.display.set_mode((window.width, window.height))
            clock = pygame.time.Clock()
            sounds = _make_human_sounds()
        elif render_mode == "rgb_array":
            screen = pygame.Surface((window.width, window.height))

        images = Images(
            convert=render_mode == "human",
            rng=self._rng if self.env_config.randomize_sprites else None,
        )

        self._game = GameConfig(
            window=window,
            images=images,
            sounds=sounds,
            rng=self._rng,
            screen=screen,
            clock=clock,
            fps=FPS,
        )

        self._window_height = window.height
        self._window_width = window.width
        self._viewport_height = window.viewport_height

        self._steps = 0
        self._spawn_entities()

    def _spawn_entities(self) -> None:
        """Create fresh game entities for a new episode."""
        self.background = Background(self._game)
        self.floor = Floor(self._game)
        self.player = Player(self._game)
        self.pipes = Pipes(self._game)
        self.score = Score(self._game)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the start of a new episode.

        Args:
            seed: Seed for reproducible pipe layouts. When None, the pipe
                random stream continues from its current state.
            options: Unused.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        self._spawn_entities()
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        self._steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the game by one frame.

        Args:
            action: 0 = no flap, 1 = flap.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if action == 1:
            self.player.flap()

        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        for pipe in self.pipes.upper:
            if self.player.crossed(pipe):
                self.score.add()

        self._steps += 1
        terminated = self.player.collided(self.pipes, self.floor)
        truncated = not terminated and self._steps >= self.env_config.max_episode_steps

        reward = (
            self.env_config.reward_death if terminated else self.env_config.reward_alive
        )

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current frame.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode in ("rgb_array", "human"):
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray | None:
        """Draw the frame to the screen or offscreen surface.

        Returns:
            RGB array of the frame in rgb_array mode, None otherwise.
        """
        screen = self._game.screen
        if screen is None:
            raise RuntimeError("rendering requires render_mode to be set")
        screen.blit(self._game.images.background, (0, 0))
        self.floor.render()
        self.pipes.render()
        self.player.render()
        self.score.render()

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            if not self.fast_render:
                self._game.tick()
            return None

        return pygame.surfarray.array3d(screen).transpose(1, 0, 2)

    def close(self) -> None:
        """Release display resources."""
        if self.render_mode == "human":
            pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Build the 4-feature observation.

        Returns:
            Array of 4 normalized features.
        """
        bird_y = self.player.y / self._viewport_height
        # Max fall speed is 10; normalize velocity accordingly.
        bird_vel_y = self.player.vel_y / 10.0

        # The next pipe is the first whose trailing edge is ahead of the bird.
        next_pipe_idx = 0
        for i, pipe in enumerate(self.pipes.upper):
            if pipe.x + pipe.w > self.player.x:
                next_pipe_idx = i
                break

        if len(self.pipes.upper) > next_pipe_idx:
            next_upper_pipe = self.pipes.upper[next_pipe_idx]
            next_lower_pipe = self.pipes.lower[next_pipe_idx]

            pipe_dist_x = (next_upper_pipe.x - self.player.x) / self._window_width

            gap_top = next_upper_pipe.y + next_upper_pipe.h
            gap_bottom = next_lower_pipe.y
            gap_center = (gap_top + gap_bottom) / 2
            pipe_dist_y = (self.player.cy - gap_center) / self._viewport_height
        else:
            pipe_dist_x = 1.0
            pipe_dist_y = 0.0

        return np.array(
            [
                np.clip(bird_y, 0.0, 1.0),
                np.clip(bird_vel_y, -1.0, 1.0),
                np.clip(pipe_dist_x, 0.0, 1.0),
                np.clip(pipe_dist_y, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Build the info dict.

        Returns:
            Dictionary with the game score and step count.
        """
        return {
            "score": self.score.score if self.score else 0,
            "steps": self._steps,
        }
