"""Flappy Bird Gymnasium environment for reinforcement learning."""

import random
from typing import Any, ClassVar

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from src.entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from src.utils import GameConfig, Images, Sounds, Window


class FlappyBirdEnv(gym.Env):
    """Custom Gymnasium Environment for Flappy Bird.

    Uses feature-based observations instead of raw pixels for efficient learning.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self, render_mode: str | None = None, fast_render: bool = False
    ) -> None:
        """Initialize the Flappy Bird environment.

        Args:
            render_mode: The mode to render the environment in.
            fast_render: If True, skip FPS cap for faster training visualization.
        """
        super().__init__()

        self.render_mode = render_mode
        self.fast_render = fast_render

        # Actions: 0 = no flap, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Feature-based observation space (4 normalized values):
        # - Bird Y position (normalized 0-1)
        # - Bird Y velocity (normalized -1 to 1)
        # - Horizontal distance to next pipe (normalized 0-1)
        # - Vertical distance to pipe gap center (normalized -1 to 1)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

        # Store window dimensions for normalization
        self._window_height = window.height
        self._window_width = window.width
        self._viewport_height = window.viewport_height

        # Initialize game entities (will be reset properly in reset())
        self.background: Background | None = None
        self.floor: Floor | None = None
        self.player: Player | None = None
        self.pipes: Pipes | None = None
        self.score: Score | None = None
        self.welcome_message: WelcomeMessage | None = None
        self.game_over_message: GameOver | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for reset.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Seed random for reproducible pipe positions
        if seed is not None:
            random.seed(seed)

        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)

        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: The action to take (0 = no flap, 1 = flap).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if action == 1:
            self.player.flap()

        # Update game state
        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        # Check for score
        for pipe in self.pipes.upper:
            if self.player.crossed(pipe):
                self.score.add()

        # Check for collision
        terminated = self.player.collided(self.pipes, self.floor)
        truncated = False

        # Calculate reward
        reward = self._calculate_reward(terminated)

        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()

        # Render if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        return None

    def _render_frame(self) -> np.ndarray | None:
        """Render a single frame.

        Returns:
            RGB array of the frame.
        """
        # Pump events to keep window responsive
        pygame.event.pump()

        self.config.screen.blit(self.config.images.background, (0, 0))
        self.floor.render()
        self.pipes.render()
        self.player.render()
        self.score.render()
        pygame.display.update()

        # Skip FPS cap in fast render mode for faster training visualization
        if not self.fast_render:
            self.config.clock.tick(self.config.fps)

        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(pygame.display.get_surface()).transpose(
                1, 0, 2
            )
        return None

    def close(self) -> None:
        """Close the environment."""
        pygame.quit()

    def _get_observation(self) -> np.ndarray:
        """Get feature-based observation.

        Returns:
            Array of 4 normalized features.
        """
        # Bird Y position normalized to [0, 1]
        bird_y = self.player.y / self._viewport_height

        # Bird Y velocity normalized to [-1, 1]
        # Max velocity is about 10-15, normalize accordingly
        bird_vel_y = self.player.vel_y / 10.0

        # Find the next pipe (first pipe that hasn't been passed)
        next_pipe_idx = 0
        for i, pipe in enumerate(self.pipes.upper):
            if pipe.x + pipe.w > self.player.x:
                next_pipe_idx = i
                break

        if len(self.pipes.upper) > next_pipe_idx:
            next_upper_pipe = self.pipes.upper[next_pipe_idx]
            next_lower_pipe = self.pipes.lower[next_pipe_idx]

            # Horizontal distance to pipe normalized to [0, 1]
            pipe_dist_x = (next_upper_pipe.x - self.player.x) / self._window_width

            # Vertical distance to gap center normalized to [-1, 1]
            gap_top = next_upper_pipe.y + next_upper_pipe.h
            gap_bottom = next_lower_pipe.y
            gap_center = (gap_top + gap_bottom) / 2
            pipe_dist_y = (self.player.cy - gap_center) / self._viewport_height
        else:
            # No pipes visible (shouldn't happen in normal gameplay)
            pipe_dist_x = 1.0
            pipe_dist_y = 0.0

        # Clamp values to observation space bounds
        observation = np.array(
            [
                np.clip(bird_y, 0.0, 1.0),
                np.clip(bird_vel_y, -1.0, 1.0),
                np.clip(pipe_dist_x, 0.0, 1.0),
                np.clip(pipe_dist_y, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        return observation

    def _calculate_reward(self, terminated: bool) -> float:
        """Calculate the reward for the current step.

        Args:
            terminated: Whether the episode has ended.

        Returns:
            The reward value.
        """
        if terminated:
            return -100.0
        return 1.0

    def _get_info(self) -> dict[str, Any]:
        """Get additional info about the environment state.

        Returns:
            Dictionary with debug information.
        """
        return {
            "score": self.score.score if self.score else 0,
            "player_y": self.player.y if self.player else 0,
            "player_vel_y": self.player.vel_y if self.player else 0,
        }


def human_play() -> None:
    """Run the game in human playable mode."""
    env = FlappyBirdEnv(render_mode="human")
    env.reset()

    running = True
    while running:
        action = 0

        # Event handling for human play
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
                break
            if _is_tap_event(event):
                action = 1

        if not running:
            break

        _, _, terminated, _, _ = env.step(action)

        if terminated:
            env.reset()

    env.close()


def _is_tap_event(event: pygame.event.Event) -> bool:
    """Check if the event is a tap event.

    Args:
        event: The pygame event to check.

    Returns:
        True if the event is a tap event.
    """
    m_left, _, _ = pygame.mouse.get_pressed()
    space_or_up = event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)
    screen_tap = event.type == pygame.FINGERDOWN
    return m_left or space_or_up or screen_tap


if __name__ == "__main__":
    human_play()
