"""Environment behavior tests: determinism, bounds, rewards, truncation."""

import numpy as np
import pytest

from reinforced_flapper.config import EnvConfig
from reinforced_flapper.env import FlappyBirdEnv


@pytest.fixture
def env() -> FlappyBirdEnv:
    """A headless environment with default settings."""
    return FlappyBirdEnv()


def _rollout(env: FlappyBirdEnv, seed: int, actions: list[int]) -> list[np.ndarray]:
    """Reset with a seed and play a fixed action sequence.

    Args:
        env: The environment.
        seed: Reset seed.
        actions: Actions to play.

    Returns:
        The observation after reset and after each step.
    """
    obs, _info = env.reset(seed=seed)
    observations = [obs]
    for action in actions:
        obs, _reward, terminated, truncated, _info = env.step(action)
        observations.append(obs)
        if terminated or truncated:
            break
    return observations


def test_observation_within_space(env: FlappyBirdEnv) -> None:
    """All observations stay inside the declared observation space."""
    rng = np.random.default_rng(0)
    obs, _info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(200):
        obs, _reward, terminated, truncated, _info = env.step(int(rng.integers(0, 2)))
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _info = env.reset(seed=int(rng.integers(0, 10_000)))


def test_same_seed_same_trajectory(env: FlappyBirdEnv) -> None:
    """The same seed and actions produce an identical trajectory."""
    actions = [1 if i % 9 == 0 else 0 for i in range(120)]
    first = _rollout(env, seed=123, actions=actions)
    second = _rollout(env, seed=123, actions=actions)
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_differ(env: FlappyBirdEnv) -> None:
    """Different seeds produce different pipe layouts."""
    actions = [1 if i % 9 == 0 else 0 for i in range(120)]
    first = _rollout(env, seed=1, actions=actions)
    second = _rollout(env, seed=2, actions=actions)
    same_length = len(first) == len(second)
    identical = same_length and all(
        np.array_equal(a, b) for a, b in zip(first, second, strict=True)
    )
    assert not identical


def test_reward_values_come_from_config() -> None:
    """Step rewards are read from the env config, not hardcoded."""
    env = FlappyBirdEnv(EnvConfig(reward_alive=0.5, reward_death=-7.0))
    env.reset(seed=0)
    total_terminated = False
    while not total_terminated:
        _obs, reward, terminated, truncated, _info = env.step(0)
        total_terminated = terminated or truncated
        assert reward == (-7.0 if terminated else 0.5)


def test_truncation_at_max_episode_steps() -> None:
    """Episodes truncate (not terminate) at the configured step cap."""
    env = FlappyBirdEnv(EnvConfig(max_episode_steps=5))
    env.reset(seed=0)
    for step in range(5):
        _obs, _reward, terminated, truncated, info = env.step(0)
        assert not terminated
        assert truncated == (step == 4)
    assert info["steps"] == 5


def test_info_reports_score_and_steps(env: FlappyBirdEnv) -> None:
    """Info carries the game score and step count."""
    _obs, info = env.reset(seed=0)
    assert info["score"] == 0
    assert info["steps"] == 0


def test_rgb_array_rendering_headless() -> None:
    """Offscreen rendering returns frames without any display."""
    env = FlappyBirdEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (512, 288, 3)
    env.close()
