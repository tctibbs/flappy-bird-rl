"""Fixed evaluation protocol.

Episode i resets with seed seed_base + i, so every policy is evaluated on
the same fixed set of pipe layouts and comparisons are paired. Scores are
game scores (pipes passed), not rewards.
"""

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel
from stable_baselines3.common.base_class import BaseAlgorithm

from reinforced_flapper.config import EnvConfig, EvalProtocolConfig
from reinforced_flapper.env import FlappyBirdEnv

ActionFn = Callable[[np.ndarray], int]


class EvalResult(BaseModel):
    """Result of evaluating one policy under the fixed protocol.

    Attributes:
        policy: Policy label, e.g. "model" or "random".
        deterministic: Whether actions were chosen greedily.
        n_episodes: Number of episodes evaluated.
        seed_base: Seed of the first episode.
        max_episode_steps: Step cap each episode ran under.
        scores: Game score per episode.
        lengths: Steps survived per episode.
    """

    policy: str
    deterministic: bool
    n_episodes: int
    seed_base: int
    max_episode_steps: int
    scores: list[int]
    lengths: list[int]

    def summary(self) -> dict[str, float]:
        """Summarize scores and lengths.

        Returns:
            Flat metric names to values.
        """
        scores = np.asarray(self.scores)
        lengths = np.asarray(self.lengths)
        return {
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "score_median": float(np.median(scores)),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "length_mean": float(lengths.mean()),
            "length_median": float(np.median(lengths)),
            "episodes_capped": float((lengths >= self.max_episode_steps).sum()),
        }


def _run_protocol(
    action_fn: ActionFn,
    env: FlappyBirdEnv,
    protocol: EvalProtocolConfig,
    *,
    policy: str,
    deterministic: bool,
) -> EvalResult:
    """Run the fixed protocol with an action function.

    Args:
        action_fn: Maps an observation to an action.
        env: The environment to evaluate in.
        protocol: The evaluation protocol.
        policy: Policy label for the result.
        deterministic: Whether the policy is deterministic, for the result.

    Returns:
        The evaluation result.
    """
    scores: list[int] = []
    lengths: list[int] = []
    for i in range(protocol.n_episodes):
        obs, info = env.reset(seed=protocol.seed_base + i)
        while True:
            obs, _reward, terminated, truncated, info = env.step(action_fn(obs))
            if terminated or truncated:
                break
        scores.append(int(info["score"]))
        lengths.append(int(info["steps"]))

    return EvalResult(
        policy=policy,
        deterministic=deterministic,
        n_episodes=protocol.n_episodes,
        seed_base=protocol.seed_base,
        max_episode_steps=env.env_config.max_episode_steps,
        scores=scores,
        lengths=lengths,
    )


def evaluate_model(
    model: BaseAlgorithm,
    env_config: EnvConfig,
    protocol: EvalProtocolConfig,
    *,
    deterministic: bool = True,
    env: FlappyBirdEnv | None = None,
) -> EvalResult:
    """Evaluate a trained model under the fixed protocol.

    Args:
        model: The trained SB3 model.
        env_config: Environment settings to evaluate under.
        protocol: The evaluation protocol.
        deterministic: Choose actions greedily. When False, DQN acts
            epsilon-greedily at its saved exploration rate.
        env: Reuse an existing environment instead of creating one.

    Returns:
        The evaluation result.
    """

    def action_fn(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=deterministic)
        return int(action)

    own_env = env is None
    env = env or FlappyBirdEnv(env_config)
    try:
        return _run_protocol(
            action_fn,
            env,
            protocol,
            policy="model",
            deterministic=deterministic,
        )
    finally:
        if own_env:
            env.close()


def evaluate_random(
    env_config: EnvConfig,
    protocol: EvalProtocolConfig,
) -> EvalResult:
    """Evaluate a uniform random policy under the fixed protocol.

    Args:
        env_config: Environment settings to evaluate under.
        protocol: The evaluation protocol.

    Returns:
        The evaluation result.
    """
    rng = np.random.default_rng(protocol.seed_base)

    def action_fn(_obs: np.ndarray) -> int:
        return int(rng.integers(0, 2))

    env = FlappyBirdEnv(env_config)
    try:
        return _run_protocol(
            action_fn,
            env,
            protocol,
            policy="random",
            deterministic=False,
        )
    finally:
        env.close()
