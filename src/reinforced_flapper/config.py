"""Typed experiment configuration: validated, hashable, YAML round-trippable.

Every meaningful hyperparameter is a typed field here. A run's config hash
(which excludes the training seed, so seed replicates of the same setup
group together) is stamped into every ledger row.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class EnvConfig(StrictModel):
    """Environment settings.

    Attributes:
        reward_alive: Reward per step survived.
        reward_death: Reward on collision.
        max_episode_steps: Step cap per episode; episodes truncate here. A
            mastered agent never dies, so an uncapped episode never ends.
        randomize_sprites: Randomize sprite variants per environment
            instance instead of using the fixed defaults.
    """

    reward_alive: float = 1.0
    reward_death: float = -100.0
    max_episode_steps: int = Field(default=10_000, ge=1)
    randomize_sprites: bool = False


class DQNConfig(StrictModel):
    """DQN hyperparameters, passed to Stable-Baselines3.

    Attributes:
        policy: SB3 policy class name.
        learning_rate: Optimizer learning rate.
        buffer_size: Replay buffer capacity in transitions.
        learning_starts: Steps collected before learning begins.
        batch_size: Minibatch size per gradient step.
        gamma: Discount factor.
        train_freq: Environment steps between gradient updates.
        gradient_steps: Gradient steps per update.
        target_update_interval: Steps between target network syncs.
        exploration_fraction: Fraction of training over which epsilon decays.
        exploration_final_eps: Final epsilon for epsilon-greedy exploration.
        net_arch: Hidden layer sizes of the Q network.
    """

    policy: Literal["MlpPolicy"] = "MlpPolicy"
    learning_rate: float = Field(default=3e-4, gt=0)
    buffer_size: int = Field(default=100_000, ge=1)
    learning_starts: int = Field(default=5_000, ge=0)
    batch_size: int = Field(default=64, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    train_freq: int = Field(default=4, ge=1)
    gradient_steps: int = Field(default=1, ge=1)
    target_update_interval: int = Field(default=1_000, ge=1)
    exploration_fraction: float = Field(default=0.2, gt=0, le=1)
    exploration_final_eps: float = Field(default=0.02, ge=0, le=1)
    net_arch: list[int] = Field(default_factory=lambda: [256, 256])


class TrainConfig(StrictModel):
    """Training run settings.

    Attributes:
        total_timesteps: Environment steps to train for.
        seed: Seed for the environment, network init, and exploration. Not
            part of the config hash, so seed replicates share a hash.
        eval_every_steps: Steps between periodic evaluations (learning curve
            samples and best-model selection).
        eval_episodes: Episodes per periodic evaluation.
    """

    total_timesteps: int = Field(default=300_000, ge=1)
    seed: int = 0
    eval_every_steps: int = Field(default=20_000, ge=1)
    eval_episodes: int = Field(default=20, ge=1)


class EvalProtocolConfig(StrictModel):
    """The fixed evaluation protocol behind every headline number.

    Episode i resets with seed seed_base + i, so all policies face the same
    fixed set of pipe layouts and results are paired across policies.

    Attributes:
        n_episodes: Number of evaluation episodes.
        seed_base: Seed of the first episode.
    """

    n_episodes: int = Field(default=100, ge=1)
    seed_base: int = 10_000


class RunConfig(StrictModel):
    """Root configuration for a training or evaluation run.

    Attributes:
        name: Human-readable experiment name.
        env: Environment settings.
        dqn: DQN hyperparameters.
        train: Training settings.
        evaluation: Fixed evaluation protocol.
    """

    name: str = "dqn_baseline"
    env: EnvConfig = Field(default_factory=EnvConfig)
    dqn: DQNConfig = Field(default_factory=DQNConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    evaluation: EvalProtocolConfig = Field(default_factory=EvalProtocolConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        """Load and validate a config from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The validated config.
        """
        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Serialize the config to a YAML file.

        Args:
            path: Destination path.
        """
        with Path(path).open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    def config_hash(self) -> str:
        """Hash the config, excluding the training seed.

        Returns:
            First 12 hex characters of the sha256 of the canonical JSON.
        """
        data = self.model_dump(mode="json")
        data["train"].pop("seed")
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]
