"""DQN agent training and inference for Flappy Bird."""

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import yaml

from src.flappy_env import FlappyBirdEnv

# Default paths
MODELS_DIR = Path("models")
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        return yaml.safe_load(f)


class TrainingCallback(BaseCallback):
    """Custom callback for descriptive training output."""

    def __init__(
        self, print_freq: int = 10, max_history: int = 100, verbose: int = 0
    ) -> None:
        """Initialize the callback.

        Args:
            print_freq: Print stats every N episodes.
            max_history: Maximum episodes to keep in memory for stats.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.print_freq = print_freq
        self.max_history = max_history
        self.episode_count = 0
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.total_reward_sum = 0.0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        """Called after each environment step.

        Returns:
            True to continue training.
        """
        # Check if episode ended
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self.episode_count += 1

            # Get episode info
            infos = self.locals.get("infos", [{}])
            if infos and "episode" in infos[0]:
                ep_info = infos[0]["episode"]
                reward = ep_info["r"]
                length = ep_info["l"]

                # Track for overall stats
                self.total_reward_sum += reward
                self.total_episodes += 1

                # Keep limited history for recent stats
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)

                # Trim history to prevent memory growth
                if len(self.episode_rewards) > self.max_history:
                    self.episode_rewards.pop(0)
                    self.episode_lengths.pop(0)

            # Print stats every N episodes
            if self.episode_count % self.print_freq == 0:
                self._print_stats()

        return True

    def _print_stats(self) -> None:
        """Print formatted training statistics."""
        if not self.episode_rewards:
            return

        # Calculate averages over recent episodes
        recent_rewards = self.episode_rewards[-self.print_freq :]
        recent_lengths = self.episode_lengths[-self.print_freq :]

        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_length = sum(recent_lengths) / len(recent_lengths)

        # Get exploration rate from model
        exploration = self.model.exploration_rate * 100

        # Get loss if available
        loss_str = ""
        if hasattr(self.model, "logger") and self.model.logger is not None:
            loss = self.model.logger.name_to_value.get("train/loss", None)
            if loss is not None:
                loss_str = f" | Loss: {loss:.3f} (prediction error)"

        # Print with descriptions
        print(
            f"Episode {self.episode_count:>4} | "
            f"Avg Length: {avg_length:>5.0f} steps | "
            f"Avg Reward: {avg_reward:>7.0f} | "
            f"Explore: {exploration:>4.1f}%{loss_str}"
        )

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.total_episodes > 0:
            total_avg = self.total_reward_sum / self.total_episodes
            print(f"\nTraining complete! Total episodes: {self.total_episodes}")
            print(f"Overall average reward: {total_avg:.0f}")


def train_agent(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    render: bool = False,
) -> Path:
    """Train a DQN agent to play Flappy Bird.

    Args:
        config_path: Path to the YAML configuration file.
        render: Whether to render the game during training.

    Returns:
        Path to the saved model.
    """
    # Load configuration
    config = load_config(config_path)
    training_config = config.get("training", {})
    dqn_config = config.get("dqn", {})
    callback_config = config.get("callback", {})

    total_timesteps = training_config.get("total_timesteps", 100000)
    model_name = training_config.get("model_name", "dqn_flappybird")

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)

    # Create the environment with optional fast rendering
    render_mode = "human" if render else None
    env = make_vec_env(
        FlappyBirdEnv,
        n_envs=1,
        wrapper_class=Monitor,  # Wrap with Monitor for episode stats
        env_kwargs={
            "render_mode": render_mode,
            "fast_render": render,  # Fast render when visualizing training
        },
    )

    # Initialize the DQN model with hyperparameters from config
    model = DQN(
        dqn_config.get("policy", "MlpPolicy"),
        env,
        verbose=0,  # We use custom callback for output
        buffer_size=dqn_config.get("buffer_size", 10000),
        learning_starts=dqn_config.get("learning_starts", 1000),
        batch_size=dqn_config.get("batch_size", 32),
        target_update_interval=dqn_config.get("target_update_interval", 500),
        learning_rate=dqn_config.get("learning_rate", 1e-4),
        exploration_fraction=dqn_config.get("exploration_fraction", 0.1),
        exploration_final_eps=dqn_config.get("exploration_final_eps", 0.05),
    )

    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Config: {config_path}")
    print(
        "\nMetric descriptions:\n"
        "  Avg Length  = Steps before death (higher is better)\n"
        "  Avg Reward  = Total reward per episode (higher is better)\n"
        "  Explore     = Random action probability (decreases over time)\n"
        "  Loss        = Neural network prediction error (lower is better)\n"
    )

    # Train with custom callback
    callback = TrainingCallback(
        print_freq=callback_config.get("print_freq", 10),
        max_history=callback_config.get("max_history", 100),
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the model
    model_path = MODELS_DIR / model_name
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}.zip")

    env.close()

    return model_path


def run_agent(
    model_path: str | Path,
    num_episodes: int = 5,
    render: bool = True,
) -> None:
    """Run a trained agent to play Flappy Bird.

    Args:
        model_path: Path to the saved model file.
        num_episodes: Number of episodes to run.
        render: Whether to render the game visually.
    """
    model_path = Path(model_path)

    # Remove .zip extension if present (DQN.load handles it automatically)
    model_path_str = str(model_path)
    if model_path_str.endswith(".zip"):
        model_path_str = model_path_str[:-4]

    # Check if model exists
    if not Path(f"{model_path_str}.zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path_str}.zip")

    # Load the model
    model = DQN.load(model_path_str)

    # Create environment with rendering if requested
    render_mode = "human" if render else None
    env = FlappyBirdEnv(render_mode=render_mode)

    episodes_completed = 0
    total_rewards = []

    while episodes_completed < num_episodes:
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episodes_completed += 1
        total_rewards.append(episode_reward)
        print(
            f"Episode {episodes_completed}/{num_episodes} - "
            f"Score: {info['score']} - Reward: {episode_reward:.0f}"
        )

    env.close()

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.0f}")


if __name__ == "__main__":
    # Default behavior: train a new model
    train_agent()
