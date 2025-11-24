"""Main entry point for Reinforced Flapper.

Supports three modes:
- human: Play the game yourself
- agent: Watch a trained agent play
- agent_training: Train a new agent
"""

import argparse

from src.agent import run_agent, train_agent
from src.flappy_env import human_play


def main() -> None:
    """Entry point of the program."""
    args = parse_args()

    if args.mode == "human":
        human_play()

    elif args.mode == "agent_training":
        train_agent(
            config_path=args.config,
            render=args.render,
        )

    elif args.mode == "agent":
        run_agent(
            model_path=args.model,
            num_episodes=args.episodes,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Flappy Bird Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "mode",
        choices=["human", "agent", "agent_training"],
        help="Mode to run: 'human' to play, 'agent' to watch trained model, "
        "'agent_training' to train a new model",
    )

    # Training arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game during training to watch it learn",
    )

    # Inference arguments
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn_flappybird",
        help="Path to trained model for agent mode (default: models/dqn_flappybird)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run in agent mode (default: 5)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
