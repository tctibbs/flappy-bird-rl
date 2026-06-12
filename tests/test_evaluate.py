"""Evaluation protocol tests."""

from reinforced_flapper.config import EnvConfig, EvalProtocolConfig
from reinforced_flapper.evaluate import evaluate_random


def test_random_eval_shape_and_reproducibility() -> None:
    """The protocol runs the requested episodes and is reproducible."""
    env_config = EnvConfig(max_episode_steps=500)
    protocol = EvalProtocolConfig(n_episodes=5, seed_base=42)

    first = evaluate_random(env_config, protocol)
    second = evaluate_random(env_config, protocol)

    assert first.n_episodes == 5
    assert len(first.scores) == 5
    assert len(first.lengths) == 5
    assert first.scores == second.scores
    assert first.lengths == second.lengths

    summary = first.summary()
    assert summary["score_mean"] >= 0
    assert summary["length_mean"] > 0
