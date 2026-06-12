"""Config schema tests: validation, YAML round-trip, hashing."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from reinforced_flapper.config import EnvConfig, RunConfig


def test_defaults_validate() -> None:
    """The default config is valid."""
    config = RunConfig()
    assert config.env.reward_alive == 1.0
    assert config.train.total_timesteps >= 1


def test_yaml_round_trip(tmp_path: Path) -> None:
    """A config survives YAML serialization unchanged."""
    config = RunConfig(name="round_trip")
    config.train.seed = 7
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = RunConfig.from_yaml(path)
    assert loaded == config


def test_unknown_fields_rejected() -> None:
    """Unknown fields fail validation instead of being silently dropped."""
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "x", "unknown_field": 1})
    with pytest.raises(ValidationError):
        EnvConfig.model_validate({"reward_aliv": 1.0})


def test_invalid_values_rejected() -> None:
    """Out-of-range values fail validation."""
    with pytest.raises(ValidationError):
        EnvConfig(max_episode_steps=0)


def test_hash_is_stable() -> None:
    """Two identical configs hash identically."""
    assert RunConfig().config_hash() == RunConfig().config_hash()


def test_hash_ignores_seed_and_name() -> None:
    """Seed replicates and renamed copies of the same config share a hash."""
    a = RunConfig()
    b = RunConfig(name="renamed")
    b.train.seed = 99
    assert a.config_hash() == b.config_hash()


def test_hash_changes_with_settings() -> None:
    """Changing any real setting changes the hash."""
    a = RunConfig()
    b = RunConfig()
    b.env.reward_death = -50.0
    assert a.config_hash() != b.config_hash()
