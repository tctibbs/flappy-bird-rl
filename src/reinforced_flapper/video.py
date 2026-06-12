"""Record evaluation videos from offscreen frames (no display needed)."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN

from reinforced_flapper.config import EnvConfig
from reinforced_flapper.env import FPS, FlappyBirdEnv


def record_video(
    model_path: Path | str,
    out_path: Path | str,
    env_config: EnvConfig | None = None,
    *,
    n_episodes: int = 1,
    seed: int = 10_000,
    max_steps: int = 3_000,
) -> Path:
    """Record a trained model playing, to an mp4 file.

    Args:
        model_path: Path to the saved model.
        out_path: Destination mp4 path.
        env_config: Environment settings; defaults are the baseline.
        n_episodes: Episodes to record back to back.
        seed: Seed of the first episode; episode i uses seed + i.
        max_steps: Frame cap per episode, to bound file size for an agent
            that never dies.

    Returns:
        The written video path.
    """
    model = DQN.load(str(model_path))
    env = FlappyBirdEnv(env_config, render_mode="rgb_array")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def frame() -> np.ndarray:
        rendered = env.render()
        if rendered is None:
            raise RuntimeError("rgb_array rendering returned no frame")
        return rendered

    writer = imageio.get_writer(str(out_path), fps=FPS, macro_block_size=1)
    try:
        for i in range(n_episodes):
            obs, _info = env.reset(seed=seed + i)
            writer.append_data(frame())
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(int(action))
                writer.append_data(frame())
                if terminated or truncated:
                    break
    finally:
        writer.close()
        env.close()
    return out_path
