"""Pre-refactor starting-line measurement.

Evaluates the existing trained model (models/dqn_flappybird.zip) and a
random policy against the current environment, before any refactoring.
Results are written as JSONL so every later claim compares to a real,
recorded number pinned to the git SHA of the unmodified code.

Run headless:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python scripts/baseline_pre_refactor.py
"""

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import DQN

from src.flappy_env import FlappyBirdEnv

MODEL_PATH = "models/dqn_flappybird"
N_EPISODES = 100
MAX_STEPS = 10_000
SEED_BASE = 10_000
OUT_PATH = Path("results/baseline_pre_refactor.jsonl")


def git_sha() -> str:
    """Return the current git commit SHA."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def run_condition(env: FlappyBirdEnv, policy: str, model: DQN | None) -> dict:
    """Run N_EPISODES under one policy condition and return summary stats."""
    scores: list[int] = []
    lengths: list[int] = []
    rng = np.random.default_rng(SEED_BASE)
    start = time.time()

    for i in range(N_EPISODES):
        seed = SEED_BASE + i
        obs, info = env.reset(seed=seed)
        steps = 0
        while steps < MAX_STEPS:
            if policy == "random":
                action = int(rng.integers(0, 2))
            elif policy == "deterministic":
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:  # epsilon-greedy at the model's saved exploration rate
                action, _ = model.predict(obs, deterministic=False)
                action = int(action)
            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        scores.append(int(info["score"]))
        lengths.append(steps)

    scores_arr = np.array(scores)
    lengths_arr = np.array(lengths)
    return {
        "policy": policy,
        "n_episodes": N_EPISODES,
        "max_steps": MAX_STEPS,
        "seed_base": SEED_BASE,
        "score_mean": float(scores_arr.mean()),
        "score_std": float(scores_arr.std(ddof=1)),
        "score_median": float(np.median(scores_arr)),
        "score_min": int(scores_arr.min()),
        "score_max": int(scores_arr.max()),
        "length_mean": float(lengths_arr.mean()),
        "length_median": float(np.median(lengths_arr)),
        "n_capped": int((lengths_arr >= MAX_STEPS).sum()),
        "wall_clock_s": round(time.time() - start, 1),
        "scores": scores,
    }


def main() -> None:
    """Measure the existing model and a random policy, write JSONL rows."""
    random.seed(SEED_BASE)  # fixes sprite selection in Images.randomize()
    OUT_PATH.parent.mkdir(exist_ok=True)

    model = DQN.load(MODEL_PATH)
    env = FlappyBirdEnv(render_mode=None)

    meta = {
        "git_sha": git_sha(),
        "model_path": f"{MODEL_PATH}.zip",
        "exploration_rate": float(model.exploration_rate),
        "python": platform.python_version(),
        "note": "pre-refactor starting line, unmodified env",
    }

    with OUT_PATH.open("w") as f:
        for policy in ["random", "deterministic", "stochastic"]:
            row = run_condition(env, policy, None if policy == "random" else model)
            row.update(meta)
            f.write(json.dumps(row) + "\n")
            print(
                f"{policy:>13}: score mean={row['score_mean']:.2f} "
                f"median={row['score_median']:.1f} std={row['score_std']:.2f} "
                f"min={row['score_min']} max={row['score_max']} "
                f"capped={row['n_capped']}/{N_EPISODES} "
                f"len mean={row['length_mean']:.0f} ({row['wall_clock_s']}s)"
            )

    env.close()
    print(f"\nWritten to {OUT_PATH} at git {meta['git_sha'][:10]}")


if __name__ == "__main__":
    main()
