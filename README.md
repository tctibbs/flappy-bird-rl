# Flappy Bird RL

[![CI](https://github.com/tctibbs/flappy-bird-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/tctibbs/flappy-bird-rl/actions/workflows/ci.yml)

A DQN agent that learns to play Flappy Bird perfectly, and a clean,
reproducible pipeline around it.

<p align="center">
  <img src="docs/media/demo.gif" alt="Trained agent playing Flappy Bird" width="216">
</p>

## Results

All 3 training seeds reach perfect play: the best checkpoint never dies
in evaluation, scoring 255 pipes on every one of 100 fixed episodes (the
protocol's ceiling). A random policy scores 0 on the same episodes.

![Learning curves](plots/learning_curves.png)

Training takes about 5 minutes per seed on a laptop CPU, fully headless.
The one hyperparameter that mattered was reward scale: +1 per frame and
-100 on death never worked, +0.1 and -1.0 solved the game
([journal](journal/003-reward-scale-was-the-whole-problem.md)).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras

uv run flapper play                                       # play it yourself
uv run flapper train --config configs/final.yaml          # train an agent
uv run flapper watch --model runs/<run_id>/best_model     # watch it play
uv run flapper evaluate --model runs/<run_id>/best_model  # 100-episode eval
```

`make check` runs lint, format, types, and tests.

## How it works

- The bird sees 4 numbers (height, velocity, distance to the next pipe,
  distance to the gap center) and decides each frame: flap or not.
- Stable-Baselines3 DQN, trained headless, evaluated on 100 fixed-seed
  episodes so every policy faces the same pipes.
- Every run records its typed, hashed config and results to
  results/ledger.jsonl, so each number traces to an exact configuration
  and git commit.

More detail: [method and results](docs/summary.md),
[decision records](docs/adr), [experiment journal](journal).

## License

MIT. Game derived from [FlapPyBird](https://github.com/sourabhv/FlapPyBird).
