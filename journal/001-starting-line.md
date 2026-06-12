# 001: The starting line is zero

Date: 2026-06-12
Git SHA: 6582c827b0 (unmodified code)
Raw data: results/baseline_pre_refactor.jsonl

## What was measured

The baseline for this project is the random policy, evaluated against the
unmodified environment: 100 episodes, episode seeds 10000..10099, step cap
10000, headless (SDL dummy drivers).

| Policy | Score mean | Score median | Score max | Length mean |
|---|---|---|---|---|
| Random (uniform flap 50%) | 0.00 | 0 | 0 | 72 steps |

Every later claim is measured against this number under the same protocol.

## Why there is no model baseline

A model file found on disk (a local, untracked models/dqn_flappybird.zip)
was evaluated as a candidate starting line and scored 0.00 deterministic
with a 31-step mean episode length, exactly the free-fall time from spawn
to floor. Inspection showed it was saved at num_timesteps=2000 with
learning_starts=1000, and a grid sweep over the observation space found it
selects flap in 3 of 3645 states: a never-flap smoke-test artifact, not a
trained agent. It was excluded as a baseline; the random policy is the
reference.

## Lesson

Never assume an artifact is what its name claims. "Trained model" meant
"smoke-test output". Measuring first kept the project from comparing
against an assumption, and the meaningful targets became the random
baseline and a fixed solved threshold defined before training (ADR 0005).
