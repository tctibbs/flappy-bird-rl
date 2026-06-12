# 001: The starting line is zero

Date: 2026-06-12
Git SHA: 6582c827b0 (unmodified code)
Raw data: results/baseline_pre_refactor.jsonl

## What was measured

The existing trained model (models/dqn_flappybird.zip) and a random policy,
evaluated against the unmodified environment. 100 episodes per condition,
episode seeds 10000..10099, step cap 10000, headless (SDL dummy drivers).

| Policy | Score mean | Score median | Score max | Length mean |
|---|---|---|---|---|
| Existing model (deterministic) | 0.00 | 0 | 0 | 31 steps |
| Existing model (eps=0.05 greedy) | 0.01 | 0 | 1 | 44 steps |
| Random (uniform flap 50%) | 0.00 | 0 | 0 | 72 steps |

## Lesson

The shipped model is not a trained agent. Inspection shows it was saved at
num_timesteps=2000 with learning_starts=1000, so it saw at most ~1000
gradient steps of signal. A grid sweep over the observation space shows it
selects flap in 3 of 3645 states: it is a never-flap policy. Its 31-step
mean episode length is exactly the free-fall time from spawn to floor.

Two takeaways:

1. Never assume a committed artifact is what its name claims. "Trained
   model" meant "smoke-test output". Measuring first was the right call.
2. The bar for "beats the prior model" is trivially low (any agent that
   passes one pipe clears it). The meaningful targets are the random
   baseline (also 0.0) and the fixed solved threshold, which must be
   defined independently of both.
