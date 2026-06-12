# ADR 0001: Keep DQN on Stable-Baselines3

Date: 2026-06-12
Status: accepted

## Context

The algorithm choice was left open: keep the existing SB3 DQN, switch to
PPO, or leave SB3 entirely. The goal is a rigorous, reproducible pipeline,
not algorithmic novelty. Flappy Bird with a 4-feature observation is a
small, fully observable, discrete-action problem that any standard method
solves.

## Decision

Keep DQN on Stable-Baselines3, with hyperparameters as typed config fields.

## Rationale

- SB3 is already a dependency and its DQN is a well-tested reference
  implementation. The pipeline's value (config hashing, ledger, fixed
  protocol) is algorithm-agnostic, so the simplest credible choice wins.
- DQN is off-policy and replay-based, which suits this cheap, serial,
  headless environment. PPO's main advantage (stable on-policy updates
  with many parallel envs) buys little here.
- The prior model in the repo is a DQN, so the starting line and the new
  result stay directly comparable.

## Consequences

- Stochastic evaluation of a DQN means epsilon-greedy at the saved final
  exploration rate, not a learned stochastic policy. Reported as such.
- If sample-efficiency sweeps become a goal, PPO or vectorized DQN can be
  added as a new config without touching the protocol or ledger.
