# ADR 0007: Bounded reward scale for the headline config

Date: 2026-06-12
Status: accepted (amends ADR 0003)

## Context

ADR 0003 kept the original +1 alive / -100 death reward as the default and
required any change to be justified by measurement. The measurement
happened: with the original scale, DQN's best checkpoint reached mean
score 1.77 (300k steps, default hyperparameters) and 0.67 (1M steps,
lr 1e-4, 256x256 network). A paired screening run identical except for
reward scale (+0.1 alive / -1.0 death, 64x64 network) reached mean 255.00
with all 100 protocol episodes hitting the step cap: perfect play. See
journal entry 003 and ledger rows with config hashes 6803fbd4b3bb and
fa504a6b676d.

## Decision

The headline config (configs/final.yaml) uses reward_alive = 0.1 and
reward_death = -1.0. This is a scale change only: the optimal policy
(survive as long as possible) is unchanged, no shaping terms are added.

## Rationale

- A 100x reward discontinuity at death dominates the TD targets and
  destabilizes Q-learning, the same reason the original DQN work clips
  rewards to [-1, 1]. The screening result reproduced this textbook
  effect cleanly.
- Game score remains the evaluation metric, so comparisons against the
  random baseline are unaffected by the reward change.

## Consequences

- The reward fields stay typed config fields; the original scale remains
  available in configs/default.yaml for reproduction of the negative
  result.
- ADR 0003's principle stands (no unmeasured reward changes); its default
  values are superseded for the headline config by this measured result.
