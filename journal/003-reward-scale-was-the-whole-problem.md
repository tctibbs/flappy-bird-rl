# 003: Reward scale was the whole problem

Date: 2026-06-12
Ledger rows: config hashes 6803fbd4b3bb (screen_a), fa504a6b676d (screen_b)

## What was tested

The default config (reward +1 alive / -100 death, lr 3e-4, net 256x256,
300k steps) trained to a best-checkpoint mean score of 1.77 with a wildly
oscillating learning curve. Two single-seed screening runs at 1M steps
isolated the suspected causes:

| Run | Reward | LR | Net | Best-checkpoint mean (n=100) |
|---|---|---|---|---|
| screen_a | +0.1 / -1.0 | 1e-4 | 64x64 | 255.00 (100/100 capped, perfect) |
| screen_b | +1.0 / -100.0 | 1e-4 | 256x256 | 0.67 |

Same algorithm, same budget, same protocol. Bounded rewards solved the
game outright; the original scale barely passed a pipe even with the
lower learning rate and bigger network.

## Lessons

1. A -100 death penalty against +1 step rewards puts a 100x discontinuity
   into the TD targets. DQN's Atari recipe clips rewards to [-1, 1] for
   exactly this reason, and this little game reproduces that lesson
   faithfully. Scale rewards before reaching for bigger networks or more
   steps.
2. Even after reaching perfect play, the periodic eval of the live DQN
   oscillates (a later eval at 850k scored 14.4 after an 800k eval of
   255). Best-checkpoint selection against a fixed eval protocol is what
   turns an unstable learner into a reliable artifact.
3. Screening with one seed is for direction, not claims. The headline
   experiment is the same config rerun on 3 seeds with the full protocol.
