# ADR 0005: Evaluation protocol and the definition of solved

Date: 2026-06-12
Status: accepted

## Context

There is no oracle: "never dies" is the ceiling and it is reachable, so an
uncapped episode never ends. Headline claims need a fixed protocol, a step
cap, multiple seeds, and a target defined before results exist. The
measured starting line is score 0.0 for both the prior model and the
random baseline.

## Decision

The fixed protocol behind every headline number:

- 100 episodes; episode i resets with seed 10000 + i, so all policies face
  the same fixed pipe layouts and comparisons are paired.
- Step cap 10000 per episode (about 275 pipes at one pipe per ~36 steps);
  capped episodes are truncations, counted and reported.
- Deterministic (greedy) and stochastic (epsilon-greedy at the saved final
  exploration rate) evaluations are run and reported separately.
- Mean, standard deviation, median, min, max of game score are reported,
  never a single run.

Solved is defined as: for every one of 3 training seeds (0, 1, 2), the
best checkpoint's deterministic evaluation has mean score >= 100 and
median score >= 100 under this protocol.

Reference points: random policy 0.0 mean, prior shipped model 0.0 mean
(both measured, ledger and journal entry 001).

## Rationale

- Score >= 100 means roughly 3600 consecutive frames without a mistake,
  far beyond memorized luck, while remaining clearly measurable under the
  10k step cap.
- Median alongside mean guards against a fat tail of early deaths being
  averaged away by capped runs.
- Per-seed (not pooled) thresholds make one bad seed fail the claim, which
  is the point of multi-seed evaluation.

## Consequences

- The cap makes "infinite" scores finite by construction; we report the
  capped-episode count so a perfect agent reads as 100/100 capped rather
  than a fake-precise mean.
- Anyone can re-run the exact protocol from the config file; the protocol
  fields are part of the config hash.
