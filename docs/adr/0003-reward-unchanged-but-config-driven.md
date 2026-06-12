# ADR 0003: Reward unchanged, but config-driven

Date: 2026-06-12
Status: accepted

## Context

The original reward (+1 per step alive, -100 on death) was hardcoded in
the environment while dead config fields (reward_alive, reward_death)
suggested otherwise. Reward shaping (for example a gap-centering term) was
an option for sample efficiency.

## Decision

Keep +1 alive / -100 death as defaults, wired through typed config fields
that the environment actually reads. No shaping until a measured result
justifies it.

## Rationale

- Survival reward is fully aligned with game score here: the only way to
  live longer is to pass pipes. Shaping adds a hyperparameter surface and
  reward-hacking risk without a demonstrated need.
- Any shaping claim should be an ablation: same config, one field changed,
  compared on the fixed protocol. Making the reward configurable is the
  prerequisite; changing it is not.

## Consequences

- reward_alive and reward_death are part of the config hash, so a shaped
  variant would be a distinct, comparable config.
- If training proves slow or unstable, a gap-centering term can be added
  as a new typed field and ablated honestly.
