# Method and results summary

Date: 2026-06-12. Every number below is a ledger row in
results/ledger.jsonl or a journal-recorded measurement.

## Outcome in plain language

The trained agent plays Flappy Bird perfectly as far as the measurement
ceiling allows: on all 3 training seeds, the best checkpoint passed 255
pipes on every one of 100 fixed evaluation episodes without dying (every
episode hit the 10000-step cap). The random baseline scores 0.00 on the
same episodes. The model that shipped with the repo also scores 0.00; it
turned out to be a 2000-step smoke-test artifact that never flaps, so the
measured starting line was zero, not a working agent. The solved target
(mean and median score >= 100 on every seed) is met with maximal margin.

## Method

- Environment: the repo's Pygame Flappy Bird wrapped as a Gymnasium env.
  Observation is 4 normalized features (bird height, vertical velocity,
  horizontal distance to next pipe, vertical distance to gap center);
  actions are flap / no flap. Simulation is fully decoupled from
  rendering and runs headless with dynamics verified identical to the
  rendered game (paired per-episode equality against the pre-refactor
  baseline).
- Algorithm: Stable-Baselines3 DQN, MLP 64x64, learning rate 1e-4,
  buffer 100k, batch 64, gamma 0.99, target update every 1000 steps,
  epsilon decayed to 0.01 over the first 10 percent of 1M steps.
- Reward: +0.1 per surviving frame, -1.0 on death. This bounded scale was
  the decisive choice: the original +1 / -100 reward kept DQN unstable
  (best mean 1.77 at 300k steps; 0.67 even at 1M steps with a larger
  network), while the bounded scale reached the ceiling. Same lesson as
  DQN's Atari reward clipping. See docs/adr/0007 and journal/003.
- Checkpointing: the live policy oscillates even after reaching perfect
  play, so every 50k steps the model is evaluated on 20 fixed-seed
  episodes and the best-scoring checkpoint is kept. Headline numbers come
  from that checkpoint under the full protocol.
- Evaluation protocol (docs/adr/0005): 100 episodes, episode i seeded
  with 10000 + i so all policies face identical pipe layouts, step cap
  10000, deterministic and stochastic (epsilon 0.01) evaluations reported
  separately. Mean, median, min, max, and capped-episode counts recorded.

## Results (deterministic, best checkpoint, n=100 per row)

| Run | Mean | Median | Min | Max | Capped |
|---|---|---|---|---|---|
| dqn_final seed 0 | 255.00 | 255 | 255 | 255 | 100/100 |
| dqn_final seed 1 | 255.00 | 255 | 255 | 255 | 100/100 |
| dqn_final seed 2 | 255.00 | 255 | 255 | 255 | 100/100 |
| random | 0.00 | 0 | 0 | 0 | 0/100 |
| prior shipped model | 0.00 | 0 | 0 | 0 | 0/100 |

Stochastic evaluation of the same checkpoints scores 13.2 to 18.1: with
1 percent random actions, a single ill-timed flap is fatal, so this
measures the game's fragility under noise rather than policy quality.

Training cost: 277 to 315 seconds per seed for 1M environment steps,
single process, laptop CPU, headless.

## What was learned

1. Measure the starting line first. The committed "trained model" was a
   never-flap policy; without measuring it, every later comparison would
   have rested on an assumption (journal/001).
2. Headless equivalence is a measurable claim. Palettized sprites lose
   their alpha silently without conversion, which would have changed
   collision masks; the fix was verified mask-by-mask and then end-to-end
   by exact reproduction of per-episode baseline results (journal/002).
3. Reward scale dominated everything else. No amount of extra training,
   network capacity, or learning-rate tuning rescued the +1 / -100
   reward within the budgets tried; scaling to +0.1 / -1.0 solved the
   game outright (journal/003).
4. With an unstable learner, checkpoint selection is part of the method,
   not a convenience. The final network at 1M steps is often mediocre;
   the best checkpoint is consistently perfect (journal/004).

## Artifacts

- results/ledger.jsonl: every training and evaluation row, with git SHA,
  config hash, seed, and full config embedded.
- plots/learning_curves.png, plots/final_scores.png.
- videos/best_agent_seed0.mp4: the seed 0 best checkpoint playing (local
  artifact, not committed; regenerate with `uv run flapper video`).
- runs/<run_id>/: per-run config, structured log, monitor.csv, learning
  curve, best and final models, final_eval.json.
- docs/adr/: seven decision records. journal/: four lessons.
