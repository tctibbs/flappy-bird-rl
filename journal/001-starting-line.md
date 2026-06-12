# 001: The starting line is zero

Date: 2026-06-12
Git SHA: 6582c827b0 (unmodified code)
Raw data: results/baseline_pre_refactor.jsonl

## What was measured

The baseline for this project: a random policy evaluated against the
unmodified environment. 100 episodes, episode seeds 10000..10099, step cap
10000, headless (SDL dummy drivers).

| Policy | Score mean | Score median | Score max | Length mean |
|---|---|---|---|---|
| Random (uniform flap 50%) | 0.00 | 0 | 0 | 72 steps |

A random flapper never passes a single pipe on any of the 100 fixed
layouts; it survives about 72 frames on average before colliding. Every
later claim is measured against this number under the same protocol.

## Lesson

Pin the starting line before training anything. The baseline number, the
episode seeds, and the step cap were all fixed and recorded here first, so
every later result is relative to a measured reference rather than an
assumption, and the solved target (ADR 0005) could be defined before any
result existed to bias it.
