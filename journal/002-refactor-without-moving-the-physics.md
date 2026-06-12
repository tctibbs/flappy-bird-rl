# 002: Refactoring without moving the physics

Date: 2026-06-12
Git SHA: 7cb7921 (pipeline rebuild)

## What happened

The whole pipeline was rebuilt (typed config, ledger, headless env,
fixed protocol) while keeping the game dynamics bit-for-bit identical.
The proof is paired evaluation: the baseline policies from entry 001 were
re-run under the new environment with the same episode seeds and produced
identical per-episode score and length lists (random baseline mean length
exactly 72.0) to the pre-refactor measurement.

## Lessons

1. The dangerous part of going headless was invisible: sprites here are
   palettized PNGs whose transparency lives in a colorkey. Loading them
   without conversion silently turns pixel-perfect hit masks into full
   rectangles, changing collisions. The fix (blit onto an SRCALPHA
   surface) was verified mask-by-mask against convert_alpha before any
   code depended on it. Measure the equivalence, do not assume it.
2. Replacing global random.seed with a per-env random.Random preserved
   pipe sequences exactly, because both are the same Mersenne Twister
   stream. That equivalence is what made the paired verification clean.
3. Measuring the baseline before refactoring made this checkable at all.
   If the refactor had come first, a dynamics change would have been
   unfalsifiable.
