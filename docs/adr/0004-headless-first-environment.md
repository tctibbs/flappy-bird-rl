# ADR 0004: Headless-first environment design

Date: 2026-06-12
Status: accepted

## Context

The original environment called pygame.display.set_mode unconditionally
and loaded sounds even when not rendering, so training needed a display
(or SDL dummy drivers) and wasted work. Collision is pixel-perfect via
sprite alpha masks, which the original obtained through convert_alpha(),
a display-dependent call.

## Decision

Simulation never touches the display or audio device. Render modes:

- None (default): no pygame display or mixer init at all. Sounds are a
  silent null implementation. Sprites are converted to per-pixel alpha by
  blitting onto an SRCALPHA surface, which needs no display.
- rgb_array: frames drawn to an offscreen Surface, still display-free, so
  videos record on headless machines.
- human: real window, FPS cap, and sound (falling back to silence if the
  mixer fails).

## Rationale

- The SRCALPHA conversion was verified to produce identical hit masks to
  convert_alpha for every sprite class (bird, pipe, base, digits), so
  headless dynamics are bit-for-bit the same as rendered dynamics. The
  pre-refactor baseline reproduced exactly (identical per-episode scores
  and lengths) under the new environment.
- No SDL_VIDEODRIVER environment juggling on any platform, including the
  owner's Windows training box and Linux CI.

## Consequences

- Rendering entities keep an optional screen; render paths fetch it via a
  narrowing helper that raises if reached headless.
- Sound effects are injected (real or null), so game logic stays unchanged
  while audio becomes an output concern.
