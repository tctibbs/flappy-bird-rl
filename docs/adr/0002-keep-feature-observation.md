# ADR 0002: Keep the 4-feature observation

Date: 2026-06-12
Status: accepted

## Context

The observation could stay as 4 normalized features (bird y, bird y
velocity, horizontal distance to next pipe, vertical distance to gap
center) or move to raw pixels with a CNN and frame stacking.

## Decision

Keep the 4-feature observation.

## Rationale

- The features are a sufficient statistic for this game: with the next
  gap's relative position and own velocity, the optimal action is fully
  determined. Pixels add representation learning cost without adding
  information.
- The headline goal is a clean pipeline with a reliably solved game.
  From-pixels learning is an optional secondary axis, explicitly not
  required, and would multiply training cost by orders of magnitude.
- Features keep training fast enough to run multi-seed experiments on a
  laptop CPU, which directly serves the multi-seed evaluation bar.

## Consequences

- The agent only sees the next pipe, so extremely tight consecutive-gap
  transitions are the residual failure mode. Acceptable; measured rather
  than assumed.
- A pixels config remains possible later as a separate observation field
  without changing the protocol or ledger.
