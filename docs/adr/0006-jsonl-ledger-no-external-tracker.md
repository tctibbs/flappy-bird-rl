# ADR 0006: JSONL ledger as source of truth, no external tracker

Date: 2026-06-12
Status: accepted

## Context

Results need to survive independent of any tracking service. A Weights
and Biases key was offered as an optional convenience for live curves and
sweeps.

## Decision

A committed, append-only JSONL ledger (results/ledger.jsonl) is the only
source of truth. Each row records run id, kind (train or eval), UTC
timestamp, git SHA, config hash, config name, seed, frames, wall clock,
flat metrics, the model path, and the full config dump. W&B is not wired
in.

## Rationale

- At this scale (a handful of runs, minutes each) a tracker adds a
  dependency, a secret to manage, and an account-bound failure mode, and
  returns little: learning curves already land in each run directory as
  curve.jsonl and are plotted from there.
- Embedding the full config in each row means a ledger row is
  self-contained even if the YAML file or run directory disappears.
- JSONL appends are atomic enough for a single writer and diff cleanly in
  git.

## Consequences

- No live dashboards. Curves are inspected from files or the generated
  plots.
- If sweeps ever grow large, a tracker can be added behind a per-run flag
  without touching the ledger, which remains authoritative.
