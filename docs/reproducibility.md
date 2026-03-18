# Reproducibility

DETB is built around the assumption that every result should be reconstructable from the artifact bundle.

## Required Artifacts Per Run

- resolved Hydra config snapshot
- run manifest
- artifact registry
- machine-readable result files
- summary markdown

## Reproducibility Rules

- Keep seeds explicit in config and manifest.
- Keep simulator version fields explicit in the manifest.
- Avoid hidden defaults that alter results without surfacing in config.
- Keep analysis code backend-agnostic so stored artifacts can be reprocessed without booting Isaac.
- Keep train and evaluation terrain definitions separate as real simulator-backed study coverage expands.

## Mock Baseline

The `mock` backend is part of the reproducibility strategy. It provides a deterministic contract target for:

- CLI behavior
- artifact naming
- schema stability
- CI validation

The real `isaaclab` backend now follows the same artifact contract for verified minimal `train` and `evaluate` smoke runs rather than inventing a separate path.
