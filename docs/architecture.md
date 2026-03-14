# DETB Architecture

## Layers

- Config layer: Hydra YAML groups in `configs/`
- Execution layer: backend seam in `detb.backends`
- Data layer: manifests, metrics, requirements, and artifact registry in `detb.io`
- Output layer: markdown, CSV, JSON, and SVG artifacts in `detb.artifacts`

## Backend Strategy

- `mock`: deterministic synthetic data for CLI smoke tests and repository regression checks
- `isaaclab`: thin adapter seam reserved for the pinned Isaac Lab stack

## Output Contract

Each command writes to `outputs/<command>/<run_id>/` and includes:

- `run_manifest.json`
- `resolved_config.yaml`
- `artifact_registry.json`
- command-specific CSV and JSON outputs
- report-ready markdown
- simple SVG plots

This keeps the evidence chain stable while simulator integration is built out behind the same interfaces.
