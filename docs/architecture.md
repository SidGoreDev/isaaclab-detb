# DETB Architecture

## Layers

- Config layer: Hydra YAML groups in `configs/`
- Execution layer: backend seam in `detb.backends`
- Simulator-native layer: external Isaac Lab extension package in `source/detb_lab`
- Data layer: manifests, metrics, requirements, and artifact registry in `detb.io`
- Output layer: markdown, CSV, JSON, and SVG artifacts in `detb.artifacts`

## Backend Strategy

- `mock`: deterministic synthetic data for CLI smoke tests and repository regression checks
- `isaaclab`: thin adapter seam for the pinned Isaac Lab stack, with custom DETB task and robot definitions loaded from `source/detb_lab`

## Task Ownership

- Hydra task groups under `configs/task` select DETB-owned registry families rather than reaching straight into stock Isaac Lab IDs.
- The baseline family currently lives under `detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c`.
- The first divergent DETB family currently lives under `detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability`.
- The first divergent DETB robot/profile family currently lives under `detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator`.

## Robot Ownership

- DETB-owned robot configs live under `detb_lab.assets.robots`.
- Task specs in `detb_lab.registry` declare the expected robot asset ID, and DETB rejects mismatched `task` and `robot` selections before launching Isaac.

## Output Contract

Each command writes to `outputs/<command>/<run_id>/` and includes:

- `run_manifest.json`
- `resolved_config.yaml`
- `artifact_registry.json`
- command-specific CSV and JSON outputs
- report-ready markdown
- simple SVG plots

This keeps the evidence chain stable while real simulator coverage expands behind the same interfaces and prevents simulator-native logic from leaking into analysis code.
