# AGENTS.md

## Purpose

This repository is the standalone home for DETB: the Isaac Lab Design Evaluation Test Bed. The project goal is not policy shipping or sim-to-real claims. The goal is reproducible engineering evidence for quadruped design evaluation, sensor trade studies, terrain generalization analysis, failure exploration, and candidate requirement generation.

## Core Principles

- Preserve reproducibility over convenience.
- Keep simulator-specific code behind thin adapters.
- Prefer explicit configs and recorded artifacts over hidden defaults.
- Treat every generated requirement as a candidate until a human approves it.
- Optimize for a public, reviewable GitHub repository, not a one-off experiment folder.

## Pinned Baseline

- Isaac Sim baseline: `5.1.0`
- Isaac Lab baseline: local checkout at `../IsaacLab-5.1`
- Isaac Lab version: `2.3.0`
- Primary environment: `conda activate isaaclab51`
- Default local execution backend may be `mock` when the simulator is not needed for smoke tests.

If these versions change, update:

- `README.md`
- `docs/bootstrap.md`
- Hydra defaults in `configs/`
- any run manifest version capture logic

## Repository Expectations

Top-level intent:

- `configs/`: Hydra configuration groups and pinned defaults
- `detb/`: core package, pipeline, data contracts, backend seams
- `docs/`: bootstrap, architecture, and workflow notes
- `scripts/`: convenience entry points only
- `tests/`: unit and integration coverage
- `outputs/`: generated artifacts only, never committed unless intentionally publishing example outputs

Do not turn the repo into a dump of logs, notebooks, or ad hoc experiment files.

## Architecture Rules

- Keep Isaac Sim / Isaac Lab dependencies isolated behind a backend or adapter layer.
- Do not spread simulator imports through analysis, reporting, or metrics code.
- Analysis modules must work from stored run artifacts wherever possible.
- Prefer deterministic, serializable data contracts:
  - run manifest
  - episode metrics
  - aggregate metrics
  - requirement ledger
  - artifact registry
- Favor simple file formats for portability: `json`, `csv`, `yaml`, and only add heavier stores like `parquet` when the implementation is justified and tested.

## Config Rules

- All user-facing experiment choices belong in Hydra configs, not hardcoded constants.
- Keep config groups stable and discoverable:
  - `robot`
  - `task`
  - `terrain`
  - `sensor`
  - `fault`
  - `study`
  - `analysis`
- New config fields must have a clear reason and map to one of:
  - simulator execution
  - experiment design
  - evidence generation
  - artifact export
- Avoid duplicating the same default in multiple config files unless required by Hydra composition.

## Code Style

- Default to ASCII.
- Keep modules small and purpose-specific.
- Prefer dataclasses or similarly explicit schemas for persisted records.
- Add comments only where intent is not obvious from the code.
- Avoid clever abstractions. This repo should be easy to inspect and extend.
- Name commands and artifacts for engineering clarity, not novelty.

## Output and Evidence Rules

Every major command should emit enough information to reconstruct the result later. Minimum expectation:

- resolved config snapshot
- run manifest
- machine-readable metrics
- reviewable summary output
- artifact registry

Never emit a candidate requirement from a single weak metric without clear evidence links.

## Testing Rules

- Add or update tests for every new command, schema, or metric calculation.
- Keep a `mock` path available for fast local validation.
- Prefer tests that validate output contracts and artifact layout, not only internal helpers.
- Regression tests should protect:
  - config loading
  - manifest generation
  - aggregate metric calculation
  - requirement generation thresholds

## GitHub and Collaboration

- Keep the README useful to first-time visitors:
  - what DETB is
  - what it is not
  - pinned baseline
  - quick start
  - available modules
- Add community-health files as the repo matures:
  - issue templates
  - PR template
  - `CODEOWNERS`
  - `SECURITY.md`
  - CI workflow
- Prefer small, reviewable commits with direct messages.
- Do not commit secrets, local credentials, or large transient outputs.

## What To Avoid

- Do not claim field performance, certification, or sim-to-real validation.
- Do not couple report generation directly to Isaac runtime imports.
- Do not bury key assumptions in notebooks or terminal logs.
- Do not introduce unpinned dependency changes casually.
- Do not replace clear evidence with screenshots or prose-only summaries.

## Agent Guidance

When making changes:

- read the local config and docs first
- preserve the pinned baseline unless explicitly changing it
- keep interfaces thin and explicit
- add tests with the code change
- update docs when behavior or workflow changes

If a task is blocked by the simulator runtime, keep the repo moving with mockable seams, contract-first implementations, and testable artifact generation.
