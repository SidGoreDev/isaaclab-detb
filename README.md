# DETB

DETB is a simulation-first design evaluation scaffold for quadruped studies built on top of the pinned Isaac Sim and Isaac Lab baseline in the parent `NVIDIA Omniverse` workspace.

This repository now provides:

- A stable package layout for configs, execution, outputs, docs, and tests.
- Hydra-based configuration groups for robot, task, terrain, sensor, fault, study, and analysis settings.
- CLI entry points for baseline runs and the first DETB analysis modules.
- Run manifests, artifact registries, episode metrics, aggregate metrics, reports, and candidate requirements.
- A `mock` backend for local smoke testing plus an `isaaclab` backend seam for the pinned simulator stack.

## Quick Start

```powershell
cd "C:\Dev Projects\NVIDIA Omniverse\IsaacLab Testbed"
python -m pip install -e .[dev]
python -m detb.cli train
python -m detb.cli evaluate
python -m detb.cli sweep --set study=sweep
python -m detb.cli generate-requirements --source-dir outputs\evaluate\<run_id>
```

## Pinned Baseline

- Isaac Sim: `5.1.0`
- Isaac Lab checkout: `..\IsaacLab-5.1`
- Isaac Lab version file: `2.3.0`
- Conda environment: `isaaclab51`

`mock` is the default execution backend so the repository can be tested without booting Isaac Sim. Switch to the simulator backend with `--set execution.backend=isaaclab` after wiring the Isaac adapter into the local environment.

## Layout

```text
configs/      Hydra configs and pinned defaults
detb/         Core package and backend seams
docs/         Bootstrap and architecture notes
scripts/      Convenience launchers
tests/        Unit and integration tests
outputs/      Generated study artifacts
```
