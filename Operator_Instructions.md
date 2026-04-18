# Operator Instructions

## Purpose

This file is the quick operator runbook for DETB: the Isaac Lab Design Evaluation Test Bed. Use it when you need to run DETB workflows, inspect outputs, and confirm where evidence artifacts are stored.

For deeper environment and handoff detail, also see `README.md` and `EXECUTION_GUIDE.md`.

## V1 Support Contract

The baseline operator path is:

- `train`
- `evaluate`
- `visualize`
- `bundle-artifacts`
- `generate-requirements`

These commands are part of the v1 support contract.

The following commands are experimental and are not part of the v1 support contract:

- `train-gui`
- `sweep`
- `sensor-eval`
- `terrain-eval`
- `failure-eval`
- `tune`

## Baseline

- Isaac Sim: `5.1.0`
- Isaac Lab checkout: `../IsaacLab-5.1`
- Isaac Lab version: `2.3.0`
- Primary environment: `conda activate isaaclab51`
- Default local smoke backend: `mock`

## Main Entry Point

Run DETB with either:

```powershell
python -m detb.cli <command>
```

or, if installed:

```powershell
detb <command>
```

## Backend Modes

- `execution.backend=mock`
  Fast local validation path. No simulator required.
- `execution.backend=isaaclab`
  Real Isaac Lab execution path.

## Core Commands

### `train`

Runs a training job and stores a DETB run directory under `outputs/train/<run_id>/`.

Typical usage:

```powershell
python -m detb.cli train
python -m detb.cli train --set execution.backend=isaaclab
```

Expected outputs:

- `resolved_config.yaml`
- `run_manifest.json`
- `artifact_registry.json`
- `summary.md`
- Isaac-backed runs also write:
  - `isaac_train_command.json`
  - `isaac_train_result.json`
  - copied checkpoint
  - copied Isaac config snapshots such as `isaac_env.yaml` and `isaac_agent.yaml`

### `evaluate`

Runs evaluation episodes for a checkpoint and stores results under `outputs/evaluate/<run_id>/`.

Typical usage:

```powershell
python -m detb.cli evaluate
python -m detb.cli evaluate --set execution.backend=isaaclab
```

Expected outputs:

- `resolved_config.yaml`
- `run_manifest.json`
- `artifact_registry.json`
- `summary.md`
- Isaac-backed runs also write:
  - `isaac_eval_runs.json`
  - per-seed result JSON files
  - per-seed stdout and stderr logs
  - copied evaluated checkpoint

### `visualize`

Runs playback for a trained policy. Preview mode writes the launch spec only. Execute mode launches Isaac playback and stores logs, telemetry, and playback diagnostics under `outputs/visualize/<run_id>/`.

Typical usage:

```powershell
python -m detb.cli visualize
python -m detb.cli visualize --set execution.backend=isaaclab --set visualization.execute=true
```

Expected outputs in execute mode:

- `resolved_config.yaml`
- `run_manifest.json`
- `artifact_registry.json`
- `summary.md`
- `visualize_command.json`
- `isaac_play_result.json`
- `isaac_play_runs.json`
- `playback_telemetry.csv`
- `isaac_play_stdout.log`
- `isaac_play_stderr.log`
- `isaac_play_debug.log`
- copied checkpoint such as `baseline_policy.pt`

Windows note:

- GUI Isaac launches currently inject `--/app/vulkan=false` to keep the simulator stable on this machine.

### `bundle-artifacts`

Rebuilds the summary for an existing run directory. Operates in place on the source run directory.

Typical usage:

```powershell
python -m detb.cli bundle-artifacts --source-dir outputs/evaluate/<run_id>
```

Expected outputs in the source run directory:

- refreshed `summary.md` rebuilt from the stored aggregate metrics

### `generate-requirements`

Creates a candidate requirement ledger from prior run artifacts. Operates in place on the source run directory.

Typical usage:

```powershell
python -m detb.cli generate-requirements --source-dir outputs/evaluate/<run_id>
```

Expected outputs in the source run directory:

- `requirement_ledger.csv`
- `requirement_ledger.json`
- `candidate_requirements.md`
- refreshed `artifact_registry.json`, `run_manifest.json`, and `resolved_config.yaml`

Treat every emitted record as a candidate requirement until a human reviewer promotes it.

## Experimental Commands

These commands are useful for development, but they are not part of the v1 support contract:

- `train-gui`
- `sweep`
- `sensor-eval`
- `terrain-eval`
- `failure-eval`
- `tune`

Treat generated requirements as candidate requirements until a human reviews them. `generate-requirements` is part of the v1 operator path, but the output is still a candidate requirement ledger rather than an approved requirement set.

## What To Check First After A Run

For any run directory, inspect these first:

- `summary.md` for the operator-readable outcome
- `run_manifest.json` for config, registry IDs, backend, and runtime metadata
- `artifact_registry.json` for the full artifact list
- command JSON for the exact launch spec
- stdout, stderr, and debug logs for simulator-backed runs

For playback specifically, also inspect:

- `isaac_play_result.json`
- `playback_telemetry.csv`

## Common Operator Patterns

Fast local smoke:

```powershell
python -m detb.cli train
python -m detb.cli evaluate
```

Real simulator smoke:

```powershell
python -m detb.cli train --set execution.backend=isaaclab
python -m detb.cli evaluate --set execution.backend=isaaclab
```

Preview launch specs without running Isaac:

```powershell
python -m detb.cli visualize
python -m detb.cli train-gui
```

Launch live GUI workflows:

```powershell
python -m detb.cli visualize --set execution.backend=isaaclab --set visualization.execute=true
python -m detb.cli train-gui --set execution.backend=isaaclab --set visualization.train_execute=true
```

## Key Code Paths

- CLI: `detb/cli.py`
- Pipeline orchestration: `detb/pipeline.py`
- Isaac backend seam: `detb/backends/isaaclab_backend.py`
- Runtime/task wiring: `source/detb_lab/detb_lab/runtime.py`
- Execution guide: `EXECUTION_GUIDE.md`

## Current Verified Snapshot

As of 2026-04-14 on this machine:

- the v1 operator path is in active use and revalidated locally
- `mock` smoke workflows are available
- real Isaac `train` smoke is working
- real Isaac `evaluate` smoke is working
- real Isaac `visualize` execute is working
- `bundle-artifacts` and `generate-requirements` operate on stored run directories

Update this file if the pinned baseline, command surface, or verified workflow status changes.
