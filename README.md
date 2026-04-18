# DETB

DETB is a simulation-first design evaluation toolkit for quadruped systems engineering. It is designed to turn Isaac Lab experiments into reviewable engineering evidence, not deployment claims or sim-to-real guarantees.

## What DETB Is

- A reproducible experiment runner for quadruped design studies.
- A contract-first pipeline for manifests, metrics, and artifact bundles.
- A framework for terrain, sensor, fault, and tuning studies.
- A public-facing GitHub repository intended for technical review.

## What DETB Is Not

- A field deployment stack.
- A certification tool.
- A claim of validated sim-to-real transfer.
- A generic reinforcement learning benchmark zoo.

## V1 Support Contract

The concrete v1 operator path is:

- `train`
- `evaluate`
- `visualize`
- `bundle-artifacts`
- `generate-requirements`

The v1 contract is supported on the `mock` backend for fast local validation, and the real Isaac Lab backend is verified for the core simulator-backed path that feeds those same artifacts.

The following commands are experimental and are not part of the v1 support contract:

- `train-gui`
- `sweep`
- `sensor-eval`
- `terrain-eval`
- `failure-eval`
- `tune`

## Current Status

As of 2026-04-14, the repository supports the v1 operator contract above with a complete `mock` backend for fast local validation and a verified Isaac Lab backend for the core simulator-backed flows. Simulator-native task and robot ownership live under `source/detb_lab`, while `detb/` remains the orchestration and evidence layer. Windows GUI launches through DETB use `--/app/vulkan=false` on this machine so the GUI path stays stable and reviewable in the launch spec.

## Pinned Baseline

- Isaac Sim: `5.1.0`
- Isaac Lab checkout: `..\IsaacLab-5.1`
- Isaac Lab version file: `2.3.0`
- Conda environment: `isaaclab51`
- Default device: `cuda:0`
- Default GPU index: `0`

On 2026-04-14, GPU `0` on this machine resolved to `NVIDIA GeForce RTX 5080 Laptop GPU` with driver `595.79`. DETB records the detected GPU and driver automatically in the run manifest.

## Session Setup

For low-friction sessions, see `docs/permissions.md`. The practical target is one persistent approval for `python -m detb.cli` plus one for `python -m pytest`.

## Quick Start

```powershell
cd "C:\Dev Projects\NVIDIA Omniverse\IsaacLab Testbed"
python -m pip install -e .[dev]
python -m pip install -e source/detb_lab
detb train
detb evaluate
detb visualize --set visualization.execute=true
detb bundle-artifacts --source-dir outputs/evaluate/<run_id>
detb generate-requirements --source-dir outputs/evaluate/<run_id>
```

## Experimental Workflows

The following commands are useful for development, but they are not part of the v1 support contract:

- `train-gui`
- `sweep`
- `sensor-eval`
- `terrain-eval`
- `failure-eval`
- `tune`

Use dry-run launch specs first so the exact Isaac Lab command is reviewable before anything opens:

```powershell
detb visualize
detb visualize --set visualization.execute=true
```

For the baseline `task=flat_walk` path, `visualization.use_pretrained_checkpoint=true` now resolves to the upstream Isaac Lab ANYmal-C pretrained checkpoint. DETB-only task variants without a published upstream equivalent fail fast instead of silently replaying the latest local smoke checkpoint.

The GUI training path remains available for manual inspection, but it is experimental:

```powershell
detb train-gui
detb train-gui --set visualization.train_execute=true
```

Tune and study workflows are also experimental:

```powershell
detb tune --set study=sweep
detb tune --set study=sweep --set objective.terrain_weight=0.5 --set objective.target_tgs=0.8
detb sweep
detb sensor-eval
detb terrain-eval
detb failure-eval
```

Select a DETB-owned task variant directly from Hydra when you want a different Isaac registry family:

```powershell
detb train --set task=flat_walk_stability
detb evaluate --set task=flat_walk_stability
detb visualize --set task=flat_walk_stability --set visualization.execute=true
```

Select the simple-actuator robot/profile path by pairing the matching task and robot groups:

```powershell
detb train --set task=flat_walk_simple_actuator --set robot=anymal_c_simple_actuator
detb evaluate --set task=flat_walk_simple_actuator --set robot=anymal_c_simple_actuator
```

## Supported Commands

| Command | Purpose |
|---------|---------|
| `detb train` | Produce a manifest, training summary, reward curve artifacts, and either a synthetic checkpoint (`mock`) or copied real checkpoint/config snapshots (`isaaclab`). |
| `detb evaluate` | Produce episode metrics, aggregate metrics, overview plot, summary markdown, and Isaac run metadata when `execution.backend=isaaclab`. |
| `detb visualize` | Launch the pinned Isaac Lab GUI playback path or emit a launch spec when `visualization.execute=false`. |
| `detb bundle-artifacts --source-dir ...` | Rebuild the summary and bundle artifacts from an existing run directory. |
| `detb generate-requirements --source-dir ...` | Generate a candidate requirement ledger from stored artifacts. |

## Experimental Commands

| Command | Purpose |
|---------|---------|
| `detb train-gui` | Launch the pinned Isaac Lab GUI training path or emit a launch spec when `visualization.train_execute=false`. |
| `detb sweep` | Run staged design-point screening using the configured study points. |
| `detb sensor-eval` | Compare configured sensor profiles under matched conditions. |
| `detb terrain-eval` | Run the configured terrain battery and compute TGS. |
| `detb failure-eval` | Sweep fault severity and detect the first critical threshold. |
| `detb tune` | Rank design points using objective weights and target thresholds. |

## Artifact Contract

Each major run writes to `outputs/<command>/<run_id>/` and includes:

- `resolved_config.yaml`
- `run_manifest.json`
- `artifact_registry.json`
- machine-readable CSV or JSON outputs
- reviewable markdown summary
- lightweight SVG plots where applicable

GUI commands also emit launch-spec JSON files so the exact Isaac Lab invocation is reviewable before execution. Executed `visualize` runs now persist `isaac_play_result.json`, `playback_telemetry.csv`, playback logs, and any recorded `videos/play/*.mp4` files in the same run directory.

## Repository Layout

```text
configs/      Hydra configs and pinned defaults
detb/         Core package, runtime, backends, and contracts
source/detb_lab/  External Isaac Lab extension package for custom tasks and robot configs
docs/         Bootstrap, architecture, research, and workflow notes
scripts/      Convenience launchers
tests/        Unit and integration tests
outputs/      Generated artifacts (gitignored)
```

## Documentation

- `docs/bootstrap.md`
- `docs/architecture.md`
- `docs/data-contracts.md`
- `docs/cli.md`
- `docs/modules.md`
- `docs/reproducibility.md`
- `docs/terrain-research.md`
- `docs/terrain-design.md`

## Roadmap

1. Keep the v1 operator contract stable across `train`, `evaluate`, `visualize`, `bundle-artifacts`, and `generate-requirements`.
2. Keep the mock-backed pipeline stable as the contract baseline.
3. Expand DETB-owned task families and robot configs under `source/detb_lab`.
4. Keep experimental study commands clearly separated from the v1 support contract.
5. Continue using GUI launch specs and Isaac Lab playback/training to inspect progress visually when needed.
6. Expand the verified real Isaac Lab backend only after the v1 path remains stable.


