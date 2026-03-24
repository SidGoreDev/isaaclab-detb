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

## Current Status

The repository supports a complete `mock` backend for fast local validation and verified Isaac Lab execution paths for `train`, `evaluate`, `visualize`, and `train-gui`. Simulator-native task and robot ownership now live under `source/detb_lab`, while `detb/` remains the orchestration and evidence layer. On March 17, 2026, minimal real `train` and `evaluate` smoke runs completed through the baseline DETB ANYmal-C family, the DETB stability-focused family, and the first divergent DETB robot/profile family using the pinned `isaaclab51` Python runtime. On March 24, 2026, Windows GUI launch verification was re-established by routing GUI Kit startup through `--/app/vulkan=false`, which avoids the Vulkan startup crash observed on this machine and is recorded directly in the launch spec.

## Pinned Baseline

- Isaac Sim: `5.1.0`
- Isaac Lab checkout: `..\IsaacLab-5.1`
- Isaac Lab version file: `2.3.0`
- Conda environment: `isaaclab51`
- Default device: `cuda:0`
- Default GPU index: `0`

On March 24, 2026, GPU `0` on this machine resolved to `NVIDIA GeForce RTX 5080 Laptop GPU` with driver `595.79`. DETB records the detected GPU and driver automatically in the run manifest.

## Session Setup

For low-friction sessions, see `docs/permissions.md`. The practical target is one persistent approval for `python -m detb.cli` plus one for `python -m pytest`.

## Quick Start

```powershell
cd "C:\Dev Projects\NVIDIA Omniverse\IsaacLab Testbed"
python -m pip install -e .[dev]
python -m pip install -e source/detb_lab
detb train
detb evaluate
detb tune --set study=sweep
detb visualize --set visualization.execute=true
detb train-gui --set visualization.train_execute=true
```

## GUI And Tuning Workflows

Use dry-run launch specs first so the exact Isaac Lab command is reviewable before anything opens:

```powershell
detb visualize
detb train-gui
```

Then execute the pinned GUI path when you are ready to inspect simulator behavior:

```powershell
detb visualize --set visualization.execute=true
detb train-gui --set visualization.train_execute=true
```

For the baseline `task=flat_walk` path, `visualization.use_pretrained_checkpoint=true` now resolves to the upstream Isaac Lab ANYmal-C pretrained checkpoint. DETB-only task variants without a published upstream equivalent fail fast instead of silently replaying the latest local smoke checkpoint.

Tune against the design sweep and adjust objective weights or thresholds directly from the CLI:

```powershell
detb tune --set study=sweep
detb tune --set study=sweep --set objective.terrain_weight=0.5 --set objective.target_tgs=0.8
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

## Commands

| Command | Purpose |
|---------|---------|
| `detb train` | Produce a manifest, training summary, reward curve artifacts, and either a synthetic checkpoint (`mock`) or copied real checkpoint/config snapshots (`isaaclab`). |
| `detb evaluate` | Produce episode metrics, aggregate metrics, overview plot, summary markdown, and Isaac run metadata when `execution.backend=isaaclab`. |
| `detb sweep` | Run staged design-point screening using the configured study points. |
| `detb tune` | Rank design points using objective weights and target thresholds. |
| `detb sensor-eval` | Compare configured sensor profiles under matched conditions. |
| `detb terrain-eval` | Run the configured terrain battery and compute TGS. |
| `detb failure-eval` | Sweep fault severity and detect the first critical threshold. |
| `detb visualize` | Launch the pinned Isaac Lab GUI playback path or emit a launch spec when `visualization.execute=false`. |
| `detb train-gui` | Launch the pinned Isaac Lab GUI training path or emit a launch spec when `visualization.train_execute=false`. |
| `detb generate-requirements --source-dir ...` | Generate a candidate requirement ledger from stored artifacts. |

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

1. Keep the mock-backed pipeline stable as the contract baseline.
2. Expand DETB-owned task families and robot configs under `source/detb_lab`.
3. Use GUI launch specs and Isaac Lab playback/training to inspect progress visually.
4. Tune objective weights and target thresholds until studies produce useful outputs.
5. Complete terrain taxonomy and generator mapping from Isaac Lab best practices.
6. Expand the verified real Isaac Lab backend from smoke runs into broader study coverage and requirement generation workflows.


