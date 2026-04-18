# DETB Execution Guide

> Updated 2026-04-14. This file is a repo-local resume and execution guide for future engineers working in DETB.

## Purpose

Use this guide when you need to resume work in `IsaacLab Testbed/` and want the shortest path to:

- restore the pinned environment
- run the supported DETB workflows
- verify expected artifacts
- understand the current verified state
- pick up the next engineering steps without re-reading the whole repo

This is not the public product overview. Treat `README.md`, `docs/bootstrap.md`, `docs/cli.md`, and `AGENTS.md` as the source of truth for project scope, architecture rules, and user-facing workflow details.

## Current Verified State

### Pinned baseline

- Isaac Sim: `5.1.0`
- Isaac Lab checkout: `../IsaacLab-5.1`
- Isaac Lab version: `2.3.0`
- Conda environment: `isaaclab51`
- Default backend: `mock`
- Default device: `cuda:0`
- Default GPU index: `0`

### Machine snapshot

Use the run manifest to confirm the detected GPU and driver on the current workstation. DETB records that metadata automatically for each run, so the manifest is the source of truth when the workstation changes.

### Workflow status snapshot

As of 2026-04-14:

- The stable v1 operator contract is `train`, `evaluate`, `visualize`, `bundle-artifacts`, and `generate-requirements`.
- `mock` backend remains the stable contract baseline for fast local validation.
- Real Isaac Lab smoke `train`, `evaluate`, and `visualize` flows are verified on the pinned runtime for the baseline operator path.
- `train-gui`, `sweep`, `sensor-eval`, `terrain-eval`, `failure-eval`, and `tune` remain experimental and are not part of the v1 support contract.

Treat any machine-specific verification details as dated snapshots, not invariants.

## Environment Setup

```powershell
conda activate isaaclab51
cd "C:\Dev Projects\NVIDIA Omniverse\IsaacLab Testbed"
python -m pip install -e .[dev]
python -m pip install -e source/detb_lab
```

For engineering sessions, prefer running DETB through the module entry point:

```powershell
python -m detb.cli --help
python -m pytest -q
```

The console script `detb` is also installed by the package, but `python -m detb.cli` keeps the invoked entry point explicit during development.

## Fast Paths

### 1. Contract smoke path with the default mock backend

Use this first when validating repository behavior without depending on Isaac runtime startup:

```powershell
python -m detb.cli train
python -m detb.cli evaluate
python -m pytest -q
```

Expected outcome:

- run directories land under `outputs/<command>/<run_id>/`
- each run writes `resolved_config.yaml`, `run_manifest.json`, and `artifact_registry.json`
- train writes reviewable training artifacts
- evaluate writes machine-readable metrics and summary output

### 2. Baseline operator path

Use this to exercise the concrete v1 support contract:

```powershell
python -m detb.cli train
python -m detb.cli evaluate
python -m detb.cli visualize --set visualization.execute=true
python -m detb.cli bundle-artifacts --source-dir outputs/evaluate/<run_id>
python -m detb.cli generate-requirements --source-dir outputs/evaluate/<run_id>
```

Expected outcome:

- `train` and `evaluate` write the standard DETB run directory bundle
- `visualize` writes playback diagnostics and optional video artifacts
- `bundle-artifacts` rebuilds the summary from an existing run directory
- `generate-requirements` emits a candidate requirement ledger from stored evidence

### 3. Minimal real Isaac Lab smoke path

Use this when you need to validate the real subprocess bridge while keeping runtime small:

```powershell
python -m detb.cli train --set execution.backend=isaaclab --set execution.output_root=outputs_smoke --set execution.num_envs=4 --set execution.train_max_iterations=1 --set execution.seeds=[11]
python -m detb.cli evaluate --set execution.backend=isaaclab --set execution.output_root=outputs_smoke --set execution.eval_episodes=2 --set execution.seeds=[11]
```

Expected outcome:

- DETB run artifacts land under `outputs_smoke/train/<run_id>/` and `outputs_smoke/evaluate/<run_id>/`
- Isaac-native logs land under `outputs/isaaclab_logs/rsl_rl/...`
- train copies a real checkpoint and resolved Isaac config snapshots into the DETB run directory
- evaluate writes episode and aggregate evidence plus Isaac run metadata

### 4. GUI preview path

Use this when you want to inspect the exact Isaac Lab launch command before opening the simulator:

```powershell
python -m detb.cli visualize
```

Expected outcome:

- no Isaac window opens
- DETB writes launch-spec JSON files into the run directory
- the exact command, task selection, and checkpoint choice remain reviewable before execution

### 5. GUI execute path

Use this only when you want live simulator playback or training:

```powershell
python -m detb.cli visualize --set visualization.execute=true
```

On Windows, DETB now injects `--/app/vulkan=false` into GUI Isaac launches by default. The launch spec records that Kit argument so the renderer choice stays explicit and reviewable.

Useful overrides:

- `--set task=flat_walk_stability`
- `--set task=flat_walk_simple_actuator --set robot=anymal_c_simple_actuator`
- `--set visualization.video=true`
- `--set visualization.load_run=<run_name>`
- `--set visualization.checkpoint=<path>`

### Experimental commands

The following commands are available but are outside the v1 support contract:

```powershell
python -m detb.cli train-gui
python -m detb.cli train-gui --set visualization.train_execute=true
python -m detb.cli sweep
python -m detb.cli sensor-eval
python -m detb.cli terrain-eval
python -m detb.cli failure-eval
python -m detb.cli tune
```

## Artifact Expectations

### Train

For every train run, expect at minimum:

- `resolved_config.yaml`
- `run_manifest.json`
- `artifact_registry.json`
- `training_summary.json`
- `training_reward_curve.csv`
- `training_reward_curve.svg` when SVG logging is enabled
- a checkpoint

With `execution.backend=mock`, the checkpoint is synthetic. With `execution.backend=isaaclab`, the checkpoint is copied from the real Isaac output into the DETB run directory.

### Evaluate

For every evaluate run, expect at minimum:

- `resolved_config.yaml`
- `run_manifest.json`
- `artifact_registry.json`
- `episode_metrics.csv`
- `aggregate_metrics.csv`
- `aggregate_metrics.json`
- `summary.md`
- overview plot artifacts when enabled by the pipeline

With `execution.backend=isaaclab`, DETB also records per-seed Isaac launch metadata and copies the evaluated checkpoint into the DETB run directory.

### Visualize

Preview mode writes a launch spec. Executed visualize runs should also capture playback diagnostics such as:

- `visualize_command.json`
- `isaac_play_result.json`
- `playback_telemetry.csv`
- `isaac_play_stdout.log`
- `isaac_play_stderr.log`
- `isaac_play_debug.log`
- `summary.md`
- `videos/play/*.mp4` when video capture is enabled

### Train GUI

Preview mode writes `train_gui_command.json`. Executed runs should also preserve the corresponding stdout/stderr and any video artifacts requested through the visualization config.

## Task And Robot Selection

Useful task selectors:

- `task=flat_walk`
- `task=flat_walk_stability`
- `task=flat_walk_simple_actuator`

Useful robot selectors:

- `robot=anymal_c`
- `robot=anymal_c_simple_actuator`

The simple-actuator path should stay paired:

```powershell
python -m detb.cli train --set task=flat_walk_simple_actuator --set robot=anymal_c_simple_actuator
python -m detb.cli evaluate --set task=flat_walk_simple_actuator --set robot=anymal_c_simple_actuator
```

## How To Sanity-Check The Repo Quickly

When resuming after a gap, use this order:

1. Activate `isaaclab51` and reinstall editable packages if needed.
2. Run `python -m pytest -q`.
3. Run a `mock` train/evaluate pair.
4. Inspect the latest run directories under `outputs/`.
5. If real backend work is relevant, run the minimal Isaac Lab smoke train/evaluate pair into `outputs_smoke/`.
6. Inspect `outputs/isaaclab_logs/rsl_rl/` for the matching real run logs.

If the repo fails before step 5, fix the contract path first. The `mock` backend is the reproducibility baseline and should remain stable.

## Current Engineering Priorities

The repo is past the original subprocess-timeout blocker. The next useful work is keeping the v1 contract stable while broadening validation in a controlled way, not re-investigating the old March 17 startup issue.

Priorities:

1. Keep the v1 path and artifact contracts stable as the regression baseline.
2. Keep the `mock` path stable for fast local validation.
3. Expand verified real-backend coverage only after the baseline operator path remains stable.
4. Keep simulator-specific behavior behind the backend and script adapters rather than leaking imports into analysis code.
5. Add or update tests whenever command behavior, artifacts, or thresholds change.
6. Update docs whenever baseline versions, verified status, or workflow expectations change.

## Working Rules

- Read local config and docs before changing behavior.
- Preserve the pinned baseline unless the task is explicitly about updating it.
- Keep interfaces thin and explicit between DETB orchestration and Isaac-native scripts.
- Prefer `json`, `csv`, and `yaml` artifacts unless heavier formats are justified and tested.
- Treat generated requirements as candidate requirements until a human approves them.

## Source References

Use these files first when this guide is not enough:

- `AGENTS.md`
- `README.md`
- `docs/bootstrap.md`
- `docs/cli.md`
- `docs/modules.md`
- `docs/reproducibility.md`

## Maintenance Note

Update this file whenever any of the following change:

- pinned baseline versions
- default execution backend or runtime paths
- command surface or recommended workflow order
- verified test snapshot
- verified real-backend status

Do not let this guide drift into a workspace scratchpad or multi-project handoff. Keep it specific to DETB and explicit about what is a stable rule versus a dated verification snapshot.
