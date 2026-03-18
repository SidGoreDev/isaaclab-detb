# CLI Reference

## Config Surface

All commands load the Hydra base config from `configs/base.yaml` unless overridden.

Useful flags:

- `--config-name <name>`
- `--config-dir <path>`
- `--set key=value`
- `--source-dir <run_dir>` for summary rebuild and requirement generation

## Commands

### `detb train`

Creates a new run directory with a manifest, config snapshot, training summary, reward curve CSV, and reward curve SVG. With `execution.backend=mock` it writes a synthetic checkpoint. With `execution.backend=isaaclab` it launches the DETB-owned Isaac Lab runner and copies the real checkpoint plus resolved Isaac config snapshots into the DETB run directory.

### `detb evaluate`

Creates a new run directory with episode metrics, aggregate metrics, overview SVG, and summary markdown. With `execution.backend=isaaclab` it also records per-seed Isaac launch metadata and copies the evaluated checkpoint into the DETB run directory.

### `detb sweep`

Runs the configured staged screening points and produces DOE-style comparison artifacts.

### `detb tune`

Ranks design points using objective weights and target thresholds from `configs/base.yaml`. Outputs a tuning summary plus machine-readable candidate ranking.

Useful overrides:

- `--set study=sweep` to use the multi-point design screen.
- `--set objective.success_weight=<value>` to change ranking pressure on success.
- `--set objective.terrain_weight=<value>` to prioritize terrain generalization more heavily.
- `--set objective.target_tgs=<value>` or `--set objective.target_failure_threshold=<value>` to tighten outcome gates.

### `detb sensor-eval`

Compares all configured sensor profiles and recommends a minimum viable profile by ROI score.

### `detb terrain-eval`

Runs the configured terrain battery, writes per-terrain success results, and computes Terrain Generalization Score.

### `detb failure-eval`

Runs the configured fault sweep and detects the first critical severity threshold where success falls below 50 percent and confidence intervals clear that threshold.

### `detb visualize`

Builds the Isaac Lab playback command using the configured `execution.isaaclab_python` interpreter against the pinned `..\IsaacLab-5.1` checkout. By default it emits a launch spec without execution. Set `--set visualization.execute=true` to actually launch the GUI playback path.

When executed, the run directory records playback diagnostics instead of only a video:

- `visualize_command.json`
- `isaac_play_result.json`
- `playback_telemetry.csv`
- `isaac_play_stdout.log`
- `isaac_play_stderr.log`
- `isaac_play_debug.log`
- `summary.md`
- `videos/play/*.mp4` when `visualization.video=true`

Useful overrides:

- `--set visualization.load_run=<run_name>` to replay a specific Isaac Lab run.
- `--set visualization.checkpoint=<path>` to load an explicit checkpoint.
- `--set visualization.video=true` to request Isaac Lab video capture.
- `--set visualization.rollout_steps=<n>` to extend or shorten the diagnostic rollout.
- `--set visualization.diagnostic_min_displacement_m=<value>` to adjust the pass/fail motion threshold.

Behavior note:

- On the baseline DETB ANYmal-C task family, `visualization.use_pretrained_checkpoint=true` resolves to the upstream Isaac Lab published ANYmal-C checkpoint.
- On DETB-only task families without a published upstream equivalent, requesting a pretrained checkpoint now fails explicitly instead of silently falling back to the latest local smoke run.

### `detb train-gui`

Builds the Isaac Lab GUI training command using the configured `execution.isaaclab_python` interpreter against the pinned `..\IsaacLab-5.1` checkout. By default it emits a launch spec without execution. Set `--set visualization.train_execute=true` to actually launch live training.

Useful overrides:

- `--set visualization.train_num_envs=<n>` to change live training scale.
- `--set visualization.train_max_iterations=<n>` to cap the interactive run.
- `--set visualization.video=true` to record training snippets through Isaac Lab.

## Task Selection

Use Hydra task groups to switch DETB-owned Isaac registry families:

```powershell
detb train --set task=flat_walk
detb train --set task=flat_walk_stability
detb train --set task=flat_walk_simple_actuator --set robot=anymal_c_simple_actuator
detb visualize --set task=flat_walk_stability --set visualization.execute=true
```

### `detb generate-requirements --source-dir <run_dir>`

Creates a candidate requirement ledger from prior run artifacts.
