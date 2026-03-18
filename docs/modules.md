# Module Guide

## Baseline Execution

- `train` establishes a run manifest and training artifacts for either the `mock` backend or the real `isaaclab` backend.
- `evaluate` establishes episode and aggregate evidence for the current configuration and preserves Isaac run metadata when the real backend is selected.
- `bundle-artifacts` rebuilds review markdown from stored metrics.

## DETB-Owned Task Variants

- `task=flat_walk` selects the baseline DETB ANYmal-C flat/rough locomotion family.
- `task=flat_walk_stability` selects a stability-focused DETB family with narrower command ranges and stronger posture and smoothness penalties.
- `task=flat_walk_simple_actuator` selects the DETB family that runs the simple DC actuator profile.

Both task groups resolve to DETB-owned Isaac registry IDs, so the same selector works for `train`, `evaluate`, `visualize`, and `train-gui`.

## DETB-Owned Robot Variants

- `robot=anymal_c` selects the baseline DETB actuator-net profile.
- `robot=anymal_c_simple_actuator` selects the DETB simple DC actuator profile.

The simple-actuator robot group is currently paired with `task=flat_walk_simple_actuator` so the selected task family and robot asset remain consistent across backends.

## Study Modules

### Design Parameter Sweep

Uses `configs/study/*.yaml` to evaluate screening points over morphology and actuation parameters.

### Tuning

The tuning module ranks candidate configurations using weighted objectives and target thresholds. It is intended to keep DETB focused on productive outcomes instead of only generating plausible-looking numbers.

The main controls live under `objective` in `configs/base.yaml`:

- `success_weight`
- `terrain_weight`
- `robustness_weight`
- `energy_weight`
- `elapsed_weight`
- `target_success_rate`
- `target_tgs`
- `target_failure_threshold`
- `target_elapsed_time_s`

### Sensor Evaluation

Compares sensor profiles under matched terrain and seed settings. Outputs include task success, energy proxy, compute cost, approximate VRAM, and ROI score.

### Terrain Evaluation

Runs the configured terrain battery and computes Terrain Generalization Score from the per-terrain success curve.

### Failure Evaluation

Sweeps configured degradation levels and identifies the first critical threshold where success drops below 50 percent and confidence intervals no longer overlap that threshold.

### Requirement Generation

Builds candidate requirements from stored aggregate results and study outputs. These records are engineering prompts for review, not automatically approved requirements.

## Isaac Lab GUI Workflows

### Visualize

`visualize` delegates to the pinned Isaac Lab playback script through the configured `execution.isaaclab_python` interpreter so DETB can use the GUI path that already exists in the local Isaac Lab installation.

By default DETB writes `visualize_command.json` without launching the GUI. This keeps the exact command, task, device, and checkpoint selection reviewable before execution.

### Train GUI

`train-gui` delegates to the pinned Isaac Lab training script through the configured `execution.isaaclab_python` interpreter so DETB can show live simulator progress while training or recording video.

By default DETB writes `train_gui_command.json` without launching the GUI. Set `visualization.train_execute=true` only when you want to run the Isaac Lab window for real.
