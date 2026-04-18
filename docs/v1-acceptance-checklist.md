# V1 Acceptance Checklist

This checklist defines the narrow DETB v1 release contract.

## Required

- `python -m pytest -q` passes on the supported Python version used in CI.
- `detb train --set execution.backend=mock` produces a run directory with `resolved_config.yaml`, `run_manifest.json`, `artifact_registry.json`, and a synthetic checkpoint.
- `detb evaluate --set execution.backend=mock` produces episode metrics, aggregate metrics, a summary, and an artifact registry entry set.
- `detb visualize` writes a launch spec without executing when `visualization.execute=false`.
- `detb generate-requirements --source-dir <run_dir>` works from stored artifacts only.
- The docs state that v1 support is centered on the mock-backed baseline and the baseline Isaac Lab path, not the experimental study commands.

## Required For A Release Cut When Isaac Is Available

- `detb train --set execution.backend=isaaclab --set task=flat_walk --set robot=anymal_c` completes and stores the real checkpoint plus Isaac snapshots.
- `detb evaluate --set execution.backend=isaaclab --set task=flat_walk --set robot=anymal_c` completes and stores real evaluation artifacts.
- `detb visualize --set execution.backend=isaaclab --set visualization.execute=true --set task=flat_walk --set robot=anymal_c` completes and stores playback artifacts.

## Out Of Scope For V1

- `train-gui`
- `sweep`
- `sensor-eval`
- `terrain-eval`
- `failure-eval`
- `tune`

## Exit Criteria

- The workflow in [`.github/workflows/ci.yml`](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/.github/workflows/ci.yml) stays green.
- The acceptance items above remain consistent with [README.md](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/README.md), [EXECUTION_GUIDE.md](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/EXECUTION_GUIDE.md), and [Operator_Instructions.md](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/Operator_Instructions.md).
