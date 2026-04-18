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

## Deferred To Post-V1

Items identified during v1 gap-closure planning that are explicitly out of scope
for the v1 release. Recorded here so they do not rot. Line counts are snapshots
at 2026-04-18; a post-v1 audit PR should refresh them.

- **Modularize `detb/pipeline.py`.** The module is 832 lines with 23 top-level
  functions. Candidate split: `train_commands`, `eval_commands`,
  `analysis_commands`, shared helpers. Readable today; refactor for
  maintainability, not correctness.
- **Decompose `scripts/detb_isaaclab_play.py::main()`.** The function is 246
  lines combining rollout, telemetry collection, fault injection state, and
  video recording. Candidate extraction: rollout-loop helper, diagnostics
  builder. Works today; refactor for testability.
- **Isolated test coverage for experimental commands.** `train-gui`, `sweep`,
  `sensor-eval`, `terrain-eval`, `failure-eval`, and `tune` share a single
  bulk integration test (`tests/test_pipeline.py::test_analysis_commands_and_requirements`).
  Out-of-scope per "Out Of Scope For V1" above, but a regression in any one
  command currently surfaces as a generic test failure rather than a targeted
  signal.
