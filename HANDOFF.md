# DETB Handoff

## Current State

- Standalone repo is working as a DETB scaffold with mock backend, docs, CI, and tests.
- Real Isaac-backed train/evaluate plumbing has been added.
- GUI flows already exist through DETB:
  - `detb visualize`
  - `detb train-gui`
- Default device is pinned to the local RTX 5080 path:
  - `execution.device: cuda:0`
  - `execution.gpu_index: 0`
- Local test suite is green:
  - `python -m pytest -q`
  - last result: `11 passed`

## Key Files Changed

- [configs/base.yaml](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/configs/base.yaml)
  - `execution.isaaclab_log_root` now points to `outputs/isaaclab_logs/rsl_rl`
  - `execution.isaaclab_python` points to `C:/Users/sidma/miniconda3/envs/isaaclab51/python.exe`
- [detb/backends/isaaclab_backend.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/detb/backends/isaaclab_backend.py)
  - DETB now launches real Isaac via direct `isaaclab51` Python, not `isaaclab.bat`
  - real `train()` and `evaluate()` subprocess bridge exists
  - reward-curve and checkpoint parsing exists
  - backend was monkey-patched at bottom of file to override train/eval/gui command builders
- [detb/pipeline.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/detb/pipeline.py)
  - runtime context injection added
  - Isaac train/eval artifact recording added
  - bottom-of-file overrides redefine `run_train` and `run_evaluate`
- [scripts/detb_isaaclab_common.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/scripts/detb_isaaclab_common.py)
  - shared config prep for Isaac-backed train/eval
  - terrain/sensor/robot/fault support rules
- [scripts/detb_isaaclab_train.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/scripts/detb_isaaclab_train.py)
  - DETB-owned Isaac training runner
  - now accepts `--log_root`
  - writes logs/checkpoints under repo-local `outputs/isaaclab_logs/rsl_rl/...`
- [scripts/detb_isaaclab_eval.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/scripts/detb_isaaclab_eval.py)
  - DETB-owned Isaac evaluation runner
  - emits DETB episode records
- [tests/test_isaaclab_backend.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/tests/test_isaaclab_backend.py)
  - updated for direct Python launcher path
  - new assertion protects `--log_root` passthrough
- [docs/permissions.md](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/docs/permissions.md)
  - notes about persistent approvals
- [scripts/dev_session.ps1](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/scripts/dev_session.ps1)
  - convenience wrapper for `python -m detb.cli`

## What Is Verified

- `isaaclab51` Python exists and can import Isaac Lab / Isaac Sim with CUDA available.
- DETB real backend launches Isaac using `cuda:0`.
- Interrupt-resilient evidence:
  - `outputs_smoke/train/train_20260317T044849Z`
    - first interrupted real smoke
  - `outputs_smoke/train/train_20260317T161027Z`
    - later real smoke after repo-local log-root fix
- Stdout from real train confirms actual Isaac startup:
  - AppLauncher uses `cuda:0`
  - flat Anymal-C task config loads
  - environment setup starts
- Repo-local Isaac log root is now active:
  - event file observed under
    - `outputs/isaaclab_logs/rsl_rl/anymal_c_flat/2026-03-17_12-10-40_detb_train_20260317T161027Z/`

## Current Blocker

- Real `train` did not return control within the 30-minute timeout even with:
  - `execution.num_envs=4`
  - `execution.train_max_iterations=1`
  - `execution.seeds=[11]`
- This means one of these is true:
  - Isaac training is still progressing but very slowly
  - training produced outputs but DETB is waiting on shutdown/cleanup
  - training is hung after startup
- We did **not** complete the next inspection because the follow-up file-list commands were interrupted during the permissions-heavy session.

## Most Important Next Steps In Yolo Mode

1. Inspect the repo-local Isaac run dir directly:

```powershell
Get-ChildItem "outputs/isaaclab_logs/rsl_rl/anymal_c_flat/2026-03-17_12-10-40_detb_train_20260317T161027Z" -Force
Get-ChildItem "outputs/isaaclab_logs/rsl_rl/anymal_c_flat/2026-03-17_12-10-40_detb_train_20260317T161027Z/params" -Force
```

2. Check whether model checkpoints were written:

```powershell
Get-ChildItem "outputs/isaaclab_logs/rsl_rl/anymal_c_flat/2026-03-17_12-10-40_detb_train_20260317T161027Z" -Filter "model_*.pt"
```

3. If checkpoints exist, inspect DETB train run dir and decide whether the backend only failed to finish result packaging:

```powershell
Get-ChildItem "outputs_smoke/train/train_20260317T161027Z" -Force
Get-Content "outputs_smoke/train/train_20260317T161027Z/isaac_train_stdout.log" -Tail 200
Get-Content "outputs_smoke/train/train_20260317T161027Z/isaac_train_stderr.log" -Tail 200
```

4. If training is producing checkpoints but not returning:
  - add more logging around `runner.learn(...)` completion in `scripts/detb_isaaclab_train.py`
  - log before and after `_latest_checkpoint(log_dir)`
  - log before writing `output_json`
  - rerun the same minimal train command

5. If no checkpoints exist:
  - inspect Isaac temp log referenced in stdout:
    - `C:\Users\sidma\AppData\Local\Temp\isaaclab_2026-03-17_12-10-40.log`
  - inspect GPU utilization during run
  - consider lowering to `num_envs=1`

6. Once real `train` succeeds, run minimal real `evaluate`:

```powershell
python -m detb.cli evaluate --set execution.backend=isaaclab --set execution.output_root=outputs_smoke --set execution.eval_episodes=2 --set execution.seeds=[11]
```

7. Then verify artifact parity:
  - `training_summary.json`
  - `training_reward_curve.csv`
  - copied checkpoint in DETB run dir
  - `episode_metrics.csv`
  - `aggregate_metrics.csv`
  - `aggregate_metrics.json`
  - `summary.md`

## Minimal Repro Command

```powershell
python -m detb.cli train --set execution.backend=isaaclab --set execution.output_root=outputs_smoke --set execution.num_envs=4 --set execution.train_max_iterations=1 --set execution.seeds=[11]
```

## Likely Cleanup After Yolo Restart

- Refactor bottom-of-file monkey-patch overrides in [detb/backends/isaaclab_backend.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/detb/backends/isaaclab_backend.py) into clean in-class methods.
- Refactor bottom-of-file overrides in [detb/pipeline.py](/C:/Dev%20Projects/NVIDIA%20Omniverse/IsaacLab%20Testbed/detb/pipeline.py) into proper in-place definitions.
- Add explicit backend test for end-to-end `train` result packaging once real train returns successfully.
- Update README/docs after real train/evaluate are confirmed, so public docs reflect actual backend status rather than the current in-progress bridge.

## Notes

- The permissions friction, not the core code path, was the main reason progress stalled in this session.
- The strongest positive signal is that the real Isaac run now starts correctly on the RTX 5080 and writes into the repo-local Isaac log root.
