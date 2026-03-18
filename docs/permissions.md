# Session Permissions

DETB is set up so the normal development loop stays inside this repo.

## What Changed

- Real Isaac-backed DETB runs now write logs under `outputs/isaaclab_logs/rsl_rl` inside this repository.
- That keeps train and evaluate artifacts in the writable project area instead of `..\IsaacLab-5.1\logs`.
- The practical result is that most work can be driven through one command family: `python -m detb.cli ...`.

## One-Time Approval Strategy

There is no global "approve everything" switch in this environment. The durable mechanism is prefix approval.

Recommended persistent approvals for long DETB sessions:

- `python -m detb.cli`
  - Covers train, evaluate, tune, terrain-eval, failure-eval, visualize, and train-gui.
- `python -m pytest`
  - Covers the local test loop.
- `git push`
  - Only needed when pushing work to GitHub.

## Working Rule

Prefer routing simulator work through DETB commands rather than calling Isaac Lab scripts directly. That keeps command execution, manifests, logs, and artifact paths consistent and minimizes approval churn.

## Typical Loop

```powershell
python -m detb.cli train --set execution.backend=isaaclab
python -m detb.cli evaluate --set execution.backend=isaaclab
python -m detb.cli visualize
python -m pytest -q
```
