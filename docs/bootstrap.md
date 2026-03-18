# DETB Bootstrap

## Source Of Truth

The parent `NVIDIA Omniverse` workspace already contains the validated Isaac baseline:

- `..\Engineering_Notebook.md`
- `..\IsaacLab Experiment 12_4_25\working_config_12_4_25.md`
- `..\IsaacLab-5.1\VERSION`

DETB standardizes on:

- Isaac Sim `5.1.0`
- Isaac Lab local checkout `..\IsaacLab-5.1`
- Isaac Lab version `2.3.0`
- Conda env `isaaclab51`
- Runtime device `cuda:0`
- GPU index `0`

## GPU Configuration

This machine exposes a single CUDA device and both `nvidia-smi` and the active PyTorch runtime resolve it as:

- `NVIDIA GeForce RTX 5080 Laptop GPU`
- driver `581.29`
- approximately `16 GB` VRAM

DETB now records detected GPU and driver metadata automatically in each run manifest. If a future machine has multiple GPUs, set the config fields in `configs/base.yaml` or override them from the CLI.

## Environment

```powershell
conda activate isaaclab51
cd "C:\Dev Projects\NVIDIA Omniverse\IsaacLab Testbed"
python -m pip install -e .[dev]
python -m pip install -e source/detb_lab
```

## Isaac Lab GUI Workflows

Keep `mock` for local repository validation. Use the pinned Isaac Lab runtime through the DETB GUI commands when you want live simulator visuals:

```powershell
detb visualize
detb train-gui
detb visualize --set visualization.execute=true
detb train-gui --set visualization.train_execute=true
```

These commands delegate through the configured `execution.isaaclab_python` interpreter against the pinned `..\IsaacLab-5.1` checkout and write a launch-spec JSON file even when execution is enabled.

## Minimal Real Smoke

Use the `isaaclab` backend when you want a small end-to-end real run that still lands in the normal DETB artifact layout:

```powershell
detb train --set execution.backend=isaaclab --set execution.num_envs=4 --set execution.train_max_iterations=1 --set execution.seeds=[11]
detb evaluate --set execution.backend=isaaclab --set execution.eval_episodes=2 --set execution.seeds=[11]
```

On March 17, 2026, both commands completed successfully on this machine and produced repo-local DETB artifacts plus repo-local Isaac Lab logs.

Use `execution.run_tier=smoke` for fast plumbing validation and `execution.run_tier=study` only when the config meets the evidence floor in `configs/analysis/default.yaml`.

## Tuning Workflow

Use the sweep study when you want meaningful ranking across multiple candidate designs:

```powershell
detb tune --set study=sweep
detb tune --set study=sweep --set objective.terrain_weight=0.5 --set objective.target_tgs=0.8
```

This keeps weight and threshold tuning explicit in the run artifacts instead of hidden in notebooks or ad hoc scripts.

## Reproducibility Contract

Every run must persist:

- resolved config snapshot
- run manifest
- artifact registry
- raw episode metrics
- aggregate metrics
- report-ready markdown

The current scaffold enforces that contract for the `mock` backend and now verifies the same artifact layout for minimal real `train` and `evaluate` runs on the `isaaclab` backend.
