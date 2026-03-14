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

## Environment

```powershell
conda activate isaaclab51
cd "C:\Dev Projects\NVIDIA Omniverse\IsaacLab Testbed"
python -m pip install -e .[dev]
```

## Isaac-Backed Execution

Keep `mock` for local repository validation. Use the simulator path only after the Isaac packages are importable in the active interpreter:

```powershell
python -m detb.cli train --set execution.backend=isaaclab
python -m detb.cli evaluate --set execution.backend=isaaclab
```

## Reproducibility Contract

Every run must persist:

- resolved config snapshot
- run manifest
- artifact registry
- raw episode metrics
- aggregate metrics
- report-ready markdown

The current scaffold enforces that contract for the `mock` backend and keeps the same artifact layout for the future Isaac path.
