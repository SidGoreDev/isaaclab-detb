from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
extension_root = repo_root / "source" / "detb_lab"

for path in (repo_root, extension_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from detb_lab.runtime import (  # noqa: E402
    BASELINE_BODY_MASS_KG,
    BASELINE_DAMPING,
    BASELINE_LEG_SCALE,
    BASELINE_STIFFNESS,
    SUPPORTED_FAULT_CLASSES,
    SUPPORTED_SENSOR_PROFILES,
    apply_common_overrides,
    apply_fault_to_actions,
    apply_robot_overrides,
    apply_sensor_profile,
    apply_terrain_profile,
    fault_delay_steps,
    load_task_cfgs,
    prepare_cfgs,
    resolve_pretrained_checkpoint_task_name,
    resolve_experiment_name,
    validate_supported_configuration,
)

__all__ = [
    "BASELINE_BODY_MASS_KG",
    "BASELINE_DAMPING",
    "BASELINE_LEG_SCALE",
    "BASELINE_STIFFNESS",
    "SUPPORTED_FAULT_CLASSES",
    "SUPPORTED_SENSOR_PROFILES",
    "apply_common_overrides",
    "apply_fault_to_actions",
    "apply_robot_overrides",
    "apply_sensor_profile",
    "apply_terrain_profile",
    "fault_delay_steps",
    "load_task_cfgs",
    "prepare_cfgs",
    "resolve_pretrained_checkpoint_task_name",
    "resolve_experiment_name",
    "validate_supported_configuration",
]
