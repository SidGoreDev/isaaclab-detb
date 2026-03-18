"""Helpers for the optional DETB Isaac Lab extension package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def detb_lab_root() -> Path:
    return repo_root() / "source" / "detb_lab"


def ensure_detb_lab_on_path() -> Path:
    root = detb_lab_root().resolve()
    if root.exists():
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
    return root


def _import_detb_lab():
    ensure_detb_lab_on_path()
    return importlib.import_module("detb_lab")


def _import_registry():
    ensure_detb_lab_on_path()
    return importlib.import_module("detb_lab.registry")


def register_detb_lab_tasks() -> None:
    package = _import_detb_lab()
    package.register_all()


def detb_lab_version() -> str:
    try:
        package = _import_detb_lab()
    except Exception:
        return "unknown"
    return str(getattr(package, "__version__", "unknown"))


def task_registry_id(cfg) -> str:
    registry_id = str(getattr(cfg.task, "registry_id", "")).strip()
    if registry_id:
        return registry_id
    return str(cfg.task.command)


def robot_asset_id(cfg) -> str:
    asset_id = str(getattr(cfg.robot, "asset_id", "")).strip()
    if asset_id:
        return asset_id
    return str(cfg.robot.name)


def robot_actuator_profile(cfg) -> str:
    return str(getattr(cfg.robot, "actuator_profile", "unknown"))


def task_spec_for_id(task_id: str):
    try:
        registry = _import_registry()
    except Exception:
        return None
    return registry.spec_for_task_id(task_id)


def robot_spec_for_id(asset_id: str):
    try:
        registry = _import_registry()
    except Exception:
        return None
    return registry.robot_spec_for_id(asset_id)


def expected_robot_asset_id_for_task(task_id: str) -> str:
    spec = task_spec_for_id(task_id)
    if spec is None:
        return ""
    return str(spec.robot_asset_id)


def resolve_train_task_id(cfg) -> str:
    return _resolve_task_variant(task_registry_id(cfg), terrain_level=int(cfg.terrain.level), play=False)


def resolve_play_task_id(cfg) -> str:
    return _resolve_task_variant(task_registry_id(cfg), terrain_level=int(cfg.terrain.level), play=True)


def experiment_name(cfg, task_id: str | None = None) -> str:
    requested = str(getattr(cfg.execution, "experiment_name", "")).strip()
    if requested:
        return requested
    resolved_task_id = task_id or resolve_train_task_id(cfg)
    spec = task_spec_for_id(resolved_task_id)
    if spec is not None:
        if int(cfg.terrain.level) <= 0:
            return spec.flat_experiment_name
        return spec.rough_experiment_name
    if "Flat-Anymal-C" in resolved_task_id:
        return "anymal_c_flat"
    if "Rough-Anymal-C" in resolved_task_id:
        return "anymal_c_rough"
    return resolved_task_id.replace(":", "_").replace("-", "_").lower()


def _resolve_task_variant(task_id: str, *, terrain_level: int, play: bool) -> str:
    spec = task_spec_for_id(task_id)
    if spec is not None:
        if terrain_level <= 0:
            return spec.flat_play_registry_id if play else spec.flat_train_registry_id
        return spec.rough_play_registry_id if play else spec.rough_train_registry_id

    resolved = task_id
    if terrain_level > 0 and "Flat-Anymal-C" in resolved:
        resolved = resolved.replace("Flat-Anymal-C", "Rough-Anymal-C")
    if play:
        if resolved.endswith("-Play-v0"):
            return resolved
        return resolved.replace("-v0", "-Play-v0")
    return resolved.replace("-Play-v0", "-v0")
