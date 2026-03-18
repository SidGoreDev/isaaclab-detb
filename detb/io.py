"""Filesystem helpers for DETB outputs."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

from detb.config import config_to_builtin
from detb.evidence import configured_eval_episodes, run_tier
from detb.extension import detb_lab_version, robot_actuator_profile, robot_asset_id, task_registry_id
from detb.models import ArtifactRecord, RunManifest


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{utc_timestamp()}"


def resolve_output_root(output_root: str | os.PathLike[str]) -> Path:
    return Path(output_root).resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: object) -> None:
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(to_serializable(payload))), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[object]) -> None:
    records = [to_serializable(row) for row in rows]
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_serializable(payload: object) -> object:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, list):
        return [to_serializable(item) for item in payload]
    if isinstance(payload, tuple):
        return [to_serializable(item) for item in payload]
    if isinstance(payload, dict):
        return {str(key): to_serializable(value) for key, value in payload.items()}
    return payload


def git_commit(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_dirty(cwd: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _detect_gpu_from_nvidia_smi(gpu_index: int) -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines or gpu_index >= len(lines):
        return None
    parts = [part.strip() for part in lines[gpu_index].split(",", 1)]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _detect_gpu_from_torch(gpu_index: int) -> tuple[str, str] | None:
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        return None
    return torch.cuda.get_device_name(gpu_index), os.environ.get("DETB_DRIVER_VERSION", "unknown_driver")


def capture_system_details(gpu_index: int = 0) -> tuple[str, str, str]:
    operating_system = platform.system()
    gpu_model = os.environ.get("DETB_GPU_MODEL")
    driver_version = os.environ.get("DETB_DRIVER_VERSION")

    detected = _detect_gpu_from_nvidia_smi(gpu_index)
    if detected is None:
        detected = _detect_gpu_from_torch(gpu_index)

    if detected is not None:
        gpu_model, driver_version = detected

    if gpu_model is None:
        gpu_model = "unknown_gpu"
    if driver_version is None:
        driver_version = platform.version()

    return operating_system, driver_version, gpu_model


def _module_version(module_name: str, *, attr_name: str = "__version__", distribution_name: str | None = None) -> str:
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name, None)
        if value:
            return str(value)
    except Exception:
        pass
    try:
        return importlib.metadata.version(distribution_name or module_name)
    except Exception:
        return "unknown"


def _cuda_version() -> str:
    try:
        import torch
    except Exception:
        return "unknown"
    value = getattr(getattr(torch, "version", None), "cuda", None)
    return str(value or "unknown")


def create_manifest(
    cfg,
    command: str,
    run_id: str,
    run_dir: Path,
    checkpoint_path: Path,
) -> RunManifest:
    gpu_index = int(getattr(cfg.execution, "gpu_index", 0))
    operating_system, driver_version, gpu_model = capture_system_details(gpu_index=gpu_index)
    repo_cwd = Path.cwd()
    isaaclab_root = Path(str(getattr(cfg.execution, "isaaclab_root", repo_cwd)))
    return RunManifest(
        run_id=run_id,
        command=command,
        timestamp=utc_timestamp(),
        git_commit=git_commit(repo_cwd),
        git_dirty=git_dirty(repo_cwd),
        isaac_sim_version=str(cfg.execution.isaac_sim_version),
        isaaclab_version=str(cfg.execution.isaaclab_version),
        isaaclab_git_commit=git_commit(isaaclab_root),
        isaaclab_git_dirty=git_dirty(isaaclab_root),
        detb_lab_version=detb_lab_version(),
        torch_version=_module_version("torch"),
        cuda_version=_cuda_version(),
        rsl_rl_version=_module_version("rsl_rl", distribution_name="rsl-rl-lib"),
        driver_version=driver_version,
        operating_system=operating_system,
        gpu_model=gpu_model,
        robot_variant=str(cfg.robot.name),
        robot_asset_id=robot_asset_id(cfg),
        robot_actuator_profile=robot_actuator_profile(cfg),
        task_family=str(cfg.task.family),
        task_registry_id=task_registry_id(cfg),
        terrain_name=str(cfg.terrain.name),
        sensor_profile=str(cfg.sensor.name),
        fault_profile=str(cfg.fault.name),
        seeds=[int(seed) for seed in cfg.execution.seeds],
        backend=str(cfg.execution.backend),
        run_tier=run_tier(cfg),
        configured_eval_episodes=configured_eval_episodes(cfg),
        configured_train_max_iterations=int(cfg.execution.train_max_iterations),
        checkpoint_path=str(checkpoint_path),
        config_snapshot_path=str(run_dir / "resolved_config.yaml"),
    )


def write_manifest_bundle(
    cfg,
    manifest: RunManifest,
    run_dir: Path,
    artifacts: list[ArtifactRecord] | None = None,
) -> None:
    write_yaml(run_dir / "resolved_config.yaml", config_to_builtin(cfg))
    write_json(run_dir / "run_manifest.json", manifest)
    if artifacts is not None:
        write_json(run_dir / "artifact_registry.json", artifacts)
