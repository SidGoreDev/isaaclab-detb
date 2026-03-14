"""Filesystem helpers for DETB outputs."""

from __future__ import annotations

import csv
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


def capture_system_details() -> tuple[str, str, str]:
    return platform.system(), platform.version(), os.environ.get("DETB_GPU_MODEL", "unknown_gpu")


def create_manifest(
    cfg,
    command: str,
    run_id: str,
    run_dir: Path,
    checkpoint_path: Path,
) -> RunManifest:
    operating_system, driver_version, gpu_model = capture_system_details()
    return RunManifest(
        run_id=run_id,
        command=command,
        timestamp=utc_timestamp(),
        git_commit=git_commit(Path.cwd()),
        isaac_sim_version=str(cfg.execution.isaac_sim_version),
        isaaclab_version=str(cfg.execution.isaaclab_version),
        driver_version=driver_version,
        operating_system=operating_system,
        gpu_model=gpu_model,
        robot_variant=str(cfg.robot.name),
        task_family=str(cfg.task.family),
        terrain_name=str(cfg.terrain.name),
        sensor_profile=str(cfg.sensor.name),
        fault_profile=str(cfg.fault.name),
        seeds=[int(seed) for seed in cfg.execution.seeds],
        backend=str(cfg.execution.backend),
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
