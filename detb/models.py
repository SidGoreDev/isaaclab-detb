"""Core DETB data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RunManifest:
    run_id: str
    command: str
    timestamp: str
    git_commit: str
    isaac_sim_version: str
    isaaclab_version: str
    driver_version: str
    operating_system: str
    gpu_model: str
    robot_variant: str
    task_family: str
    terrain_name: str
    sensor_profile: str
    fault_profile: str
    seeds: list[int]
    backend: str
    checkpoint_path: str
    config_snapshot_path: str
    git_dirty: bool = False
    isaaclab_git_commit: str = "unknown"
    isaaclab_git_dirty: bool = False
    detb_lab_version: str = "unknown"
    torch_version: str = "unknown"
    cuda_version: str = "unknown"
    rsl_rl_version: str = "unknown"
    robot_asset_id: str = "unknown"
    robot_actuator_profile: str = "unknown"
    task_registry_id: str = ""
    run_tier: str = "smoke"
    configured_eval_episodes: int = 0
    configured_train_max_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    artifact_type: str
    relative_path: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeMetric:
    episode_id: str
    terrain_level: int
    terrain_name: str
    fault_level: float
    fault_name: str
    success: int
    distance_m: float
    elapsed_time_s: float
    energy_proxy: float
    failure_label: str
    seed: int
    sensor_profile: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetric:
    metric_name: str
    aggregation_scope: str
    mean: float
    median: float
    stddev: float
    ci_low: float
    ci_high: float
    n: int
    seed_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementRecord:
    req_id: str
    statement: str
    status: str
    source_run_id: str
    source_metric: str
    confidence_interval: str
    assumptions: str
    reviewer: str
    artifact_links: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
