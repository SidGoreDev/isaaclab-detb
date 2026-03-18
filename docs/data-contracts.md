# Data Contracts

DETB uses explicit persisted records so every run can be reconstructed from artifacts alone.

## RunManifest

Required fields:

- `run_id`
- `command`
- `timestamp`
- `git_commit`
- `git_dirty`
- `isaac_sim_version`
- `isaaclab_version`
- `isaaclab_git_commit`
- `isaaclab_git_dirty`
- `detb_lab_version`
- `torch_version`
- `cuda_version`
- `rsl_rl_version`
- `driver_version`
- `operating_system`
- `gpu_model`
- `robot_variant`
- `robot_asset_id`
- `robot_actuator_profile`
- `task_family`
- `task_registry_id`
- `terrain_name`
- `sensor_profile`
- `fault_profile`
- `seeds`
- `backend`
- `run_tier`
- `configured_eval_episodes`
- `configured_train_max_iterations`
- `checkpoint_path`
- `config_snapshot_path`

## ArtifactRecord

Required fields:

- `artifact_type`
- `relative_path`
- `description`

## EpisodeMetric

Required fields:

- `episode_id`
- `terrain_level`
- `terrain_name`
- `fault_level`
- `fault_name`
- `success`
- `distance_m`
- `elapsed_time_s`
- `energy_proxy`
- `failure_label`
- `seed`
- `sensor_profile`

## AggregateMetric

Required fields:

- `metric_name`
- `aggregation_scope`
- `mean`
- `median`
- `stddev`
- `ci_low`
- `ci_high`
- `n`
- `seed_count`

## RequirementRecord

Required fields:

- `req_id`
- `statement`
- `status`
- `source_run_id`
- `source_metric`
- `confidence_interval`
- `assumptions`
- `reviewer`
- `artifact_links`

## Rules

- Confidence intervals are mandatory for released aggregate metrics.
- `run_tier=study` requires the configured minimum seed and eval-episode counts before requirement generation can promote evidence.
- Candidate requirements default to `candidate` status.
- Requirement generation must operate from stored artifacts rather than hidden in-memory state.
- Artifact file names should remain stable across commands so they can be consumed by later analysis.
