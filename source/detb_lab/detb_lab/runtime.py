"""Simulator-native helpers for DETB Lab tasks."""

from __future__ import annotations

from collections import deque

from detb_lab.registry import robot_spec_for_id, spec_for_task_id

BASELINE_BODY_MASS_KG = 32.0
BASELINE_LEG_SCALE = 1.0
BASELINE_STIFFNESS = 180.0
BASELINE_DAMPING = 8.0
SUPPORTED_SENSOR_PROFILES = {"proprio"}
SUPPORTED_FAULT_CLASSES = {"nominal", "torque_reduction", "latency"}


def load_task_cfgs(task_name: str, agent_entry_point: str = "rsl_rl_cfg_entry_point"):
    from detb_lab import register_all
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    register_all()
    task_id = task_name.split(":")[-1]
    env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task_id, agent_entry_point)
    return env_cfg, agent_cfg


def resolve_experiment_name(task_name: str, terrain_level: int = 0, override: str = "") -> str:
    requested = override.strip()
    if requested:
        return requested
    spec = spec_for_task_id(task_name)
    if spec is not None:
        return spec.flat_experiment_name if int(terrain_level) <= 0 else spec.rough_experiment_name
    if "Flat-Anymal-C" in task_name:
        return "detb_anymal_c_flat"
    if "Rough-Anymal-C" in task_name:
        return "detb_anymal_c_rough"
    return task_name.replace(":", "_").replace("/", "_").replace("-", "_").lower()


def validate_supported_configuration(
    sensor_name: str,
    task_name: str,
    leg_length_scale: float,
    stiffness: float,
    damping: float,
    fault_class: str | None = None,
    *,
    robot_asset_id: str = "detb.anymal_c",
    actuator_profile: str = "actuator_net",
) -> None:
    if sensor_name not in SUPPORTED_SENSOR_PROFILES:
        raise ValueError(
            "The real Isaac backend currently supports only the proprio sensor profile. "
            f"Received: {sensor_name}."
        )
    if abs(leg_length_scale - BASELINE_LEG_SCALE) > 1e-6:
        raise ValueError(
            "The real Isaac backend does not yet support leg-length overrides. "
            f"Received leg_length_scale={leg_length_scale}."
        )
    if fault_class is not None and fault_class not in SUPPORTED_FAULT_CLASSES:
        raise ValueError(f"Unsupported fault class for real Isaac evaluation: {fault_class}")

    task_spec = spec_for_task_id(task_name)
    if task_spec is not None and robot_asset_id != task_spec.robot_asset_id:
        raise ValueError(
            "The selected DETB task family expects robot asset "
            f"'{task_spec.robot_asset_id}', received '{robot_asset_id}'."
        )
    robot_spec = robot_spec_for_id(robot_asset_id)
    if robot_spec is None:
        raise ValueError(f"Unsupported robot asset for the real Isaac backend: {robot_asset_id}")
    if actuator_profile != robot_spec.actuator_profile:
        raise ValueError(
            "The real Isaac backend currently expects the robot actuator profile "
            f"'{robot_spec.actuator_profile}', received '{actuator_profile}'."
        )
    if abs(stiffness - robot_spec.baseline_stiffness) > 1e-6 or abs(damping - robot_spec.baseline_damping) > 1e-6:
        raise ValueError(
            "The real Isaac backend does not yet support stiffness or damping overrides for the "
            f"'{robot_spec.actuator_profile}' profile. Expected "
            f"stiffness={robot_spec.baseline_stiffness} and damping={robot_spec.baseline_damping}."
        )


def apply_common_overrides(
    env_cfg,
    agent_cfg,
    *,
    device: str,
    num_envs: int,
    seed: int,
    experiment_name: str,
    run_name: str,
    max_iterations: int | None = None,
) -> None:
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(seed)
    env_cfg.sim.device = str(device)
    agent_cfg.seed = int(seed)
    agent_cfg.device = str(device)
    agent_cfg.experiment_name = str(experiment_name)
    agent_cfg.run_name = str(run_name)
    if max_iterations is not None:
        agent_cfg.max_iterations = int(max_iterations)


def apply_sensor_profile(env_cfg, sensor_name: str) -> None:
    if sensor_name == "proprio":
        env_cfg.scene.height_scanner = None
        if hasattr(env_cfg.observations.policy, "height_scan"):
            env_cfg.observations.policy.height_scan = None
        return
    raise ValueError(f"Unsupported sensor profile: {sensor_name}")


def _set_subterrain_proportions(terrain_generator, weights: dict[str, float]) -> None:
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("Terrain weights must include at least one positive proportion.")
    for name, sub_terrain in terrain_generator.sub_terrains.items():
        raw = float(weights.get(name, 0.0))
        sub_terrain.proportion = raw / total


def apply_terrain_profile(env_cfg, terrain_name: str, terrain_level: int) -> None:
    normalized_name = terrain_name.lower()
    if int(terrain_level) <= 0 or "flat" in normalized_name:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.max_init_terrain_level = None
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
        return

    terrain_generator = env_cfg.scene.terrain.terrain_generator
    if terrain_generator is None:
        raise ValueError("Expected a terrain generator for non-flat Isaac terrain evaluation.")

    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.max_init_terrain_level = max(1, min(int(terrain_level), 5))
    terrain_generator.curriculum = False
    terrain_generator.use_cache = False
    terrain_generator.num_rows = max(int(getattr(terrain_generator, "num_rows", 1)), 1)
    terrain_generator.num_cols = max(int(getattr(terrain_generator, "num_cols", 1)), 1)

    if "slope" in normalized_name:
        _set_subterrain_proportions(
            terrain_generator,
            {
                "hf_pyramid_slope": 0.5,
                "hf_pyramid_slope_inv": 0.5,
            },
        )
    elif "stairs" in normalized_name:
        _set_subterrain_proportions(
            terrain_generator,
            {
                "pyramid_stairs": 0.5,
                "pyramid_stairs_inv": 0.5,
            },
        )
    elif "mixed" in normalized_name:
        _set_subterrain_proportions(
            terrain_generator,
            {
                "boxes": 0.35,
                "pyramid_stairs": 0.2,
                "pyramid_stairs_inv": 0.15,
                "random_rough": 0.15,
                "hf_pyramid_slope": 0.075,
                "hf_pyramid_slope_inv": 0.075,
            },
        )
    else:
        _set_subterrain_proportions(terrain_generator, {"random_rough": 1.0})


def _scale_cfg_value(value, scale: float):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: float(item) * scale for key, item in value.items()}
    return float(value) * scale


def _constant_like_cfg(value, constant: float):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: float(constant) for key in value.keys()}
    return float(constant)


def apply_robot_overrides(
    env_cfg,
    body_mass_kg: float,
    torque_limit_scale: float,
    stiffness: float,
    damping: float,
) -> None:
    mass_delta = float(body_mass_kg) - BASELINE_BODY_MASS_KG
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (mass_delta, mass_delta)

    for actuator in env_cfg.scene.robot.actuators.values():
        if hasattr(actuator, "effort_limit"):
            actuator.effort_limit = _scale_cfg_value(actuator.effort_limit, float(torque_limit_scale))
        if hasattr(actuator, "effort_limit_sim"):
            actuator.effort_limit_sim = _scale_cfg_value(actuator.effort_limit_sim, float(torque_limit_scale))
        if hasattr(actuator, "saturation_effort"):
            actuator.saturation_effort = _scale_cfg_value(actuator.saturation_effort, float(torque_limit_scale))
        if hasattr(actuator, "stiffness"):
            actuator.stiffness = _constant_like_cfg(actuator.stiffness, float(stiffness))
        if hasattr(actuator, "damping"):
            actuator.damping = _constant_like_cfg(actuator.damping, float(damping))


def prepare_cfgs(
    task_name: str,
    *,
    device: str,
    num_envs: int,
    seed: int,
    experiment_name: str,
    run_name: str,
    sensor_name: str,
    terrain_name: str,
    terrain_level: int,
    body_mass_kg: float,
    torque_limit_scale: float,
    leg_length_scale: float,
    stiffness: float,
    damping: float,
    max_iterations: int | None = None,
    robot_asset_id: str = "detb.anymal_c",
    actuator_profile: str = "actuator_net",
):
    validate_supported_configuration(
        sensor_name,
        task_name,
        leg_length_scale,
        stiffness,
        damping,
        robot_asset_id=robot_asset_id,
        actuator_profile=actuator_profile,
    )
    env_cfg, agent_cfg = load_task_cfgs(task_name)
    apply_common_overrides(
        env_cfg,
        agent_cfg,
        device=device,
        num_envs=num_envs,
        seed=seed,
        experiment_name=experiment_name,
        run_name=run_name,
        max_iterations=max_iterations,
    )
    apply_terrain_profile(env_cfg, terrain_name, terrain_level)
    apply_sensor_profile(env_cfg, sensor_name)
    apply_robot_overrides(env_cfg, body_mass_kg, torque_limit_scale, stiffness, damping)
    return env_cfg, agent_cfg


def fault_delay_steps(severity: float, latency_steps: int) -> int:
    if latency_steps <= 0 or severity <= 0:
        return 0
    scaled = int(round(float(latency_steps) * float(severity) / 0.2))
    return max(1, scaled)


def apply_fault_to_actions(actions, fault_class: str, severity: float, latency_steps: int, history: deque):
    if fault_class == "nominal" or severity <= 0:
        return actions, history
    if fault_class == "torque_reduction":
        return actions * max(0.0, 1.0 - float(severity)), history
    if fault_class == "latency":
        delay_steps = fault_delay_steps(severity, latency_steps)
        if delay_steps <= 0:
            return actions, history
        history.append(actions.clone())
        if len(history) <= delay_steps:
            return actions * 0.0, history
        delayed_actions = history.popleft()
        return delayed_actions, history
    raise ValueError(f"Unsupported fault class: {fault_class}")
