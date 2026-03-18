"""Pure-Python registry metadata for DETB Lab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    key: str
    family: str
    flat_train_registry_id: str
    flat_play_registry_id: str
    rough_train_registry_id: str
    rough_play_registry_id: str
    flat_env_cfg_entry_point: str
    flat_play_env_cfg_entry_point: str
    rough_env_cfg_entry_point: str
    rough_play_env_cfg_entry_point: str
    flat_rsl_rl_cfg_entry_point: str
    rough_rsl_rl_cfg_entry_point: str
    flat_experiment_name: str
    rough_experiment_name: str
    robot_asset_id: str


@dataclass(frozen=True)
class RobotSpec:
    asset_id: str
    robot_name: str
    asset_reference: str
    actuator_profile: str
    cfg_entry_point: str
    baseline_stiffness: float
    baseline_damping: float


ANYMAL_C_FLAT_AGENT = (
    "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.agents."
    "rsl_rl_ppo_cfg:DetbAnymalCFlatPPORunnerCfg"
)
ANYMAL_C_ROUGH_AGENT = (
    "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.agents."
    "rsl_rl_ppo_cfg:DetbAnymalCRoughPPORunnerCfg"
)
ANYMAL_C_STABILITY_FLAT_AGENT = (
    "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability.agents."
    "rsl_rl_ppo_cfg:DetbAnymalCStabilityFlatPPORunnerCfg"
)
ANYMAL_C_STABILITY_ROUGH_AGENT = (
    "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability.agents."
    "rsl_rl_ppo_cfg:DetbAnymalCStabilityRoughPPORunnerCfg"
)
ANYMAL_C_SIMPLE_ACTUATOR_FLAT_AGENT = (
    "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator.agents."
    "rsl_rl_ppo_cfg:DetbAnymalCSimpleActuatorFlatPPORunnerCfg"
)
ANYMAL_C_SIMPLE_ACTUATOR_ROUGH_AGENT = (
    "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator.agents."
    "rsl_rl_ppo_cfg:DetbAnymalCSimpleActuatorRoughPPORunnerCfg"
)

TASK_SPECS: dict[str, TaskSpec] = {
    "velocity_anymal_c": TaskSpec(
        key="velocity_anymal_c",
        family="locomotion",
        flat_train_registry_id="DETB-Velocity-Flat-Anymal-C-v0",
        flat_play_registry_id="DETB-Velocity-Flat-Anymal-C-Play-v0",
        rough_train_registry_id="DETB-Velocity-Rough-Anymal-C-v0",
        rough_play_registry_id="DETB-Velocity-Rough-Anymal-C-Play-v0",
        flat_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg:"
            "DetbAnymalCFlatEnvCfg"
        ),
        flat_play_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg:"
            "DetbAnymalCFlatEnvCfg_PLAY"
        ),
        rough_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg:"
            "DetbAnymalCRoughEnvCfg"
        ),
        rough_play_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg:"
            "DetbAnymalCRoughEnvCfg_PLAY"
        ),
        flat_rsl_rl_cfg_entry_point=ANYMAL_C_FLAT_AGENT,
        rough_rsl_rl_cfg_entry_point=ANYMAL_C_ROUGH_AGENT,
        flat_experiment_name="detb_anymal_c_flat",
        rough_experiment_name="detb_anymal_c_rough",
        robot_asset_id="detb.anymal_c",
    ),
    "velocity_anymal_c_stability": TaskSpec(
        key="velocity_anymal_c_stability",
        family="locomotion",
        flat_train_registry_id="DETB-Velocity-Flat-Anymal-C-Stability-v0",
        flat_play_registry_id="DETB-Velocity-Flat-Anymal-C-Stability-Play-v0",
        rough_train_registry_id="DETB-Velocity-Rough-Anymal-C-Stability-v0",
        rough_play_registry_id="DETB-Velocity-Rough-Anymal-C-Stability-Play-v0",
        flat_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability.flat_env_cfg:"
            "DetbAnymalCStabilityFlatEnvCfg"
        ),
        flat_play_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability.flat_env_cfg:"
            "DetbAnymalCStabilityFlatEnvCfg_PLAY"
        ),
        rough_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability.rough_env_cfg:"
            "DetbAnymalCStabilityRoughEnvCfg"
        ),
        rough_play_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_stability.rough_env_cfg:"
            "DetbAnymalCStabilityRoughEnvCfg_PLAY"
        ),
        flat_rsl_rl_cfg_entry_point=ANYMAL_C_STABILITY_FLAT_AGENT,
        rough_rsl_rl_cfg_entry_point=ANYMAL_C_STABILITY_ROUGH_AGENT,
        flat_experiment_name="detb_anymal_c_stability_flat",
        rough_experiment_name="detb_anymal_c_stability_rough",
        robot_asset_id="detb.anymal_c",
    ),
    "velocity_anymal_c_simple_actuator": TaskSpec(
        key="velocity_anymal_c_simple_actuator",
        family="locomotion",
        flat_train_registry_id="DETB-Velocity-Flat-Anymal-C-SimpleActuator-v0",
        flat_play_registry_id="DETB-Velocity-Flat-Anymal-C-SimpleActuator-Play-v0",
        rough_train_registry_id="DETB-Velocity-Rough-Anymal-C-SimpleActuator-v0",
        rough_play_registry_id="DETB-Velocity-Rough-Anymal-C-SimpleActuator-Play-v0",
        flat_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator.flat_env_cfg:"
            "DetbAnymalCSimpleActuatorFlatEnvCfg"
        ),
        flat_play_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator.flat_env_cfg:"
            "DetbAnymalCSimpleActuatorFlatEnvCfg_PLAY"
        ),
        rough_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator.rough_env_cfg:"
            "DetbAnymalCSimpleActuatorRoughEnvCfg"
        ),
        rough_play_env_cfg_entry_point=(
            "detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c_simple_actuator.rough_env_cfg:"
            "DetbAnymalCSimpleActuatorRoughEnvCfg_PLAY"
        ),
        flat_rsl_rl_cfg_entry_point=ANYMAL_C_SIMPLE_ACTUATOR_FLAT_AGENT,
        rough_rsl_rl_cfg_entry_point=ANYMAL_C_SIMPLE_ACTUATOR_ROUGH_AGENT,
        flat_experiment_name="detb_anymal_c_simple_actuator_flat",
        rough_experiment_name="detb_anymal_c_simple_actuator_rough",
        robot_asset_id="detb.anymal_c_simple_actuator",
    )
}

ROBOT_SPECS: dict[str, RobotSpec] = {
    "detb.anymal_c": RobotSpec(
        asset_id="detb.anymal_c",
        robot_name="anymal_c",
        asset_reference="DETB ANYmal-C baseline",
        actuator_profile="actuator_net",
        cfg_entry_point="detb_lab.assets.robots.anymal_c:DETB_ANYMAL_C_CFG",
        baseline_stiffness=180.0,
        baseline_damping=8.0,
    ),
    "detb.anymal_c_simple_actuator": RobotSpec(
        asset_id="detb.anymal_c_simple_actuator",
        robot_name="anymal_c_simple_actuator",
        asset_reference="DETB ANYmal-C simple actuator profile",
        actuator_profile="dc_motor",
        cfg_entry_point="detb_lab.assets.robots.anymal_c:DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG",
        baseline_stiffness=40.0,
        baseline_damping=5.0,
    ),
}


def spec_for_task_id(task_id: str) -> TaskSpec | None:
    for spec in TASK_SPECS.values():
        if task_id in {
            spec.flat_train_registry_id,
            spec.flat_play_registry_id,
            spec.rough_train_registry_id,
            spec.rough_play_registry_id,
        }:
            return spec
    return None


def robot_spec_for_id(asset_id: str) -> RobotSpec | None:
    return ROBOT_SPECS.get(asset_id)


def register_all_tasks() -> None:
    import gymnasium as gym

    for spec in TASK_SPECS.values():
        _register_task(
            spec.flat_train_registry_id,
            spec.flat_env_cfg_entry_point,
            spec.flat_rsl_rl_cfg_entry_point,
        )
        _register_task(
            spec.flat_play_registry_id,
            spec.flat_play_env_cfg_entry_point,
            spec.flat_rsl_rl_cfg_entry_point,
        )
        _register_task(
            spec.rough_train_registry_id,
            spec.rough_env_cfg_entry_point,
            spec.rough_rsl_rl_cfg_entry_point,
        )
        _register_task(
            spec.rough_play_registry_id,
            spec.rough_play_env_cfg_entry_point,
            spec.rough_rsl_rl_cfg_entry_point,
        )


def _register_task(task_id: str, env_cfg_entry_point: str, rsl_rl_cfg_entry_point: str) -> None:
    import gymnasium as gym

    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
    )
