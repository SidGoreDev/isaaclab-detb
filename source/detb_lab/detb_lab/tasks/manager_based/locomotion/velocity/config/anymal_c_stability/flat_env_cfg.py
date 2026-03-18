# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DETB stability-focused configs for flat ANYmal-C velocity."""

from isaaclab.utils import configclass

from .rough_env_cfg import DetbAnymalCStabilityRoughEnvCfg


@configclass
class DetbAnymalCStabilityFlatEnvCfg(DetbAnymalCStabilityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.35, 0.35)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.7, 0.7)

        self.rewards.track_lin_vel_xy_exp.weight = 1.2
        self.rewards.track_ang_vel_z_exp.weight = 0.65
        self.rewards.flat_orientation_l2.weight = -10.0
        self.rewards.dof_torques_l2.weight = -4.0e-5
        self.rewards.feet_air_time.weight = 0.0


@configclass
class DetbAnymalCStabilityFlatEnvCfg_PLAY(DetbAnymalCStabilityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
