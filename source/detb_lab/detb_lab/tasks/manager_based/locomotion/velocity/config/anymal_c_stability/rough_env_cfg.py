# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DETB stability-focused configs for rough ANYmal-C velocity."""

from isaaclab.utils import configclass

from detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    DetbAnymalCRoughEnvCfg,
)


@configclass
class DetbAnymalCStabilityRoughEnvCfg(DetbAnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.08
        self.commands.base_velocity.heading_control_stiffness = 0.65
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        self.rewards.track_lin_vel_xy_exp.weight = 1.15
        self.rewards.track_ang_vel_z_exp.weight = 0.6
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_acc_l2.weight = -5.0e-7
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.flat_orientation_l2.weight = -7.5


@configclass
class DetbAnymalCStabilityRoughEnvCfg_PLAY(DetbAnymalCStabilityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
