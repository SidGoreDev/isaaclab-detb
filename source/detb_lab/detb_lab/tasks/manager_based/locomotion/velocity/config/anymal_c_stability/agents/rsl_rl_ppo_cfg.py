# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DETB-owned PPO configs for stability-focused ANYmal-C velocity."""

from isaaclab.utils import configclass

from detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    DetbAnymalCFlatPPORunnerCfg,
    DetbAnymalCRoughPPORunnerCfg,
)


@configclass
class DetbAnymalCStabilityRoughPPORunnerCfg(DetbAnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "detb_anymal_c_stability_rough"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.algorithm.learning_rate = 7.5e-4
        self.algorithm.entropy_coef = 0.003


@configclass
class DetbAnymalCStabilityFlatPPORunnerCfg(DetbAnymalCFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "detb_anymal_c_stability_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
        self.algorithm.learning_rate = 7.5e-4
        self.algorithm.entropy_coef = 0.003
