# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DETB PPO configs for simple-actuator ANYmal-C locomotion."""

from isaaclab.utils import configclass

from detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    DetbAnymalCFlatPPORunnerCfg,
    DetbAnymalCRoughPPORunnerCfg,
)


@configclass
class DetbAnymalCSimpleActuatorRoughPPORunnerCfg(DetbAnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "detb_anymal_c_simple_actuator_rough"


@configclass
class DetbAnymalCSimpleActuatorFlatPPORunnerCfg(DetbAnymalCFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "detb_anymal_c_simple_actuator_flat"
