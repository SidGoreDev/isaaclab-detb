# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DETB configs for flat ANYmal-C locomotion with the simple actuator profile."""

from isaaclab.utils import configclass

from detb_lab.assets.robots.anymal_c import DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG
from detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import (
    DetbAnymalCFlatEnvCfg,
    DetbAnymalCFlatEnvCfg_PLAY,
)


@configclass
class DetbAnymalCSimpleActuatorFlatEnvCfg(DetbAnymalCFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class DetbAnymalCSimpleActuatorFlatEnvCfg_PLAY(DetbAnymalCFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
