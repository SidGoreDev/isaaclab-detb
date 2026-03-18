# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DETB configs for rough ANYmal-C locomotion with the simple actuator profile."""

from isaaclab.utils import configclass

from detb_lab.assets.robots.anymal_c import DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG
from detb_lab.tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    DetbAnymalCRoughEnvCfg,
    DetbAnymalCRoughEnvCfg_PLAY,
)


@configclass
class DetbAnymalCSimpleActuatorRoughEnvCfg(DetbAnymalCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class DetbAnymalCSimpleActuatorRoughEnvCfg_PLAY(DetbAnymalCRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DETB_ANYMAL_C_SIMPLE_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
