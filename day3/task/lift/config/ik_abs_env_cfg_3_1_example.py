# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


from task.lift.config import joint_pos_env_cfg_3_1

# 사전 정의된 Franka Panda High PD 세팅 import
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# 강체 들어올리는 환경
@configclass
#Q3: 로봇 position 제어를 위한 cfg class 상속하기.
class FrankaCubeLiftEnvCfg():
    """
    Franka Panda 로봇이 큐브(또는 물체)를 들어올리는 RL 환경 Config 클래스
    (joint_pos_env_cfg_3_1.FrankaCubeLiftEnvCfg 를 상속받아 사용)
    """
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 먼저 실행
        super().__post_init__()

        # 1. 로봇 설정: PD 제어를 사용하는 Franka로 설정 (IK 성능 향상 목적)
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)   # 초기 로봇 위치
        
        #Q9. Inverse Kinematics 액션을 위한 actions cfg 설정
        # 2. 액션(Action) 설정: Franka에 맞는 IK 기반 액션으로 세팅
        self.actions.arm_action = 
