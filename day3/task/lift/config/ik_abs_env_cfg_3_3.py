# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


from task.lift.config import joint_pos_env_cfg_3_3

# 사전 정의된 Franka Panda High PD 세팅 import
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# 강체 들어올리는 환경
@configclass
class FrankaYCBPickPlaceEnvCfg(joint_pos_env_cfg_3_3.FrankaYCBPickPlaceEnvCfg):
    """
    Franka Panda 로봇이 큐브(또는 물체)를 들어올리는 RL 환경 Config 클래스
    (joint_pos_env_cfg_3_3.FrankaYCBPickPlaceEnvCfg 를 상속받아 사용)
    """
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 먼저 실행
        super().__post_init__()

        # 로봇 설정: PD 제어를 사용하는 Franka로 설정 (IK 성능 향상 목적)
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)   # 초기 로봇 위치
        
        # 액션(Action) 설정: Franka에 맞는 IK 기반 액션으로 세팅
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",                 # 대상 로봇 이름
            joint_names=["panda_joint.*"],      # 제어할 관절 이름(정규표현식)
            body_name="panda_hand",             # IK 기준 바디(엔드이펙터)
            controller=DifferentialIKControllerCfg(
                command_type="pose",            # pose 명령 타입
                use_relative_mode=False,        # 절대좌표로 명령
                ik_method="dls"                 # damped least squares IK 방법 사용
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107]           # 엔드이펙터와 tcp 오프셋 (z축 방향)
            ),         
        )


@configclass
class FrankaYCBPickPlaceEnvCfg_PLAY(FrankaYCBPickPlaceEnvCfg):
    """ 테스트/데모/인터랙티브 용도의 작은 환경 세팅 (학습이 아니라 play 모드) """
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 실행
        super().__post_init__()

        # 1. 환경 개수, spacing 축소 (더 가볍게)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # 2. 관측치 노이즈/랜덤화 비활성화 (실험/시연/디버깅 용도)
        self.observations.policy.enable_corruption = False
