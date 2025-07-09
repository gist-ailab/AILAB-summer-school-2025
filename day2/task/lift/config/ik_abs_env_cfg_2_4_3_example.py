# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka 로봇을 사용한 역기구학(IK) 제어 기반 리프팅 환경 설정

1. Franka Panda 로봇의 IK 제어 설정
2. Differential IK 제어기 설정
3. 엔드 이펙터 위치/방향 직접 제어
4. 조인트 위치 제어에서 IK 제어로의 변경
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트
# ============================================================================
from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

# 조인트 위치 제어 설정을 상속받기 위한 임포트
from task.lift.config import joint_pos_env_cfg_2_4_3_example

# ============================================================================
# 2. 미리 정의된 설정들 임포트
# ============================================================================
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  



# ============================================================================
# 3. IK 제어 기반 리프팅 환경 설정 클래스
# ============================================================================

@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg_2_4_3.FrankaCubeLiftEnvCfg):
    """
    Franka Panda 로봇과 DexCube를 사용한 IK 제어 기반 리프팅 환경
    
    joint_pos_env_cfg.FrankaCubeLiftEnvCfg를 상속받아 조인트 위치 제어를 IK 제어로 변경
    - 로봇: Franka Panda (IK 제어)
    - 오브젝트: DexCube (리프팅 대상)
    - 제어 방식: Differential IK 제어 (엔드 이펙터 위치/방향 직접 제어)
    """
    def __post_init__(self):
        """환경 설정 초기화 후 실행되는 메서드"""
        # 부모 클래스의 초기화 실행
        super().__post_init__()

        # ============================================================================
        # 4. Franka 로봇 설정 
        # ============================================================================
        # PD 제어기를 사용하는 Franka 설정으로 변경

        ############# Franka robot 설정 #############
        #### FRANKA_PANDA_HIGH_PD_CFG 사용 ####
        # replace() 메서드로 prim_path를 환경별로 고유하게 설정

        # 로봇의 초기 위치 설정 (테이블 위에 배치)

        ############################################

        # ============================================================================
        # 5. 액션 설정 (Differential IK 제어)
        # ============================================================================
        # 팔 액션: Differential IK 제어 (엔드 이펙터 위치/방향 직접 제어)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",  # 제어할 로봇 이름
            joint_names=["panda_joint.*"],  # 제어할 조인트들 (정규표현식)
            body_name="panda_hand",  # 제어할 body (엔드 이펙터)
            # Differential IK 제어기 설정
            controller=DifferentialIKControllerCfg(
                command_type="pose",  # 명령 타입: 위치 + 방향
                use_relative_mode=False,  # 절대 모드 사용 (상대 모드 아님)
                ik_method="dls"  # IK 해결 방법: Damped Least Squares
            ),
            # 엔드 이펙터 오프셋 설정 (그리퍼 중심점)
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.107)  # z축 방향 오프셋 (그리퍼 길이)
            )
        )
