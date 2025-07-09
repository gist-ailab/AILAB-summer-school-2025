# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from task.lift.custom_lift_env_cfg_2_4_2 import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.sim import PinholeCameraCfg


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)


        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, use_default_offset=True
        )

        ############# Gripper action 설정 #############
        #### mdp.BinaryJointPositionActionCfg 사용 ####


        ##############################################


        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        ############# Cube object 생성 #############
        #### RigidObjectCfg, InitialStateCfg, UsdFileCfg, RigidBodyPropertiesCfg 사용 ####


        ###########################################


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Franka 로봇을 사용한 조인트 위치 제어 기반 리프팅 환경 설정

1. Franka Panda 로봇의 구체적인 설정
2. 조인트 위치 직접 제어 방식 설정
3. DexCube 오브젝트 설정
4. 엔드 이펙터 센서 설정
"""

# ============================================================================
# 1. 필요한 라이브러리 임포트
# ============================================================================
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
# 부모 클래스인 LiftEnvCfg 임포트
from task.lift.custom_lift_env_cfg_2_4_2_example import LiftEnvCfg

# ============================================================================
# 2. 미리 정의된 설정들 임포트
# ============================================================================
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.sim import PinholeCameraCfg

# ============================================================================
# 3. Franka 로봇 + 큐브 리프팅 환경 설정 클래스
# ============================================================================

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    """
    Franka Panda 로봇과 DexCube를 사용한 조인트 위치 제어 기반 리프팅 환경

    - 로봇: Franka Panda (조인트 위치 제어)
    - 오브젝트: DexCube (리프팅 대상)
    - 제어 방식: 조인트 위치 직접 제어
    """
    
    def __post_init__(self):
        """환경 설정 초기화 후 실행되는 메서드"""
        # 부모 클래스의 초기화 실행
        super().__post_init__()

        # ============================================================================
        # 4. Franka 로봇 설정
        # ============================================================================
        # Franka Panda 로봇을 씬에 추가
        # replace() 메서드로 prim_path를 환경별로 고유하게 설정
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 로봇의 초기 위치 설정 (테이블 위에 배치)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)

        # ============================================================================
        # 5. 액션 설정 (조인트 위치 제어)
        # ============================================================================
        # 팔 액션: 조인트 위치 직접 제어
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",  # 제어할 로봇 이름
            joint_names=["panda_joint.*"],  # 제어할 조인트들 (정규표현식)
            scale=1.0,  # 액션 스케일링
            use_default_offset=True  # 기본 오프셋 사용
        )


        ############# Gripper action 설정 #############
        #### mdp.BinaryJointPositionActionCfg 사용 ####


        ##############################################

      
        
        # 명령 설정: 엔드 이펙터 body 이름 지정
        self.commands.object_pose.body_name = "panda_hand"

        # ============================================================================
        # 6. DexCube 오브젝트 설정
        # ============================================================================
        ############# Cube object 생성 #############
        #### RigidObjectCfg, InitialStateCfg, UsdFileCfg, RigidBodyPropertiesCfg 사용 ####


        ###########################################
        

        # ============================================================================
        # 7. 엔드 이펙터 센서 설정
        # ============================================================================
        # 프레임 변환 센서 설정 (엔드 이펙터 위치 추적)
        marker_cfg = FRAME_MARKER_CFG.copy()  # 마커 설정 복사
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # 마커 크기 설정
        marker_cfg.prim_path = "/Visuals/FrameTransformer"  # 마커 경로
        
        # 엔드 이펙터 프레임 변환 센서 설정
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",  # 기준 프레임 (로봇 베이스)
            debug_vis=False,  # 디버그 시각화 비활성화
            visualizer_cfg=marker_cfg,  # 시각화 설정
            # 추적할 타겟 프레임들
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",  # 타겟 프레임 (엔드 이펙터)
                    name="end_effector",  # 프레임 이름
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),  # 오프셋 (엔드 이펙터 중심점)
                    ),
                ),
            ],
        )