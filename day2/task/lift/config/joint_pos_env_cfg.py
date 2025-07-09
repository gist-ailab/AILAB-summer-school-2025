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
from task.lift.custom_lift_env_cfg import LiftEnvCfg

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
        
        # 그리퍼 액션: 이진 조인트 위치 제어 (열기/닫기)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",  # 제어할 로봇 이름
            joint_names=["panda_finger.*"],  # 그리퍼 조인트들
            open_command_expr={"panda_finger_.*": 0.04},  # 열린 상태 위치
            close_command_expr={"panda_finger_.*": 0.0},  # 닫힌 상태 위치
        )
        
        # 명령 설정: 엔드 이펙터 body 이름 지정
        self.commands.object_pose.body_name = "panda_hand"

        current_marker_cfg = FRAME_MARKER_CFG.copy()
        current_marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        current_marker_cfg.prim_path = "/Visuals/CurrentFrameTransformer"
        self.commands.object_pose.current_pose_marker_cfg = current_marker_cfg

        goal_marker_cfg = FRAME_MARKER_CFG.copy()
        goal_marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        goal_marker_cfg.prim_path = "/Visuals/GoalFrameTransformer"
        self.commands.object_pose.goal_pose_marker_cfg = goal_marker_cfg


        # ============================================================================
        # 6. DexCube 오브젝트 설정
        # ============================================================================
        # 리프팅할 큐브 오브젝트 설정
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",  # 오브젝트의 USD 경로
            # 초기 상태 설정
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0, 0.6),  # 초기 위치 (테이블 위)
                rot=(1, 0, 0, 0)    # 초기 회전 (쿼터니언)
            ),
            # 오브젝트 스폰 설정
            spawn=UsdFileCfg(
                # DexCube USD 파일 경로
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),  # 크기 스케일링
                # 물리적 속성 설정
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,  # 위치 솔버 반복 횟수
                    solver_velocity_iteration_count=1,   # 속도 솔버 반복 횟수
                    max_angular_velocity=1000.0,         # 최대 각속도
                    max_linear_velocity=1000.0,          # 최대 선속도
                    max_depenetration_velocity=5.0,      # 최대 분리 속도
                    disable_gravity=False,               # 중력 활성화
                ),
            ),
        )

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