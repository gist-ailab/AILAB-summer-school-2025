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
from task.lift.custom_lift_env_cfg_3_1 import LiftEnvCfg

# 미리 정의된 마커/로봇/카메라 config 불러오기
from isaaclab.markers.config import FRAME_MARKER_CFG  # 프레임(좌표계) 시각화 마커 설정
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # 프랑카 로봇 기본 config
from isaaclab.sim import PinholeCameraCfg               # 핀홀 카메라 모델 설정


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    """
    Franka Panda 로봇을 사용한 cube lifting 환경 설정.
    카메라, 엔드이펙터 프레임 등 센서/액터 세부 설정을 오버라이드함.
    """
    def __post_init__(self):
        # 부모 환경 config 초기화
        super().__post_init__()

        #Q4: 로봇 모델과 prim_path 를 지정하세요.
        # 로봇 모델을 Franka로 지정, prim_path도 변경
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        #Q5: Joint Position Control 액션을 위한 Isaac Lab 에서 제공하는 mdp 라이브러리의 actions cfg 설정
        # 액션 설정
        # 로봇 팔(arm): 관절 위치 제어(Joint Position Control)
        self.actions.arm_action = (
            asset_name="robot",
            joint_names=["panda_joint.*"],      # 정규표현식으로 모든 panda_joint 대상
            scale=0.5,
            use_default_offset=True
        )

        #Q6: 그리퍼 open/close 액션을 위한 Isaac Lab 에서 제공하는 mdp 라이브러리의 actions cfg 설정
        # 그리퍼(gripper): 바이너리(open/close) 제어
        self.actions.gripper_action = (
            asset_name="robot",
            joint_names=["panda_finger.*"],     # 모든 그리퍼 관절 대상
            open_command_expr={"panda_finger_.*": 0.04},    # 열 때 명령어
            close_command_expr={"panda_finger_.*": 0.0},    # 닫을 때 명령어
        )

        #Q7: 카메라 위치 종속관계를 위한 prim_path, 사용할 data_types(RGBD), prim_path 를 고려한 카메라 위치 offset 및 카메라 기준 좌표계에 대한 카메라 센서 설정을 하세요.
        # 카메라 센서 설정 (handeye 카메라: 프랑카 손 끝에 부착)
        self.scene.camera = CameraCfg(
            prim_path=,      # 카메라가 위치할 prim 경로
            update_period=0.1,      # 시뮬레이션 업데이트 간격(초)
            height=480, width=640,  # 해상도
            data_types=,      # RGB+Depth
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            # 카메라 위치/방향 오프셋 (ROS convention, Z축 90도 회전)
            offset=,
        )

        # 엔드이펙터(EE) 프레임(좌표계) 설정 및 프레임 시각화 마커 세팅
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)         # 시각화 마커 크기 조정
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",           # 로봇 base frame
            #Q8: 디버그용 시각화 on/off 설정해보시고, ee_frame 이 어떻게 정의되었는지도 체크해보세요.
            debug_vis=,                                         # 디버그용 시각화 on/off
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",    # EE 프레임 대상
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),       # TCP 오프셋
                ),
            ],
        )
