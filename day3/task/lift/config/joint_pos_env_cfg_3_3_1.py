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
from task.lift.custom_pickplace_env_cfg_3_3_1 import YCBPickPlaceEnvCfg

# 미리 정의된 마커/로봇/카메라 config 불러오기
from isaaclab.markers.config import FRAME_MARKER_CFG  # 프레임(좌표계) 시각화 마커 설정
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # 프랑카 로봇 기본 config
from isaaclab.sim import PinholeCameraCfg               # 핀홀 카메라 모델 설정


@configclass
class FrankaYCBPickPlaceEnvCfg(YCBPickPlaceEnvCfg):
    """
    Franka Panda 로봇을 사용한 cube lifting RL 환경 설정.
    카메라, 엔드이펙터 프레임 등 센서/액터 세부 설정을 오버라이드함.
    """
    def __post_init__(self):
        # 부모 환경 config 초기화
        super().__post_init__()

        # 1. 로봇 모델을 Franka로 지정, prim_path도 변경
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. 액션 설정
        # 2-1. 로봇 팔(arm): 관절 위치 제어(Joint Position Control)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],      # 정규표현식으로 모든 panda_joint 대상
            scale=0.5,
            use_default_offset=True
        )
        # 2-2. 그리퍼(gripper): 바이너리(open/close) 제어
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],     # 모든 그리퍼 관절 대상
            open_command_expr={"panda_finger_.*": 0.04},    # 열 때 명령어
            close_command_expr={"panda_finger_.*": 0.0},    # 닫을 때 명령어
        )

        # 3. 오브젝트 명령(command) 설정 시, 엔드이펙터 바디 명칭 지정
        # self.commands.object_pose.body_name = "panda_hand"

        # 4. 카메라 센서 설정 (handeye 카메라: 프랑카 손 끝에 부착)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/handeye_camera",      # 카메라가 위치할 prim 경로
            update_period=0.1,      # 시뮬레이션 업데이트 간격(초)
            height=480, width=640,  # 해상도
            data_types=["rgb", "distance_to_image_plane"],      # RGB+Depth
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            # 카메라 위치/방향 오프셋 (ROS convention, Z축 90도 회전)
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.035, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678), convention="ros"),
        )


        # 5. 엔드이펙터(EE) 프레임(좌표계) 설정 및 프레임 시각화 마커 세팅
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)         # 시각화 마커 크기 조정
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",           # 로봇 base frame
            debug_vis=False,                                         # 디버그용 시각화 on/off
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",    # EE 프레임 대상
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),       # TCP 오프셋
                ),
            ],
        )


@configclass
class FrankaYCBPickPlaceEnvCfg_PLAY(FrankaYCBPickPlaceEnvCfg):
    """
    데모/테스트/시연(play)용 작은 환경 설정.
    - 환경 수 적고, 랜덤성/노이즈 없이 항상 같은 관측값 제공.
    """
    def __post_init__(self):
        # 부모 환경 config 초기화
        super().__post_init__()
        # 1. 환경 수와 spacing 축소
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 2. 관측 노이즈/랜덤화 비활성화
        self.observations.policy.enable_corruption = False